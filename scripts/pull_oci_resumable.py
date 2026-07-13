from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
CHUNK_SIZE = 8 * 1024 * 1024
REPORT_STEP = 256 * 1024 * 1024


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str
    tag: str


def parse_image_reference(value: str) -> ImageReference:
    prefix = "ghcr.io/"
    if not value.startswith(prefix):
        raise ValueError("Only public ghcr.io images are supported")

    remainder = value[len(prefix) :]
    if "@" in remainder:
        repository, reference = remainder.rsplit("@", 1)
        tag = reference.replace(":", "-")
    else:
        last_slash = remainder.rfind("/")
        last_colon = remainder.rfind(":")
        if last_colon > last_slash:
            repository, tag = remainder.rsplit(":", 1)
        else:
            repository, tag = remainder, "latest"
        reference = tag

    if not repository or "/" not in repository or not reference:
        raise ValueError(f"Invalid GHCR image reference: {value}")
    return ImageReference("ghcr.io", repository.lower(), reference, tag)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def blob_path(layout_dir: Path, digest: str) -> Path:
    algorithm, hex_digest = digest.split(":", 1)
    if algorithm != "sha256" or len(hex_digest) != 64:
        raise ValueError(f"Unsupported blob digest: {digest}")
    return layout_dir / "blobs" / algorithm / hex_digest


def blob_is_valid(path: Path, digest: str, expected_size: int) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == digest.split(":", 1)[1]
    )


def get_token(image: ImageReference, timeout: int) -> str:
    query = urllib.parse.urlencode(
        {
            "service": image.registry,
            "scope": f"repository:{image.repository}:pull",
        }
    )
    with urllib.request.urlopen(
        f"https://{image.registry}/token?{query}", timeout=timeout
    ) as response:
        payload = json.load(response)
    token = payload.get("token") or payload.get("access_token")
    if not token:
        raise RuntimeError("GHCR did not return an anonymous pull token")
    return str(token)


def registry_request(
    image: ImageReference,
    path: str,
    *,
    accept: str | None = None,
    range_start: int | None = None,
    timeout: int,
):
    token = get_token(image, timeout)
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "voice-tts-colab/1"}
    if accept:
        headers["Accept"] = accept
    if range_start is not None:
        headers["Range"] = f"bytes={range_start}-"
    request = urllib.request.Request(
        f"https://{image.registry}/v2/{image.repository}/{path}",
        headers=headers,
    )
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_manifest(
    image: ImageReference, reference: str, *, timeout: int
) -> tuple[bytes, str]:
    encoded_reference = urllib.parse.quote(reference, safe=":")
    with registry_request(
        image,
        f"manifests/{encoded_reference}",
        accept=MANIFEST_ACCEPT,
        timeout=timeout,
    ) as response:
        raw = response.read()
        media_type = response.headers.get_content_type()
    return raw, media_type


def select_platform_manifest(index: dict[str, Any], os_name: str, architecture: str) -> dict[str, Any]:
    for descriptor in index.get("manifests", []):
        platform = descriptor.get("platform") or {}
        if platform.get("os") == os_name and platform.get("architecture") == architecture:
            return descriptor
    raise RuntimeError(f"Image does not contain platform {os_name}/{architecture}")


def write_verified_blob(layout_dir: Path, digest: str, raw: bytes) -> Path:
    expected = digest.split(":", 1)[1]
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Metadata digest mismatch for {digest}: got sha256:{actual}")
    target = blob_path(layout_dir, digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return target


def download_blob(
    image: ImageReference,
    descriptor: dict[str, Any],
    layout_dir: Path,
    *,
    attempts: int,
    retry_delay: float,
    timeout: int,
) -> Path:
    digest = str(descriptor["digest"])
    expected_size = int(descriptor["size"])
    target = blob_path(layout_dir, digest)
    target.parent.mkdir(parents=True, exist_ok=True)

    if blob_is_valid(target, digest, expected_size):
        print(f"Cached {digest[:19]} ({expected_size / (1024 ** 3):.2f} GiB)", flush=True)
        return target
    if target.exists():
        target.unlink()

    partial = target.with_name(target.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == expected_size:
            if blob_is_valid(partial, digest, expected_size):
                os.replace(partial, target)
                return target
            partial.unlink()
            existing = 0

        try:
            with registry_request(
                image,
                f"blobs/{digest}",
                range_start=existing if existing else None,
                timeout=timeout,
            ) as response:
                status = getattr(response, "status", response.getcode())
                if existing and status == 200:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("Registry ignored HTTP Range; restarting this blob")
                if existing and status != 206:
                    raise RuntimeError(f"Expected HTTP 206 while resuming, got {status}")

                downloaded = existing
                next_report = max(REPORT_STEP, ((existing // REPORT_STEP) + 1) * REPORT_STEP)
                mode = "ab" if existing else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(CHUNK_SIZE):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report or downloaded == expected_size:
                            print(
                                f"{digest[:19]}: {downloaded / (1024 ** 3):.2f}/"
                                f"{expected_size / (1024 ** 3):.2f} GiB",
                                flush=True,
                            )
                            next_report += REPORT_STEP

            if downloaded != expected_size:
                raise IOError(f"Unexpected EOF at {downloaded} of {expected_size} bytes")
            if not blob_is_valid(partial, digest, expected_size):
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 verification failed for {digest}")
            os.replace(partial, target)
            print(f"Verified {digest[:19]}", flush=True)
            return target
        except (
            http.client.IncompleteRead,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
            RuntimeError,
        ) as exc:
            last_error = exc
            current = partial.stat().st_size if partial.exists() else 0
            print(
                f"Retry {attempt}/{attempts} for {digest[:19]} from "
                f"{current / (1024 ** 3):.2f} GiB: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(retry_delay * min(attempt, 5))

    raise RuntimeError(f"Failed to download {digest} after {attempts} attempts") from last_error


def build_oci_archive(
    image_name: str,
    layout_dir: Path,
    output_path: Path,
    *,
    os_name: str,
    architecture: str,
    attempts: int,
    retry_delay: float,
    timeout: int,
) -> None:
    image = parse_image_reference(image_name)
    layout_dir.mkdir(parents=True, exist_ok=True)

    top_raw, top_media_type = fetch_manifest(image, image.reference, timeout=timeout)
    top_payload = json.loads(top_raw)
    if top_media_type in INDEX_MEDIA_TYPES or "manifests" in top_payload:
        selected = select_platform_manifest(top_payload, os_name, architecture)
        manifest_digest = str(selected["digest"])
        manifest_raw, manifest_media_type = fetch_manifest(image, manifest_digest, timeout=timeout)
        platform = selected.get("platform") or {"os": os_name, "architecture": architecture}
    else:
        manifest_raw = top_raw
        manifest_media_type = top_media_type
        manifest_digest = "sha256:" + hashlib.sha256(manifest_raw).hexdigest()
        platform = {"os": os_name, "architecture": architecture}

    write_verified_blob(layout_dir, manifest_digest, manifest_raw)
    manifest = json.loads(manifest_raw)
    descriptors = [manifest["config"], *manifest.get("layers", [])]
    for descriptor in descriptors:
        download_blob(
            image,
            descriptor,
            layout_dir,
            attempts=attempts,
            retry_delay=retry_delay,
            timeout=timeout,
        )

    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "mediaType": manifest_media_type,
                "digest": manifest_digest,
                "size": len(manifest_raw),
                "annotations": {"org.opencontainers.image.ref.name": image.tag},
                "platform": platform,
            }
        ],
    }
    (layout_dir / "oci-layout").write_text(
        json.dumps({"imageLayoutVersion": "1.0.0"}) + "\n", encoding="utf-8"
    )
    (layout_dir / "index.json").write_text(
        json.dumps(index, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    print(f"Creating OCI archive: {output_path}", flush=True)
    subprocess.run(
        [
            "tar",
            "--format=posix",
            "-C",
            str(layout_dir),
            "-cf",
            str(output_path),
            "oci-layout",
            "index.json",
            "blobs",
        ],
        check=True,
    )
    print(f"OCI archive ready: {output_path} ({output_path.stat().st_size / (1024 ** 3):.2f} GiB)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumably download a public GHCR image and build an OCI archive"
    )
    parser.add_argument("image")
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--os", default="linux")
    parser.add_argument("--architecture", default="amd64")
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    build_oci_archive(
        args.image,
        args.layout_dir,
        args.output,
        os_name=args.os,
        architecture=args.architecture,
        attempts=args.attempts,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
