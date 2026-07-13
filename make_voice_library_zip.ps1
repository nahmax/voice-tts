$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "local_voice_library"
$Zip = Join-Path $Root "local_voice_library_upload.zip"

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Missing local voice library: $Source"
}

if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}

Compress-Archive -Path (Join-Path $Source "*") -DestinationPath $Zip -Force

Write-Host "Created $Zip"
Write-Host "Upload this ZIP in voice_tts_colab_gpu.ipynb cell 4a when IMPORT_LIBRARY_ZIP = True."
