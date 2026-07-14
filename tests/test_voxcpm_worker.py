import unittest

from scripts.voxcpm_worker import _is_transient_model_download_error


class VoxCPMWorkerTests(unittest.TestCase):
    def test_hugging_face_signed_url_error_is_retryable(self):
        error = RuntimeError("403 Forbidden: Auth failed: SignatureError: invalid key pair id")
        self.assertTrue(_is_transient_model_download_error(error))

    def test_model_configuration_error_is_not_retryable(self):
        self.assertFalse(_is_transient_model_download_error(ValueError("invalid model config")))


if __name__ == "__main__":
    unittest.main()
