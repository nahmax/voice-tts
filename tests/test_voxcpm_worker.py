import ast
import inspect
import textwrap
import unittest

from scripts.voxcpm_worker import _is_transient_model_download_error, synthesize


class VoxCPMWorkerTests(unittest.TestCase):
    def test_hugging_face_signed_url_error_is_retryable(self):
        error = RuntimeError("403 Forbidden: Auth failed: SignatureError: invalid key pair id")
        self.assertTrue(_is_transient_model_download_error(error))

    def test_model_configuration_error_is_not_retryable(self):
        self.assertFalse(_is_transient_model_download_error(ValueError("invalid model config")))

    def test_seed_is_applied_outside_voxcpm_generate_call(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(synthesize)))
        generate_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ]
        self.assertEqual(len(generate_calls), 1)
        self.assertNotIn("seed", {keyword.arg for keyword in generate_calls[0].keywords})
        called_methods = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("manual_seed", called_methods)
        self.assertIn("manual_seed_all", called_methods)


if __name__ == "__main__":
    unittest.main()
