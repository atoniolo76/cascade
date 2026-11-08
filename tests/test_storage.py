import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cascade_profiler.storage import s3_destination, upload_run


class StorageTest(unittest.TestCase):
    def test_empty_prefix_and_validation(self):
        with patch("boto3.client"):
            destination, metadata = s3_destination({"bucket": "runs", "prefix": ""}, "fixture")
        self.assertEqual(destination.uri(), "s3://runs/fixture")
        self.assertEqual(metadata["prefix"], "")
        for invalid in ({}, {"bucket": "runs", "aws_secret_access_key": "not-a-real-key"},
                        {"bucket": "runs", "endpoint": "https://user:password@example.com"},
                        {"bucket": "runs", "endpoint": "https://example.com?token=secret"},
                        {"bucket": "runs", "addressing_style": "invalid"}):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                s3_destination(invalid, "fixture")

    def test_upload_failure_keeps_local_run(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            run = Path(directory)
            (run / "trace.txt").write_text("trace")
            manifest = {"status": "completed", "s3_upload_status": "pending"}
            destination = Mock()
            destination.upload.side_effect = RuntimeError("upload failed")
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                upload_run(run, destination, manifest)
            self.assertEqual(json.loads((run / "run.json").read_text())["s3_upload_status"], "failed")
            self.assertTrue((run / "trace.txt").is_file())


if __name__ == "__main__":
    unittest.main()
