"""The GPU test runner must preserve pytest's failure through tee."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class GpuRunnerTest(unittest.TestCase):
    def test_pytest_failure_is_not_reported_as_success(self):
        script = Path(__file__).resolve().parents[1] / "run-gpu-tests.sh"
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python3"
            python.write_text(
                '#!/bin/bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\n'
                'echo "=== FAILED ==="\nexit 7\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(script)],
                cwd=tmp,
                env={**os.environ, "PATH": tmp + os.pathsep + os.environ["PATH"]},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
