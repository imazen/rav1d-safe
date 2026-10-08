#!/usr/bin/env python3
"""Check that the corpus runner preserves flags and refuses vacuous success."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConformanceRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.workspace.name)
        self.capture = self.directory / "arguments.json"
        self.binary = self.directory / "decoder"
        self.binary.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "sys.exit(int(os.environ.get('DECODER_EXIT', '0')))\n"
        )
        self.binary.chmod(0o755)
        self.vector = self.directory / "vector with spaces.obu"
        self.vector.write_bytes(b"fixture")
        self.manifest = self.directory / "vectors.tsv"
        self.header = "bitdepth\tcategory\ttest_name\tfile_path\texpected_md5\tfilmgrain\textra_args\n"

    def tearDown(self):
        self.workspace.cleanup()

    def write_manifest(self, path=None, extra=""):
        self.manifest.write_text(
            self.header + f"8-bit\tdata\tfixture\t{path or self.vector}\t{'0' * 32}\t1\t{extra}\n"
        )

    def run_gate(self, *arguments, exit_code=0):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/conformance_test.sh"),
             "--binary", str(self.binary), "--manifest", str(self.manifest),
             *arguments],
            capture_output=True, text=True,
            env=dict(os.environ, CAPTURE=str(self.capture), DECODER_EXIT=str(exit_code)),
        )

    def test_empty_manifest_fails_without_invoking_decoder(self):
        self.manifest.write_text(self.header)
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No vectors selected", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_missing_vector_fails_without_invoking_decoder(self):
        self.write_manifest(self.directory / "missing.obu")
        result = self.run_gate("--expected", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MISSING:", result.stdout)
        self.assertFalse(self.capture.exists())

    def test_decoder_failure_reaches_caller(self):
        self.write_manifest()
        result = self.run_gate(exit_code=17)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fail=1", result.stdout)
        self.assertIn("decoder exit=17", result.stdout)

    def test_timeout_status_reaches_caller(self):
        self.write_manifest()
        result = self.run_gate(exit_code=124)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("decoder exit=124", result.stdout)
        self.assertIn("deadline=120s", result.stdout)

    def test_incomplete_selection_fails(self):
        self.write_manifest()
        result = self.run_gate("--expected", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected 2 vectors, selected 1", result.stderr)

    def test_runtime_tier_threads_and_decode_modes_are_preserved(self):
        self.write_manifest(extra="--oppoint 2 --alllayers 0 --decodeframetype key --limit 3")
        result = self.run_gate("--level", "scalar", "--threads", "4", "--delay", "2", "--expected", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(self.capture.read_text()), [
            "-q", "--level", "scalar", "--threads", "4", "--delay", "2",
            "--filmgrain", "--oppoint", "2", "--alllayers", "0",
            "--decodeframetype", "key", "--limit", "3", str(self.vector), "0" * 32,
        ])

    def test_default_extraction_handles_excluded_sanitizer_arrays(self):
        corpus = self.directory / "corpus"
        regular = corpus / "8-bit" / "issues"
        fuzz = corpus / "oss-fuzz"
        regular.mkdir(parents=True)
        fuzz.mkdir()
        (regular / "vector.obu").write_bytes(b"fixture")
        (regular / "meson.build").write_text(
            "tests += [['fixture', files('vector.obu'), '" + "0" * 32 + "']]\n"
            "test('zsvc', dav1d, args: [files('vector.obu'), '--filmgrain', "
            "'--oppoint', '2', '--alllayers', '0', '--decodeframetype', 'key', "
            "'--limit=3', '--verify', '" + "0" * 32 + "'])\n"
        )
        # The caller excludes OSS-fuzz by default. Its array rows still need
        # a valid TSV shape before that selection can happen.
        (fuzz / "meson.build").write_text("asan_tests = ['42']\n")
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/conformance_test.sh"),
             "--binary", str(self.binary), "--threads", "4", "--expected", "2"],
            capture_output=True, text=True,
            env=dict(os.environ, DAV1D_TEST_DATA_DIR=str(corpus),
                     CAPTURE=str(self.capture)),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("total=2 pass=2 fail=0 errors=0 excluded=1", result.stdout)
        self.assertEqual(json.loads(self.capture.read_text()), [
            "-q", "--level", "native", "--threads", "4", "--delay", "0",
            "--filmgrain", "--oppoint=2", "--alllayers=0",
            "--decodeframetype=key", "--limit=3",
            str(regular / "vector.obu"), "0" * 32,
        ])


if __name__ == "__main__":
    unittest.main()
