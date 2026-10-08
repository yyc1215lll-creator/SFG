"""Dependency-free release checks; no GPU or model download is performed."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from defaults import IMAGE_DEFAULTS, VIDEO_DEFAULTS


def cli(script, *arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout


class ReleaseTests(unittest.TestCase):
    def test_python_syntax(self):
        paths = list(ROOT.glob("*.py"))
        for directory in ("image", "video", "tests"):
            paths.extend((ROOT / directory).rglob("*.py"))
        self.assertGreaterEqual(len(paths), 53)
        for path in paths:
            with self.subTest(source=str(path.relative_to(ROOT))):
                source = path.read_bytes()
                ast.parse(source, filename=str(path))
                compile(source, str(path), "exec")

    def test_image_defaults(self):
        self.assertEqual(len(IMAGE_DEFAULTS), 4)
        for model, expected in IMAGE_DEFAULTS.items():
            with self.subTest(model=model):
                result = json.loads(cli("run_image.py", "--model", model, "--print-config"))
                self.assertEqual(result.pop("model"), model)
                self.assertFalse(result.pop("baseline"))
                self.assertEqual(result, expected)

    def test_video_defaults(self):
        self.assertEqual(len(VIDEO_DEFAULTS), 3)
        for task, expected in VIDEO_DEFAULTS.items():
            with self.subTest(task=task):
                result = json.loads(cli("run_video.py", "--task", task, "--print-config"))
                self.assertEqual(result.pop("task"), task)
                self.assertEqual(result, expected)

    def test_image_overrides_and_baseline_guidance(self):
        result = json.loads(cli(
            "run_image.py", "--model", "flux-dev", "--baseline",
            "--guidance", "3.5", "--u-s", "0", "--u-x", "-0.25",
            "--omega", "8", "--start-step", "2", "--end-step", "10",
            "--print-config",
        ))
        self.assertTrue(result["baseline"])
        for key, value in dict(guidance=3.5, u_s=0, u_x=-0.25, omega=8,
                               start_step=2, end_step=10).items():
            self.assertEqual(result[key], value)
        # The baseline flag disables the branch; it does not retune embedded g.
        default_baseline = json.loads(cli(
            "run_image.py", "--model", "flux-dev", "--baseline", "--print-config"))
        self.assertEqual(default_baseline["guidance"], 1.0)

    def test_video_baseline(self):
        for task in VIDEO_DEFAULTS:
            with self.subTest(task=task):
                result = json.loads(cli(
                    "run_video.py", "--task", task, "--baseline",
                    "--guidance", "6", "--print-config"))
                self.assertEqual(result["guidance"], 6)
                for key in ("u_s", "u_x", "omega", "active_steps"):
                    self.assertEqual(result[key], 0)

    def test_manifest_creation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="sfg-release-test-") as directory:
            directory = Path(directory)
            target = directory / "t2v.jsonl"
            cli("prepare_video_manifest.py", "--prompt", "A boat on a lake.",
                "--seed", "43", "--output", str(target))
            row = json.loads(target.read_text())
            self.assertEqual(row, dict(prompt="A boat on a lake.", seed=43,
                                       relative_output="sample-0.mp4"))
            with self.assertRaises(subprocess.CalledProcessError):
                cli("prepare_video_manifest.py", "--prompt", "Do not overwrite.",
                    "--output", str(target))
            # The helper hashes supplied bytes; it does not decode the image.
            reference = directory / "reference.png"
            reference.write_bytes(b"manifest-hashing-fixture")
            output = directory / "i2v.jsonl"
            cli("prepare_video_manifest.py", "--prompt", "The subject moves.",
                "--reference-image", str(reference), "--output", str(output))
            row = json.loads(output.read_text())
            self.assertEqual(row["reference_image"], str(reference.resolve()))
            self.assertEqual(row["reference_image_sha256"],
                             hashlib.sha256(reference.read_bytes()).hexdigest())

    def test_default_parameter_values(self):
        image_values = {
            "sd3m": (40, 1.0, 0.15, -0.40, 6.0, 1, 0),
            "sd35m": (40, 7.5, 0.25, -0.25, 3.5, 1, 20),
            "flux-dev": (28, 1.0, 0.35, -0.35, 14.0, 1, 0),
            "flux-de-distill": (28, 3.5, 0.35, -0.45, 7.5, 1, 0),
        }
        keys = ("steps", "guidance", "u_s", "u_x", "omega", "start_step", "end_step")
        for model, expected in image_values.items():
            self.assertEqual(tuple(IMAGE_DEFAULTS[model][key] for key in keys), expected)
        video_values = {
            "t2v": (25, 1.0, 0.25, -0.25, 4.0, 7, "joint_encoder"),
            "i2v": (25, 1.0, 0.275, -0.275, 6.0, 7, "joint_encoder"),
            "i2v-text": (25, 1.0, 0.20, -0.20, 3.0, 7, "text_only"),
        }
        keys = ("steps", "guidance", "u_s", "u_x", "omega", "active_steps", "condition_scope")
        for task, expected in video_values.items():
            self.assertEqual(tuple(VIDEO_DEFAULTS[task][key] for key in keys), expected)

    def test_cli_help(self):
        for script in ("run_image.py", "run_video.py"):
            with self.subTest(script=script):
                self.assertIn("SFG", cli(script, "--help"))


if __name__ == "__main__":
    unittest.main()
