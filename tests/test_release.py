"""Dependency-free release checks; no GPU or model download is performed."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from defaults import IMAGE_DEFAULTS, VIDEO_DEFAULTS
from run_image import sd3_sfg_options


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
            "sd3m": (40, 1.0, 0.0, -0.50, 12.0, 1, 13),
            "sd35m": (40, 7.5, 0.25, -0.25, 3.5, 1, 10),
            "flux-dev": (28, 1.0, 0.35, -0.35, 14.0, 1, 0),
            "flux-de-distill": (28, 3.5, 0.35, -0.40, 6.0, 1, 6),
        }
        keys = ("steps", "guidance", "u_s", "u_x", "omega", "start_step", "end_step")
        for model, expected in image_values.items():
            self.assertEqual(tuple(IMAGE_DEFAULTS[model][key] for key in keys), expected)
        video_values = {
            "t2v": (25, 1.0, 0.25, -0.25, 4.0, 7, "joint_encoder"),
            "i2v": (25, 1.0, 0.40, -0.40, 6.0, 7, "joint_encoder"),
            "i2v-text": (25, 1.0, 0.20, -0.20, 3.0, 7, "text_only"),
        }
        keys = ("steps", "guidance", "u_s", "u_x", "omega", "active_steps", "condition_scope")
        for task, expected in video_values.items():
            self.assertEqual(tuple(VIDEO_DEFAULTS[task][key] for key in keys), expected)

    def test_cli_help(self):
        for script in ("run_image.py", "run_video.py"):
            with self.subTest(script=script):
                self.assertIn("SFG", cli(script, "--help"))

    def test_sd3_sampler_options(self):
        one_sided = sd3_sfg_options(IMAGE_DEFAULTS["sd3m"], "all")
        self.assertEqual(one_sided["bridge_direction"], "text_to_image")
        self.assertEqual(one_sided["sfg_strength_u"], -0.50)
        self.assertIsNone(one_sided["sfg_strength_u_t2i"])
        both = sd3_sfg_options(IMAGE_DEFAULTS["sd35m"], "all")
        self.assertEqual(both["bridge_direction"], "both")
        self.assertEqual(both["sfg_strength_u"], 0.25)
        self.assertEqual(both["sfg_strength_u_t2i"], -0.25)
        for model in ("sd3m", "sd35m"):
            config = IMAGE_DEFAULTS[model]
            options = sd3_sfg_options(config, "all")
            self.assertEqual(options["bridge_start_step"], config["start_step"])
            self.assertEqual(options["bridge_end_step"], config["end_step"])
            self.assertEqual(options["bridge_omega"], config["omega"])
            self.assertEqual(options["bridge_layers"], "all")

    def test_sd3_attention_strength_mapping(self):
        # Exercise the actual processor initializer without importing GPU packages.
        path = ROOT / "image/sd3/latent_sd35.py"
        tree = ast.parse(path.read_text())
        processor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == "SFGJointAttnProcessor2_0")
        methods = [node for node in processor.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("__init__", "_normalize_bridge_direction")]
        minimal = ast.ClassDef(name="Processor", bases=[], keywords=[], body=methods,
                               decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[]))
        namespace = {"F": SimpleNamespace(scaled_dot_product_attention=None)}
        exec(compile(module, str(path), "exec"), namespace)
        for model in ("sd3m", "sd35m"):
            config = IMAGE_DEFAULTS[model]
            options = sd3_sfg_options(config, "all")
            instance = namespace["Processor"](**{key: options[key] for key in (
                "sfg_strength_u", "sfg_strength_u_t2i", "bridge_direction")})
            self.assertEqual(instance.bridge_scale_delta_i2t, -config["u_s"])
            self.assertEqual(instance.bridge_scale_delta_t2i, -config["u_x"])

    def test_sampling_window_boundaries(self):
        for model, config in IMAGE_DEFAULTS.items():
            if model.startswith("sd3"):
                path = ROOT / "image/sd3/latent_sd35.py"
                name = "_use_sfg_this_step"
            else:
                filename = "flux_dev.py" if model == "flux-dev" else "flux_dedistill.py"
                path = ROOT / "image" / filename
                name = "use_bridge_step"
            tree = ast.parse(path.read_text())
            function = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            function.decorator_list = []
            module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            namespace = {}
            exec(compile(module, str(path), "exec"), namespace)
            active = [step + 1 for step in range(config["steps"])
                      if namespace[name](step, config["start_step"], config["end_step"])]
            end = config["end_step"] or config["steps"]
            self.assertEqual(active, list(range(config["start_step"], end + 1)))
            self.assertFalse(namespace[name](0, 2, 3))
            self.assertTrue(namespace[name](1, 2, 3))
            self.assertTrue(namespace[name](2, 2, 3))
            self.assertFalse(namespace[name](3, 2, 3))


if __name__ == "__main__":
    unittest.main()
