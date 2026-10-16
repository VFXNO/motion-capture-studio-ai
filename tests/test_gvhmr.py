import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from backends.gvhmr.motion import build_capture, z_up, NAMES, PARENTS
from backends.gvhmr.diagnostics import diagnose
from gvhmr_fixture import capture, parameters

spec = importlib.util.spec_from_file_location("gvhmr_importer", ROOT / "blender_importer.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


class GVHMRTest(unittest.TestCase):
    def test_conversion_preserves_pelvis_and_transforms_axes_without_mutation(self):
        params, skeletons = parameters()
        before = copy.deepcopy((params, skeletons))
        data = build_capture(params, skeletons, {"fps": 30, "width": 640, "height": 480})
        self.assertEqual((params, skeletons), before)
        self.assertEqual(z_up([1, 2, 3]), [1., -3., 2.])
        self.assertEqual(data["frames"][0]["root"][:2], [0., 0.])
        self.assertAlmostEqual(data["frames"][0]["root"][2], 1.12)
        self.assertAlmostEqual(data["frames"][1]["root"][0], .03)
        self.assertAlmostEqual(data["frames"][1]["root"][1], -.01)
        self.assertEqual(data["frames"][3]["rotations"][0], z_up(params["global_orient"][3]))
        self.assertAlmostEqual(data["frames"][-1]["time"], 11 / 30)
        self.assertEqual(data["metadata"]["layers"], ["body", "root"])

    def test_shape_offsets_are_retained_for_every_frame(self):
        data = capture()
        self.assertNotEqual(data["frames"][0]["offsets"][1], data["frames"][-1]["offsets"][1])
        self.assertEqual(data["skeleton"]["names"], importer.GVHMR_NAMES)
        self.assertEqual(PARENTS, importer.GVHMR_PARENTS)
        self.assertEqual(len(NAMES), 22)
        self.assertIs(importer.validate_capture(data), data)

    def test_reject_nonfinite_and_bad_dimensions(self):
        for key in ("rotations", "offsets"):
            for value in (float("nan"), float("inf"), "0", True, 10001):
                data = capture()
                data["frames"][0][key][3][0] = value
                with self.assertRaises(ValueError):
                    importer.validate_capture(data)
        for key in ("rotations", "offsets", "root"):
            data = capture()
            data["frames"][0][key].pop()
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        params, skeletons = parameters()
        params["body_pose"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            build_capture(params, skeletons, {"fps": 30})

    def test_reject_changed_hierarchy_and_double_pelvis_translation(self):
        for key, value in (("names", ["bad"] * 22), ("parents", [-1] * 22)):
            data = capture()
            data["skeleton"][key] = value
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        data = capture()
        data["frames"][0]["offsets"][0] = [0, 0, 1]
        with self.assertRaises(ValueError):
            importer.validate_capture(data)

    def test_reject_bad_timing_frame_count_and_metadata(self):
        for value in (0, math.nan, True, "30", 241):
            data = capture()
            data["metadata"]["fps"] = value
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        data = capture()
        data["frames"][1]["time"] = 0
        with self.assertRaises(ValueError):
            importer.validate_capture(data)
        data = capture()
        data["frames"] = []
        with self.assertRaises(ValueError):
            importer.validate_capture(data)

    def test_doctor_never_claims_ready_with_missing_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            report = diagnose(Path(folder))
        self.assertFalse(report["prerequisitesPresent"])
        self.assertFalse(report["readyForAttempt"])
        self.assertFalse(report["inferenceVerified"])
        self.assertEqual(len(report["assets"]), 5)
        self.assertTrue(all(not asset["present"] for asset in report["assets"]))

    def test_cli_does_not_assume_licenses_or_overwrite(self):
        video = ROOT / "tests/fixtures/test.webm"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output.gvhmr.json"
            command = [sys.executable, str(ROOT / "tools/gvhmr.py"), "run", str(video), str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--accept-noncommercial", result.stderr)
            self.assertFalse(output.exists())
            output.write_text("keep me", encoding="utf8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertIn("overwrite", result.stderr)
            self.assertEqual(output.read_text(encoding="utf8"), "keep me")


if __name__ == "__main__":
    unittest.main()
