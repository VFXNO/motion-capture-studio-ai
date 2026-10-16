"""Schema checks that do not require a Blender installation."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("importer", Path(__file__).parents[1] / "blender_importer.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


def points(count):
    return [{"x": 0.1, "y": 0.2, "z": -0.1} for _ in range(count)]


def capture():
    return {"format": "framecraft-mocap", "version": 1,
            "metadata": {"fps": 24, "width": 1280, "height": 720},
            "faceEdges": [[0, 1]],
            "frames": [{"time": 0, "body": {"world": points(33)},
                        "hands": [{"side": "Left", "world": points(21)}],
                        "face": {"landmarks": points(478), "blendshapes": {"eyeBlinkLeft": 0.5}}}]}


class CaptureValidationTest(unittest.TestCase):
    def test_ground_root_schema_and_backward_compatibility(self):
        data = capture()
        data["rootMotion"] = {"enabled": True, "method": "grounded-perspective-v1", "pitch": 10}
        data["frames"][0]["root"] = {"position": {"x": 0.2, "y": 1.5, "z": 1}, "status": "tracked"}
        importer.validate_capture(data)
        self.assertEqual(importer.root_location(data["frames"][0], (0, 0, 0)), (.2, 1.5, 1))
        self.assertEqual(importer.root_location({}, (.2, 1.5, 1)), (.2, 1.5, 1))
        importer.validate_capture(capture())
        self.assertEqual(importer.root_motion_summary(data), {"x": 0, "y": 0, "z": 0})
        self.assertIsNone(importer.root_motion_summary(capture()))

    def test_invalid_root_positions_and_transform_settings(self):
        for value in (float("nan"), float("inf"), "1", 10001):
            data = capture()
            data["frames"][0]["root"] = {"position": {"x": value, "y": 0, "z": 1}, "status": "tracked"}
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        for pitch in (float("nan"), 90, "10"):
            data = capture()
            data["rootMotion"] = {"enabled": True, "method": "grounded-perspective-v1", "pitch": pitch}
            with self.assertRaises(ValueError):
                importer.validate_capture(data)

    def test_camera_to_ground_axes_match_exporter(self):
        self.assertEqual(importer.ground_coordinates({"x": 1, "y": 2, "z": 3}), (1, 3, -2))
        result = importer.ground_coordinates({"x": 0.5, "y": 0, "z": 2}, 30)
        self.assertAlmostEqual(result[1], 3 ** .5)
        self.assertAlmostEqual(result[2], -1)

    def test_valid_body_hands_face(self):
        data = capture()
        self.assertIs(importer.validate_capture(data), data)

    def test_missing_detections_allowed(self):
        data = capture()
        data["frames"].append({"time": 1 / 24, "body": None, "hands": [], "face": None})
        importer.validate_capture(data)

    def test_wrong_schema_and_fps(self):
        for key, value in (("format", "other"), ("version", 2)):
            data = capture()
            data[key] = value
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        for fps in (0, -1, float("nan"), float("inf"), 241, "24"):
            data = capture()
            data["metadata"]["fps"] = fps
            with self.assertRaises(ValueError):
                importer.validate_capture(data)

    def test_missing_and_nonfinite_landmarks(self):
        for value in (None, float("nan"), float("inf"), "0.1"):
            data = capture()
            data["frames"][0]["body"]["world"][0]["x"] = value
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        data = capture()
        data["frames"][0]["hands"][0]["world"].pop()
        with self.assertRaises(ValueError):
            importer.validate_capture(data)

    def test_frames_are_bounded_and_times_increasing(self):
        data = capture()
        data["frames"].append(copy.deepcopy(data["frames"][0]))
        with self.assertRaises(ValueError):
            importer.validate_capture(data)
        data = capture()
        data["frames"] = []
        with self.assertRaises(ValueError):
            importer.validate_capture(data)

    def test_invalid_topology_and_property_paths_rejected(self):
        for edge in ([0, 478], [-1, 1], [0], ["0", 1]):
            data = capture()
            data["faceEdges"] = [edge]
            with self.assertRaises(ValueError):
                importer.validate_capture(data)
        data = capture()
        data["frames"][0]["face"]["blendshapes"] = {'bad["path"]': 1}
        with self.assertRaises(ValueError):
            importer.validate_capture(data)


if __name__ == "__main__":
    unittest.main()
