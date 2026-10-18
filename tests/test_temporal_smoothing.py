"""Time-based pose cleanup without Blender or third-party dependencies."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("importer", Path(__file__).parents[1] / "blender_importer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TemporalSmoothingTest(unittest.TestCase):
    def test_jitter_drops_without_delaying_or_flattening_slow_movement(self):
        times = [i / 30 for i in range(121)]
        truth = [math.sin(t * math.pi) for t in times]
        values = [(p + .08 * (-1) ** i, 0, 0) for i, p in enumerate(truth)]
        cleaned = module.smooth_point_track(values, times, [1] * len(times), 8, .045)
        original_error = sum((p[0] - t) ** 2 for p, t in zip(values[4:-4], truth[4:-4]))
        new_error = sum((p[0] - t) ** 2 for p, t in zip(cleaned[4:-4], truth[4:-4]))
        self.assertLess(new_error, original_error * .1)
        self.assertGreater(max(p[0] for p in cleaned), .95)
        peak = max(range(5, 25), key=lambda i: cleaned[i][0])
        self.assertLessEqual(abs(peak - 15), 1)

    def test_velocity_spikes_do_not_open_the_lowpass_filter(self):
        times = [i / 30 for i in range(15)]
        values = [(i * .01, 0, 0) for i in range(15)]
        values[7] = (1.2, 0, 0)
        result = module.smooth_point_track(values, times, [1] * len(times), 8, .045)
        self.assertLess(max(math.dist(a, b) for a, b in zip(result, result[1:])), .1)
        self.assertLess(result[7][0], .25)

    def test_missing_body_and_time_gaps_are_not_filled_and_source_is_unchanged(self):
        def body(x):
            return {"world": [{"x": x, "y": 0, "z": 0, "visibility": 1} for _ in range(33)]}
        frames = [{"time": 0, "body": body(0)}, {"time": .03, "body": None},
                  {"time": .06, "body": body(1)}, {"time": 2, "body": body(4)}]
        original = copy.deepcopy(frames)
        result = module.smooth_body_frames(frames, .5)
        self.assertEqual(frames, original)
        self.assertIsNone(result[1]["body"])
        self.assertEqual(result[0]["body"]["world"][0]["x"], 0)
        self.assertEqual(result[2]["body"]["world"][0]["x"], 1)
        self.assertEqual(result[3]["body"]["world"][0]["x"], 4)
        self.assertIs(module.smooth_body_frames(frames, 0), frames)

    def test_vfr_timestamps_not_assumed_frame_numbers(self):
        times = [0, .01, .025, .06, .1, .15]
        values = [(3 * t, -t, 0) for t in times]
        result = module.smooth_point_track(values, times, [1] * len(times), 8, .015)
        self.assertTrue(all(math.isfinite(v) for point in result for v in point))
        self.assertTrue(all(a[0] < b[0] for a, b in zip(result, result[1:])))
        self.assertLess(abs(result[2][0] - values[2][0]), .025)


if __name__ == "__main__":
    unittest.main()
