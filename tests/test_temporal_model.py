"""Temporal-model transport tests; optional numpy, no model downloads."""
import copy
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from backends.rtmw3d.temporal import coco_to_h36m, flip_pose, infer_sequence, segments, add_temporal_body


@unittest.skipUnless(np is not None, "Run in the isolated local-AI environment")
class TemporalModelTest(unittest.TestCase):
    def test_coco_anatomical_sides_and_portrait_letterbox(self):
        k = np.zeros((1, 133, 4), np.float32)
        k[..., 0], k[..., 1], k[..., 3] = 360, 640, .9
        k[0, 5, 0], k[0, 6, 0] = 500, 200
        k[0, 11, 0], k[0, 12, 0] = 450, 250
        k[0, 15, 1] = 1200
        out = coco_to_h36m(k, 720, 1280)
        self.assertGreater(out[0,11,0], 0)  # COCO left shoulder -> H36M left shoulder
        self.assertLess(out[0,14,0], 0)
        self.assertGreater(out[0,4,0], 0)   # left hip is 4, NOT 1
        self.assertLess(out[0,1,0], 0)
        self.assertAlmostEqual(out[0,6,1], .875)
        self.assertAlmostEqual(out[0,11,2], .9, places=6)
        self.assertLess(abs(out[0,11,0]), 1)

    def test_flip_is_involution_and_preserves_confidence(self):
        x = np.random.default_rng(4).normal(size=(1,243,17,3)).astype(np.float32)
        np.testing.assert_array_equal(x, flip_pose(flip_pose(x)))

    def test_overlaps_cover_every_timestamp_without_stretching_vfr_time(self):
        times = np.cumsum(np.resize([.02,.04,.03], 420)); times -= times[0]
        x = np.zeros((len(times),17,3),np.float32)
        x[:,:,2] = 1
        x[:,11,0] = times*.02
        before = x.copy()
        y = infer_sequence(x,times,lambda batch:batch.copy())
        # Edge padding holds the final observation beyond the last timestamp.
        # Interior linear motion must be exact; the fractional last grid cell
        # can differ by at most one 30-Hz step (no fabricated extrapolation).
        interior = times <= np.floor(times[-1]*30)/30
        np.testing.assert_allclose(y[interior,11,0], times[interior]*.02, atol=1e-6)
        self.assertLessEqual(abs(y[-1,11,0]-times[-1]*.02), .02/30)
        np.testing.assert_allclose(y[:,0], 0, atol=1e-6)
        np.testing.assert_array_equal(x,before)

    def test_short_clip_has_no_time_rescaling_or_missing_output(self):
        for length in (1,2,27,243,244):
            x = np.zeros((length,17,3),np.float32); x[:,:,2]=1; x[:,11,0]=.2
            y = infer_sequence(x,np.arange(length)/24,lambda batch:batch.copy())
            self.assertEqual(y.shape,(length,17,3))
            np.testing.assert_allclose(y[:,11,0],.2,atol=1e-6)

    def test_missing_bodies_split_context_and_predictions_do_not_replace_raw(self):
        k = np.zeros((133,4)); k[:,0]=360; k[:,1]=640; k[:,3]=.9
        frames = [{"time":t,"keypoints":k.tolist() if i!=2 else None} for i,t in enumerate((0,.03,.06,.09,1))]
        self.assertEqual(list(segments(frames)),[[0,1],[3],[4]])
        data = {"format":"framecraft-rtmw3d-tracking","version":1,"metadata":{"width":720,"height":1280},"frames":frames}
        before = copy.deepcopy(data)
        result = add_temporal_body(data,Path.cwd(),predict=lambda x:x.copy())
        self.assertEqual(data,before)
        self.assertNotIn("temporalBody",result["frames"][2])
        self.assertIn("temporalBody",result["frames"][3])
        self.assertEqual(result["metadata"]["bodyEstimator"]["provider"],"CPUExecutionProvider")

    def test_bad_network_output_fails_explicitly(self):
        x = np.ones((3,17,3),np.float32)
        with self.assertRaises(ValueError):
            infer_sequence(x,[0,.03,.06],lambda batch:np.full_like(batch,np.nan))


if __name__ == "__main__":
    unittest.main()
