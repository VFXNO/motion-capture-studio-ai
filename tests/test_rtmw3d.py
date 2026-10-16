"""Run with .venv-rtmw3d Python; no model downloads/GPU required."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

try:
    import av
    import numpy as np
except ImportError:
    av = np = None

from backends.rtmw3d.capture import choose_box, sample_video
from backends.rtmw3d.runtime import decode_depth, Z_RANGE
from backends.rtmw3d.models import download_one, digest
from blender_importer import validate_capture


class RTMWTest(unittest.TestCase):
    def test_depth_uses_288_depth_bins_not_384_image_height(self):
        self.assertEqual(decode_depth(144), 0)
        self.assertAlmostEqual(decode_depth(288), Z_RANGE)
        self.assertAlmostEqual(decode_depth(0), -Z_RANGE)

    @unittest.skipUnless(np is not None, "Run in .venv-rtmw3d for optional inference dependencies")
    def test_actor_association_does_not_switch_to_distant_person(self):
        boxes = np.array([[0,0,100,200],[500,0,600,250]])
        np.testing.assert_array_equal(choose_box(boxes), boxes[1])
        np.testing.assert_array_equal(choose_box(boxes, boxes[0]), boxes[0])
        self.assertIsNone(choose_box(boxes[1:], boxes[0]))
        self.assertIsNone(choose_box([]))
        self.assertIsNone(choose_box([[0,0,float('nan'),10]]))

    def test_model_files_are_not_overwritten_when_hash_disagrees(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"model.onnx"
            target.write_bytes(b"user file")
            spec = {"name": target.name, "sha256": "0"*64, "url": "https://invalid.example"}
            with self.assertRaises(ValueError):
                download_one(spec, target.parent)
            self.assertEqual(target.read_bytes(), b"user file")
            spec["sha256"] = digest(target)
            self.assertEqual(download_one(spec, target.parent), target)

    @unittest.skipUnless(av is not None, "Run in .venv-rtmw3d for timestamp/codec tests")
    def test_timestamp_sampling_follows_vfr_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"vfr.mp4"
            timestamps = [0,33,90,170,220,300]
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width, stream.height = 64, 64
                stream.pix_fmt, stream.time_base = "yuv420p", Fraction(1,1000)
                stream.codec_context.time_base = Fraction(1,1000)
                for i, pts in enumerate(timestamps):
                    frame = av.VideoFrame.from_ndarray(np.full((64,64,3),i*30,np.uint8),format="bgr24")
                    frame.pts, frame.time_base = pts, Fraction(1,1000)
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            samples = sample_video(path, 24, 8)
            info = next(samples)
            self.assertEqual(info["samples"], 8)
            for i,(target,source,image) in enumerate(samples):
                self.assertAlmostEqual(target,i/24)
                nearest = min(timestamps,key=lambda t:abs(t/1000-target))/1000
                self.assertAlmostEqual(source, nearest, places=4)
                self.assertEqual(image.shape,(64,64,3))

    def test_face68_schema_is_explicit_and_bounded(self):
        point = {"x":0,"y":0,"z":0}
        data = {"format":"framecraft-mocap","version":1,
                "metadata":{"width":640,"height":960,"fps":24,"engine":"rtmw3d-x"},
                "faceTopology":"coco-wholebody-68","faceEdges":[[0,67]],
                "frames":[{"time":0,"hands":[],"face":{"topology":"coco-wholebody-68",
                          "landmarks":[point]*68,"world":[point]*68,"blendshapes":{}}}]}
        validate_capture(data)
        data["faceEdges"] = [[0,68]]
        with self.assertRaises(ValueError): validate_capture(data)
        data["faceEdges"] = []
        data["metadata"]["engine"] = "mediapipe"
        with self.assertRaises(ValueError): validate_capture(data)
        data["metadata"]["engine"] = "rtmw3d-x"
        data["frames"][0]["face"]["blendshapes"] = {"invented":.7}
        with self.assertRaises(ValueError): validate_capture(data)


if __name__ == "__main__":
    unittest.main()
