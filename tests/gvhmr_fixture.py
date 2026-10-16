"""Synthetic skeleton/rotations, not SMPL assets or model predictions."""
from backends.gvhmr.motion import build_capture


def parameters(count=12):
    rest = [[0, .92, 0], [-.1, .9, 0], [.1, .9, 0], [0, 1.05, 0],
            [-.1, .5, .02], [.1, .5, .02], [0, 1.2, 0], [-.1, .1, 0], [.1, .1, 0],
            [0, 1.38, 0], [-.1, .06, .15], [.1, .06, .15], [0, 1.52, 0],
            [-.1, 1.43, 0], [.1, 1.43, 0], [0, 1.68, 0], [-.23, 1.43, 0], [.23, 1.43, 0],
            [-.5, 1.43, 0], [.5, 1.43, 0], [-.75, 1.43, 0], [.75, 1.43, 0]]
    skeletons = [[[x * (1 + i * .001), y, z] for x, y, z in rest] for i in range(count)]
    params = {"global_orient": [[.1 * i, .08 * i, -.02 * i] for i in range(count)],
              "transl": [[2 + i * .03, .2 + i * .002, 5 + i * .01] for i in range(count)],
              "body_pose": [[0.] * 63 for _ in range(count)]}
    for i, pose in enumerate(params["body_pose"]):
        # Twist the left upper arm about its resting axis, flex knee and elbow.
        pose[(16 - 1) * 3] = .1 * i
        pose[(18 - 1) * 3 + 2] = .06 * i
        pose[(4 - 1) * 3] = .03 * i
    return params, skeletons


def capture(count=12):
    params, skeletons = parameters(count)
    return build_capture(params, skeletons, {"fps": 30, "width": 640, "height": 480, "filename": "synthetic"})
