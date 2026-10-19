"""Observed arm directions with neural depth priors, in scaled-orthographic space.

Raw neural predictions are never changed. The torso/legs stay model-derived;
only arm vectors supported by detector agreement or a stronger dedicated-body
detection are corrected. Depth
signs are selected over each continuous sequence, not greedily frame by frame.
"""
import copy
import numpy as np


def camera_scales(poses, observations):
    # Exclude elbows/wrists: their neural XY errors must not set camera scale.
    model = [11, 14, 4, 1, 5, 2, 6, 3]
    coco = [5, 6, 11, 12, 13, 14, 15, 16]
    scales = []
    for pose, observed in zip(poses, observations):
        x, y = pose[model, :2], observed[coco, :2]
        base = np.clip((observed[coco, 2] - .3) / .5, 0, 1)
        weights = base.copy()
        scale = 0
        for _ in range(4):
            total = max(1e-9, weights.sum())
            a = x - (x * weights[:, None]).sum(0) / total
            b = y - (y * weights[:, None]).sum(0) / total
            scale = (a * b * weights[:, None]).sum() / max(1e-9, (a * a * weights[:, None]).sum())
            errors = np.linalg.norm(b - a * scale, axis=1)
            threshold = max(4, 2.5 * np.median(errors))
            weights = base * np.minimum(1, threshold / np.maximum(errors, 1e-9))
        if not np.isfinite(scale) or scale <= 1:
            raise ValueError('Cannot estimate arm observation scale.')
        scales.append(scale)
    return np.asarray(scales)


def depth_path(magnitudes, prior, lengths, times):
    """Two-state dynamic program: choose depth hemisphere using the whole clip."""
    candidates = magnitudes[:, None] * np.array([-1, 1])
    unary = ((candidates - prior[:, None]) / lengths) ** 2
    cost = unary[0].copy()
    parents = np.zeros((len(prior), 2), dtype=np.int8)
    for i in range(1, len(prior)):
        # Same physical velocity penalty across source sampling rates.
        dt = max(.001, times[i] - times[i - 1])
        transition = 2 * (1 / 30 / dt) * ((candidates[i, :, None] - candidates[i - 1, None, :]) / lengths) ** 2
        combined = transition + cost[None, :]
        parents[i] = combined.argmin(axis=1)
        cost = unary[i] + combined.min(axis=1)
    state = int(cost.argmin())
    result = np.empty(len(prior))
    for i in range(len(prior) - 1, -1, -1):
        result[i] = candidates[i, state]
        state = parents[i, state]
    return result


def refine_arms(poses, observations, native, times):
    poses = np.asarray(poses, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    native = np.asarray(native, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    count = len(times)
    if not 1 <= count <= 1800 or poses.shape != (count, 17, 3) or observations.shape != (count, 17, 3) or native.shape != (count, 133, 4):
        raise ValueError('Invalid observed-arm input shape.')
    if not all(np.isfinite(x).all() for x in (poses, observations, native, times)) or (np.diff(times) <= 0).any():
        raise ValueError('Observed-arm inputs must be finite with increasing times.')
    if len(times)>1 and (np.diff(times)>.15).any():
        raise ValueError('Split gaps before observed-arm refinement.')
    result = poses.copy()
    scales = camera_scales(poses, observations)
    evidence = []
    for joints, coco in (((11, 12, 13), (5, 7, 9)), ((14, 15, 16), (6, 8, 10))):
        for parent, child, a, b in zip(joints, joints[1:], coco, coco[1:]):
            original = poses[:, child] - poses[:, parent]
            length = float(np.median(np.linalg.norm(original, axis=1)))
            if length < .01:
                raise ValueError('Degenerate neural arm length.')
            # Agreement is additional evidence, not an assertion of correctness.
            agreement = np.maximum(np.linalg.norm(observations[:, a, :2] - native[:, a, :2], axis=1),
                                   np.linalg.norm(observations[:, b, :2] - native[:, b, :2], axis=1))
            confidence = np.minimum(observations[:, a, 2], observations[:, b, 2])
            native_confidence = np.minimum(native[:, a, 3], native[:, b, 3])
            agreement_weight = np.exp(-(agreement / (scales * length * .35)) ** 2)
            # An uncertain wholebody elbow must not veto a much stronger
            # dedicated-body observation during a rapid raised-arm transition.
            advantage = np.clip((confidence - native_confidence - .1) / .2, 0, 1)
            support = np.maximum(agreement_weight * np.clip((native_confidence - .3) / .3, 0, 1), advantage)
            reliability = np.clip((confidence - .35) / .35, 0, 1) * support
            xy = (observations[:, b, :2] - observations[:, a, :2]) / scales[:, None]
            xy = original[:, :2] * (1 - reliability[:, None]) + xy * reliability[:, None]
            # Impossible projected reach is shortened in-plane; never choose
            # arbitrary out-of-plane bends to conceal contradictory observations.
            xy *= np.minimum(1, length / np.maximum(1e-9, np.linalg.norm(xy, axis=1)))[:, None]
            magnitude = np.sqrt(np.maximum(0, length**2 - (xy**2).sum(axis=1)))
            z = depth_path(magnitude, original[:, 2], length, times)
            # Refine continuously through the image plane. A discrete sign
            # change at nonzero depth is a snap, not a physical arm movement.
            radius = np.linalg.norm(xy, axis=1) / length
            prior = np.clip(original[:, 2] / length, -.999, .999)
            angle = np.arcsin(np.clip(z / length, -.999, .999))
            dt = np.maximum(.001, np.diff(times))
            rate = np.minimum(4, 1 / (30 * dt))
            for _ in range(120):
                gradient = -2 * (np.cos(angle) - radius) * np.sin(angle)
                gradient += .7 * (np.sin(angle) - prior) * np.cos(angle)
                velocity = np.diff(angle) * rate
                term = .06 * velocity * rate
                gradient[:-1] -= term
                gradient[1:] += term
                if len(angle) > 2:
                    acceleration = np.diff(velocity)
                    term = .4 * acceleration
                    gradient[:-2] += term * rate[:-1]
                    gradient[1:-1] -= term * (rate[:-1] + rate[1:])
                    gradient[2:] += term * rate[1:]
                angle = np.clip(angle - .04 / max(1, float(rate.max())**2 if len(rate) else 1) * gradient, -1.55, 1.55)
            direction = xy / np.maximum(1e-9, np.linalg.norm(xy, axis=1))[:, None]
            vector = np.column_stack((direction * (length * np.cos(angle))[:, None], length * np.sin(angle)))
            degenerate = np.linalg.norm(xy, axis=1) < 1e-8
            vector[degenerate] = original[degenerate] * (length / np.maximum(1e-9, np.linalg.norm(original[degenerate], axis=1)))[:, None]
            result[:, child] = result[:, parent] + vector
            evidence.append(reliability)
    return result, {'meanEvidence': float(np.mean(evidence)), 'method': 'observed-arms-neural-depth-v1'}


def refine_tracking_arms(tracking):
    from .temporal import segments, body_keypoints
    result = copy.deepcopy(tracking)
    for frame in result['frames']:
        frame.pop('refinedBody', None)
    evidence = []
    for part in segments(tracking['frames']):
        # Split missing neural predictions too, rather than joining separate
        # inference segments or reusing stale fitted data from an earlier pass.
        runs, current = [], []
        for index in part:
            if tracking['frames'][index].get('temporalBody') is None:
                if current: runs.append(current)
                current = []
            else:
                current.append(index)
        if current: runs.append(current)
        for run in runs:
            frames = [tracking['frames'][i] for i in run]
            native = [f['keypoints'] for f in frames]
            observed = [np.asarray(body_keypoints(f))[:, [0, 1, 3]] for f in frames]
            fitted, stats = refine_arms([f['temporalBody'] for f in frames], observed, native, [f['time'] for f in frames])
            for index, pose in zip(run, fitted):
                result['frames'][index]['refinedBody'] = pose.tolist()
            evidence.extend([stats['meanEvidence']] * len(run))
    if not evidence:
        raise ValueError('No neural body predictions available for observed-arm refinement.')
    result['metadata']['bodyEstimator']['refinement'] = {
        'method': 'observed-arms-neural-depth-v1', 'meanEvidence': float(np.mean(evidence)),
        'notes': 'Observed arm vectors; robust torso/leg camera scale; clip-wide lengths; sequence depth-sign selection and continuous depth transitions. Raw temporalBody remains separate. Hidden depth is still inferred.'}
    return result
