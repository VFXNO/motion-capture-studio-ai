"""Regressions for observed raised arms, depth transitions and raw transport."""
import copy
import unittest
import numpy as np

from backends.rtmw3d.arm_refinement import refine_arms, refine_tracking_arms


def fixture(count=60):
    pose=np.zeros((17,3))
    for j,p in {0:(0,0,0),1:(-.1,0,0),2:(-.1,.4,0),3:(-.1,.8,0),4:(.1,0,0),5:(.1,.4,0),6:(.1,.8,0),
                7:(0,-.2,0),8:(0,-.4,0),9:(0,-.5,0),10:(0,-.6,0),11:(.2,-.4,0),12:(.3,-.6,.1),13:(.3,-.8,.1),
                14:(-.2,-.4,0),15:(-.3,-.2,.1),16:(-.3,0,.1)}.items(): pose[j]=p
    poses=np.repeat(pose[None],count,axis=0)
    observed=np.ones((count,17,3)); observed[:,:,:2]=[360,640]
    for a,b in [(11,5),(12,7),(13,9),(14,6),(15,8),(16,10),(4,11),(1,12),(5,13),(2,14),(6,15),(3,16)]:
        observed[:,b,:2]=poses[:,a,:2]*600+[360,640]
    native=np.ones((count,133,4)); native[:,:,:2]=[360,640]; native[:,:,2]=0
    native[:,:17,:2]=observed[:,:,:2]
    return poses,observed,native,np.arange(count)/30


class ArmRefinementTest(unittest.TestCase):
    def test_correctly_detected_raised_wrist_overrules_backward_neural_xy(self):
        poses,obs,native,times=fixture()
        poses[:,13]=poses[:,12]+[-.2,0,0]  # Neural wrist points into the head.
        before=poses.copy()
        fitted,_=refine_arms(poses,obs,native,times)
        arm=fitted[:,13]-fitted[:,12]
        self.assertTrue((arm[:,1]<-.18).all())
        self.assertTrue((abs(arm[:,0])<1e-5).all())
        np.testing.assert_array_equal(poses,before)
        untouched=[i for i in range(17) if i not in (12,13,15,16)]
        np.testing.assert_array_equal(fitted[:,untouched],before[:,untouched])

    def test_depth_sign_crossing_is_continuous_without_changing_lengths(self):
        poses,obs,native,times=fixture()
        poses[:30,16,2]=.23
        poses[30:,16,2]=-.03
        fitted,_=refine_arms(poses,obs,native,times)
        vectors=fitted[:,16]-fitted[:,15]
        lengths=np.linalg.norm(vectors,axis=1)
        np.testing.assert_allclose(lengths,np.median(np.linalg.norm(poses[:,16]-poses[:,15],axis=1)),atol=1e-8)
        unit=vectors/lengths[:,None]
        angles=np.degrees(np.arccos(np.clip((unit[:-1]*unit[1:]).sum(1),-1,1)))
        raw=poses[:,16]-poses[:,15]
        raw/=np.linalg.norm(raw,axis=1)[:,None]
        raw_jump=np.degrees(np.arccos(np.clip((raw[:-1]*raw[1:]).sum(1),-1,1))).max()
        self.assertGreater(raw_jump,60)
        self.assertLess(angles.max(),raw_jump/2)
        self.assertGreater(vectors[0,2],0)
        self.assertLess(vectors[-1,2],0)

    def test_uncertain_wholebody_detection_does_not_veto_stronger_body_detector(self):
        poses,obs,native,times=fixture()
        poses[:,13]=poses[:,12]+[-.2,0,0]
        native[:,9,:2]=obs[:,9,:2]+[150,150]
        native[:,9,3]=.4
        fitted,_=refine_arms(poses,obs,native,times)
        self.assertTrue(((fitted[:,13]-fitted[:,12])[:,1]<-.18).all())

    def test_raw_predictions_are_preserved_and_missing_samples_stay_missing(self):
        poses,obs,native,times=fixture(8)
        frames=[{'time':float(t),'keypoints':n.tolist(),'body2d':o.tolist(),'temporalBody':p.tolist()} for p,o,n,t in zip(poses,obs,native,times)]
        frames[3]={'time':.1,'keypoints':None}
        tracking={'metadata':{'bodyEstimator':{'model':'motionagformer-b'}},'frames':frames}
        before=copy.deepcopy(tracking)
        result=refine_tracking_arms(tracking)
        self.assertEqual(tracking,before)
        self.assertNotIn('refinedBody',result['frames'][3])
        self.assertIn('refinedBody',result['frames'][4])
        for a,b in zip(frames,result['frames']):
            self.assertEqual(a.get('temporalBody'),b.get('temporalBody'))
            self.assertEqual(a.get('keypoints'),b.get('keypoints'))

    def test_single_frame_and_invalid_data(self):
        values=fixture(1)
        fitted,_=refine_arms(*values)
        self.assertTrue(np.isfinite(fitted).all())
        values[0][0,0,0]=float('nan')
        with self.assertRaises(ValueError): refine_arms(*values)


if __name__=='__main__': unittest.main()
