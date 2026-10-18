import unittest
import numpy as np
from run_experiment import EdgeGuard, augment, corrupt, balanced_data


class ProtocolTests(unittest.TestCase):
    def test_hierarchy_routes_only_suspected_attacks(self):
        class Triage:
            def predict(self,X): return np.array([0,1,0,1])
        class Family:
            def predict(self,X):
                np.testing.assert_array_equal(X,np.array([[2.],[4.]]))
                return np.array([3,5])
        model=EdgeGuard(normal_id=6);model.stage1_=Triage();model.stage2_=Family()
        np.testing.assert_array_equal(model.predict(np.arange(1,5).reshape(-1,1)),[6,3,6,5])

    def test_corruption_operators_and_training_range(self):
        X=np.array([[-10.,0.],[.25,.75],[10.,1.]])
        low=np.zeros(2);high=np.ones(2)
        np.testing.assert_array_equal(corrupt(X,'noise',0,low,high,42),X)
        np.testing.assert_array_equal(corrupt(X,'dropout',1,low,high,42),np.zeros_like(X))
        quant=corrupt(X,'quantization',2,low,high,42)
        self.assertTrue(np.isin(quant,[0,1]).all())
        np.testing.assert_array_equal(corrupt(X,'noise',.2,low,high,42),corrupt(X,'noise',.2,low,high,42))

    def test_augmentation_preserves_labels_and_weights(self):
        X=np.ones((8,3));y=np.arange(8);w=np.arange(1,9)
        aug,labels,weights=augment(X,y,w,42)
        self.assertEqual(aug.shape,(24,3))
        np.testing.assert_array_equal(aug[:8],X)
        np.testing.assert_array_equal(labels,np.tile(y,3))
        np.testing.assert_array_equal(weights,np.tile(w,3))

    def test_smote_floor_does_not_downsample_majority(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(4010,2));y=np.r_[np.zeros(4002,dtype=int),np.ones(8,dtype=int)]
        x,b,w=balanced_data(X,y,True,True)
        np.testing.assert_array_equal(np.bincount(b),[4002,4000])
        self.assertAlmostEqual(w[b==0].sum(),w[b==1].sum())


if __name__=='__main__': unittest.main()
