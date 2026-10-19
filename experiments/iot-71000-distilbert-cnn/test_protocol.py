import json
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fetch_data import canonical_label
from model import DualHead,multitask_loss,predict_routed


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4);torch.manual_seed(42)

    def test_original_benign_events_never_become_attacks(self):
        events={'1':{'description':'control center:Generator Control'},'2':{'description':'drift-off:drift-off'}}
        for marker in [False,'1 (benign event)','1']:self.assertEqual(canonical_label(marker,events),'Benign')
        with self.assertRaises(ValueError):canonical_label(None,events)
        self.assertEqual(canonical_label('2',events),'DriftOff')
        with self.assertRaises(ValueError):canonical_label('999',events)

    def test_published_matrix_arithmetic(self):
        f=json.loads((Path(__file__).parent/'paper_facts.json').read_text());m=np.array(f['figure5_confusion_matrix'])
        self.assertEqual(int(m.sum()),8641);self.assertEqual(int(m.trace()),8524)
        self.assertEqual(round(m.trace()/m.sum()*100,2),98.65)
        np.testing.assert_array_equal(m.sum(1),f['figure4_counts'])

    def test_singleton_attention_has_no_query_key_dependence(self):
        torch.manual_seed(42);m=nn.MultiheadAttention(24,4,batch_first=True,dropout=0).eval()
        x=torch.randn(3,1,24,requires_grad=True)
        before,weights=m(x,x,x,need_weights=True)
        torch.testing.assert_close(weights,torch.ones_like(weights))
        before.square().sum().backward()
        torch.testing.assert_close(m.in_proj_weight.grad[:48],torch.zeros_like(m.in_proj_weight.grad[:48]),atol=1e-7,rtol=0)
        with torch.no_grad():m.in_proj_weight[:48].normal_(0,10)
        after,_=m(x,x,x)
        torch.testing.assert_close(before,after,atol=1e-6,rtol=0)

    def test_multiple_tokens_remove_that_degeneracy(self):
        torch.manual_seed(42);m=nn.MultiheadAttention(24,4,batch_first=True,dropout=0).eval()
        x=torch.randn(3,4,24);out,_=m(x,x,x);out.square().sum().backward()
        self.assertGreater(float(m.in_proj_weight.grad[:48].abs().sum()),0)

    def test_benign_samples_excluded_from_attack_family_loss(self):
        b=torch.randn(4,2);a=torch.randn(4,5);labels=torch.zeros(4,dtype=torch.long)
        one=multitask_loss(b,a,labels,torch.full((4,),-1),torch.ones(5))
        two=multitask_loss(b,a*100,labels,torch.full((4,),-1),torch.ones(5))
        torch.testing.assert_close(one,two)

    def test_full_width_real_distilbert_forward_backward(self):
        model=DualHead(32,initialization='random');x=torch.randn(4,32)
        b,a=model(x);self.assertEqual(tuple(b.shape),(4,2));self.assertEqual(tuple(a.shape),(4,5))
        loss=multitask_loss(b,a,torch.tensor([0,1,1,1]),torch.tensor([-1,0,2,4]),torch.ones(5))
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.binary_head.weight.grad.norm()),0)
        model.eval();p,probs=predict_routed(model,x,1,[0,2,3,4,5])
        self.assertEqual(tuple(probs.shape),(4,6));torch.testing.assert_close(probs.sum(1),torch.ones(4),atol=1e-6,rtol=0)
        self.assertTrue(torch.isin(p,torch.arange(6)).all())

    def test_focal_loss_is_finite_and_masks_benign_rows(self):
        b=torch.randn(4,2);a=torch.randn(4,5);yb=torch.tensor([0,1,1,0]);ya=torch.tensor([-1,0,4,-1])
        ce=multitask_loss(b,a,yb,ya,torch.ones(5));focal=multitask_loss(b,a,yb,ya,torch.ones(5),focal=True)
        self.assertTrue(torch.isfinite(focal));self.assertLessEqual(float(focal),float(ce))

    def test_all_ablation_variants_have_valid_dual_outputs(self):
        for variant in ['cnn_only','distilbert_only','without_attention','focal']:
            with self.subTest(variant=variant):
                m=DualHead(16,variant=variant,initialization='random').eval()
                with torch.inference_mode():b,a=m(torch.randn(2,16))
                self.assertEqual(tuple(b.shape),(2,2));self.assertEqual(tuple(a.shape),(2,5))


if __name__=='__main__':unittest.main()
