"""Numerical and protocol checks for the dedicated CIFAR-10 campaign."""
import os
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from campaign.mcl_cifar10_ablation import (
    DEFAULT, experimental_class, make_loaders, metric_dict, preserved_rng,
    evaluate_stage, variants,
)
from MCL.mcl import MCL
from models import ContinualLearningModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*4*4,512)

    def get_features(self,x):
        return torch.relu(self.fc(x.flatten(1)))


def model(scenario='class_il'):
    torch.manual_seed(42)
    m=ContinualLearningModel(TinyBackbone(),512,2,5,scenario)
    for t in range(5):
        m.ensure_head(t)
    m.seen_upto=1
    return m


class CampaignTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_reference_loss_and_gradients_match_original(self):
        for scenario in ['class_il','task_il']:
            original=MCL(model(scenario),torch.device('cpu'),scenario=scenario)
            experiment=experimental_class()(copy.deepcopy(original.model),torch.device('cpu'),scenario=scenario)
            original.teacher=copy.deepcopy(original.model).eval()
            experiment.teacher=copy.deepcopy(original.teacher)
            with torch.no_grad():
                original.model.backbone.fc.weight.add_(.01)
                experiment.model.backbone.fc.weight.add_(.01)
            x=torch.randn(4,3,4,4)
            y=torch.tensor([0,1,0,1])+(2 if scenario=='class_il' else 0)
            losses=[]
            for method in [original,experiment]:
                z=method.logits(x,1)
                loss=method.criterion(z,y)+method.extra_loss(x,y,z,1)
                loss.backward()
                losses.append(loss)
            torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
            for (name,a),(_,b) in zip(original.model.named_parameters(),experiment.model.named_parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)

    def test_selected_width_distillation_matches_manual_kl(self):
        import torch.nn.functional as F
        method=experimental_class()(model(),torch.device('cpu'),distill_widths=[32])
        method.teacher=copy.deepcopy(method.model).eval()
        with torch.no_grad():
            method.model.backbone.fc.weight.add_(.02)
        x=torch.randn(4,3,4,4);y=torch.tensor([2,3,2,3])
        z=method.logits(x,1)
        actual=method.criterion(z,y)+method.extra_loss(x,y,z,1)
        ce=torch.stack([F.cross_entropy(method._doll(method.model.heads['1'],method._features,m),y-2)
                        for m in method.granularities]).mean()
        tf=method.teacher.get_features(x)
        zs=method._heads_of(method.model,[0],method._features,32)
        zt=method._heads_of(method.teacher,[0],tf,32)
        expected=ce+4*F.kl_div(F.log_softmax(zs/2,dim=1),F.softmax(zt/2,dim=1),reduction='batchmean')
        torch.testing.assert_close(actual,expected)

    def test_global_ce_uses_seen_classes(self):
        method=experimental_class()(model(),torch.device('cpu'),global_ce=True,sdft_lambda=0.)
        x=torch.randn(4,3,4,4);y=torch.tensor([2,3,2,3])
        z=method.logits(x,1)
        self.assertEqual(z.shape,(4,4))
        loss=method.criterion(z,y)+method.extra_loss(x,y,z,1)
        loss.backward()
        self.assertGreater(float(method.model.heads['0'].weight.grad.abs().sum()),0)

    def test_main_readout_matches_production_predict(self):
        from torch.utils.data import DataLoader,TensorDataset
        method=experimental_class()(model(),torch.device('cpu'))
        for c in range(4):
            method.density_stats[c]=(torch.randn(512),torch.rand(512)+.1)
        triples=[]
        for task in range(5):
            ds=TensorDataset(torch.randn(4,3,4,4),torch.tensor([0,1,0,1])+2*task)
            loader=DataLoader(ds,batch_size=4)
            triples.append([loader]*3)
        rows,_=evaluate_stage(method,triples,1,{**DEFAULT,'scenario':'class_il'},[0.,.25,1.])
        for task in range(2):
            expected=method.evaluate(triples[task][2],task)
            self.assertEqual(rows['main'][task],expected)
        with method.model.full_output_space(5):
            for task in range(2,5):
                self.assertEqual(rows['main'][task],method.evaluate(triples[task][2],task))

    def test_metrics_use_random_baseline_and_forgetting(self):
        metrics=metric_dict([[.9,.6],[.5,.8]],[.5,.5])
        self.assertAlmostEqual(metrics['ACC'],65)
        self.assertAlmostEqual(metrics['BWT'],-40)
        self.assertAlmostEqual(metrics['FWT'],10)

    def test_data_splits_fixed_disjoint_and_nested(self):
        root=os.environ.get('GTEP_DATA_ROOT', './data')
        if not Path(root,'cifar-10-batches-py').exists():
            self.skipTest('CIFAR-10 not available')
        config={**DEFAULT,'seed':42,'scenario':'class_il','workers':0}
        _,full=make_loaders(config,root)
        _,small=make_loaders({**config,'train_fraction':.1,'seed':43},root)
        for c,parts in full['per_class'].items():
            self.assertEqual([len(parts[k]) for k in ['train','val','test']],[4500,500,1000])
            self.assertFalse(set(parts['train']) & set(parts['val']))
            self.assertTrue(set(small['per_class'][c]['train']).issubset(parts['train']))
            self.assertEqual(small['per_class'][c]['val'],parts['val'])
            self.assertEqual(small['per_class'][c]['test'],parts['test'])

    def test_rng_restoration_and_variant_names(self):
        torch.manual_seed(4)
        state=torch.get_rng_state()
        with preserved_rng():
            torch.rand(20)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        names=[v['name'] for v in variants()]
        self.assertEqual(len(names),len(set(names)))
        self.assertIn('no_weight_alignment',names)


if __name__=='__main__':
    unittest.main()
