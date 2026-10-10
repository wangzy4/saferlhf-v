"""CPU adapter tests (no DeepSpeed/model downloads) plus runner contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]


def method(name, namespace):
    source=ROOT/'safe_rlhf_v/trainers/text_image_to_text/active_safe_rlhf_v.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,str(source),'exec'),namespace)
    return namespace[name]


class PPOIntegration(unittest.TestCase):
    def test_corrected_costs_and_rewards_not_clipped(self):
        namespace={'torch':torch,'last_valid_indices':lambda mask:mask.long().sum(-1)-1}
        fn=method('add_kl_divergence_regularization_with_cost',namespace)
        t=SimpleNamespace(kl_coeff=.02)
        reward=torch.tensor([.6,.2])
        cost=torch.tensor([-12.,125.])
        mask=torch.tensor([[True,True,False],[True,True,True]])
        old=torch.tensor([[-.8,-.5,17.],[-.8,-.4,-.1]])
        ref=old-.2
        r,c=fn(t,reward,cost,old,ref,mask)
        torch.testing.assert_close(c.sum(-1),cost)
        self.assertEqual(c[0,2],0)
        torch.testing.assert_close(r.sum(-1),reward-.02*.2*mask.sum(-1))
        # KL has no place in the target binary human-risk cost.
        self.assertEqual(c[0,0],0)

    def test_fixed_multiplier_no_label_normalization(self):
        namespace={'torch':torch,'masked_mean':lambda x,m:(x*m).sum()/m.sum()}
        fn=method('actor_loss_fn_with_cost',namespace)
        t=SimpleNamespace(active_multiplier=2.,clip_range_ratio=.2)
        lp=torch.zeros((2,3),requires_grad=True)
        old=torch.zeros_like(lp)
        r=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        c=torch.tensor([[2.,-3.,5.],[1.,2.,3.]])
        mask=torch.ones_like(lp,dtype=torch.bool)
        loss=fn(t,lp,old,r,c,mask)
        torch.testing.assert_close(loss,-(r-2*c).mean())
        loss.backward()
        torch.testing.assert_close(lp.grad,-(r-2*c)/6)


if __name__=='__main__':unittest.main()
