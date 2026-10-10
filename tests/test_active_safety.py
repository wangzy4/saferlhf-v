"""Exhaustive finite-pool identities + numeric ONS/certification checks."""
import importlib.util
import itertools
from pathlib import Path
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1]/'safe_rlhf_v/utils/active_safety.py'
spec = importlib.util.spec_from_file_location('active_safety',PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class ActiveSafetyMath(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[.4,1.],[-.9,1.],[.7,1.]])
        self.f = np.array([.1,.8,.6])
        self.gram = self.v@self.v.T

    def test_all_labels_and_adaptive_two_queries_unbiased_and_martingale_variance(self):
        n,b = 3,2
        weights = core.lure_weights(n,b)
        for labels in itertools.product((0,1),repeat=n):
            C = np.array(labels)
            target = C@self.v/n
            mean = np.zeros(2)
            second = cross = variance_rhs = 0.
            U = list(range(n))
            vv,vc,cc = core.centered_geometry(self.gram,self.f,U)
            beta,q,_ = core.joint_design(vv,vc,cc,self.f,.2,n)
            for i in U:
                cost1 = core.corrected_costs(self.f,U,{},i,C[i],beta,q)
                H1 = cost1@self.v/n
                left = [k for k in U if k!=i]
                vv2,vc2,cc2 = core.centered_geometry(self.gram,self.f,left)
                # Adaptive to previous label; this is not iid sampling.
                m2 = np.array([.15 if C[i] else .85]*len(left))
                beta2,q2,_ = core.joint_design(vv2,vc2,cc2,m2,.2,n)
                for pos,k in enumerate(left):
                    cost2 = core.corrected_costs(self.f,left,{i:C[i]},k,C[k],beta2,q2)
                    H2 = cost2@self.v/n
                    prob = q[i]*q2[pos]
                    aggregate = weights[0]*H1+weights[1]*H2
                    mean += prob*aggregate
                    second += prob*np.sum((aggregate-target)**2)
                    cross += prob*np.dot(H1-target,H2-target)
                    variance_rhs += prob*(weights[0]**2*np.sum((H1-target)**2)+weights[1]**2*np.sum((H2-target)**2))
            np.testing.assert_allclose(mean,target,atol=1e-12)
            self.assertAlmostEqual(cross,0.,places=12)
            self.assertAlmostEqual(second,variance_rhs,places=12)

    def test_equation9_centering_and_exact_design_variance(self):
        U = [1,2]
        C = np.array([1,0,1])
        beta,q = .63,np.array([.3,.7])
        center = self.f[U,None]*self.v[U]-(self.f[U,None]*self.v[U]).mean(0)
        target = C@self.v/3
        variance = 0.
        for pos,i in enumerate(U):
            eff = core.corrected_costs(self.f,U,{0:1},i,C[i],beta,q)
            H = eff@self.v/3
            direct = self.v[0]/3+(C[i]*self.v[i]-beta*center[pos])/(3*q[pos])
            np.testing.assert_allclose(H,direct)
            variance += q[pos]*np.sum((H-target)**2)
        rhs = np.sum(np.sum((C[U,None]*self.v[U]-beta*center)**2,axis=1)/q)/9-np.sum((C[U]@self.v[U]/3)**2)
        self.assertAlmostEqual(variance,rhs,places=12)

    def test_uniform_lure_and_census(self):
        for n in (3,8,32):
            for b in range(1,n+1):
                f = np.linspace(0,1,n)
                weights = core.lure_weights(n,b)
                self.assertAlmostEqual(weights.sum(),1)
                effective = np.zeros(n)
                revealed = {}
                remaining = list(range(n))
                for j in range(b):
                    i,label = remaining[-1],j%2
                    effective += weights[j]*core.corrected_costs(f,remaining,revealed,i,label,0,np.full(len(remaining),1/len(remaining)))
                    revealed[i]=label
                    remaining.remove(i)
                self.assertAlmostEqual(effective.mean(),np.mean(list(revealed.values())))
                if b==n:
                    np.testing.assert_allclose(effective,[revealed[i] for i in range(n)])

    def test_gram_geometry_matches_vectors(self):
        for U in ([0,1,2],[0,2],[1]):
            vv,vc,cc = core.centered_geometry(self.gram,self.f,U)
            center = self.f[U,None]*self.v[U]-(self.f[U,None]*self.v[U]).mean(0)
            np.testing.assert_allclose(vv,np.sum(self.v[U]**2,axis=1))
            np.testing.assert_allclose(vc,np.sum(self.v[U]*center,axis=1),atol=1e-14)
            np.testing.assert_allclose(cc,np.sum(center**2,axis=1),atol=1e-14)

    def test_joint_solver_vs_independent_scipy(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(17)
        for _ in range(10):
            v = np.c_[rng.normal(size=(8,3)),np.ones(8)]
            f,m = rng.random(8),rng.random(8)
            vv,vc,cc = core.centered_geometry(v@v.T,f,list(range(8)))
            beta,q,_ = core.joint_design(vv,vc,cc,m,.2,8)
            def objective(x):
                return np.sum((m*vv-2*x[0]*m*vc+x[0]**2*cc)/x[1:])/64
            result = minimize(objective,np.r_[.5,np.full(8,1/8)],method='SLSQP',bounds=[(0,1)]+[(.2/8,1)]*8,
                constraints=[{'type':'eq','fun':lambda x:x[1:].sum()-1}],options={'ftol':1e-12,'maxiter':1000})
            self.assertTrue(result.success,result.message)
            self.assertLessEqual(abs(objective(np.r_[beta,q])-result.fun),1e-8)
            self.assertTrue(np.all(q>=.2/8-1e-14))
        np.testing.assert_allclose(core.allocation(np.zeros(3),.2),[1/3]*3)
        np.testing.assert_allclose(core.allocation([1,0,0],1),[1/3]*3)

    def test_perfect_proxy_zero_variance(self):
        f = np.array([0,1,1.])
        vv,vc,cc = core.centered_geometry(self.gram,f,[0,1,2])
        beta,q,_ = core.joint_design(vv,vc,cc,f,.2,3)
        for i in range(3):
            eff = core.corrected_costs(f,[0,1,2],{},i,int(f[i]),beta,q)
            np.testing.assert_allclose(eff@self.v/3,f@self.v/3,atol=1e-7)

    def test_ons_prediction_bounds_metric_projection_regret(self):
        rng = np.random.default_rng(8)
        ons = core.ONSCalibrator(4,.2)
        comparator = np.array([.2,-.5,.4,.3])
        regret = 0.
        for _ in range(200):
            phi = rng.normal(size=4)
            phi /= max(1,np.linalg.norm(phi))
            C = int(rng.integers(2))
            prediction = ons.predict(phi)
            result = ons.update(phi,C,.9,10,.02)
            weight = result['importance_loss_weight']
            regret += weight*((prediction-C)**2-((1+phi@comparator)/2-C)**2)
            self.assertLessEqual(np.linalg.norm(ons.u),1+1e-12)
            self.assertGreaterEqual(ons.predict(phi),0)
            self.assertLessEqual(ons.predict(phi),1)
        self.assertLessEqual(regret,(1+4*np.log(1+200/4))/.2)
        np.testing.assert_allclose(core.ONSCalibrator(16,.2).A,np.eye(16)*25)
        with self.assertRaises(ValueError):
            ons.predict(np.ones(4))

    def test_certificate_range_martingale_and_mixture(self):
        U,C = [0,1,2],np.array([1,0,1])
        beta,q = .7,np.array([.2,.3,.5])
        lo,hi,bet = core.certification_range(self.f,U,{},beta,q)
        expectation = 0.
        for i in U:
            x = core.corrected_costs(self.f,U,{},i,C[i],beta,q).mean()
            self.assertGreaterEqual(x,lo-1e-12)
            self.assertLessEqual(x,hi+1e-12)
            expectation += q[i]*(1+bet*(C.mean()-x))
        self.assertAlmostEqual(expectation,1)
        bound = core.certify_upper([0]*1024,[.5]*1024,32768)
        self.assertLess(bound['risk_upper'],.05)
        self.assertEqual(core.certify_upper([1],[.5],1)['risk_upper'],1)
        a = core.mixture_weight(.3,.01,.05)
        self.assertAlmostEqual((1-a)*.3+a*.01,.05)
        self.assertEqual(core.mixture_weight(.03,.01,.05),0)
        with self.assertRaises(ValueError):
            core.mixture_weight(.3,.05,.05)

    def test_reference_binomial_bound_vs_scipy(self):
        from scipy.stats import beta
        for n in (8,512):
            for k in (0,1,n//2,n-1,n):
                expected=1. if k==n else float(beta.ppf(.99,k+1,n-k))
                self.assertAlmostEqual(core.reference_upper(k,n),expected,places=12)
        self.assertAlmostEqual(core.reference_upper(0,512),1-.01**(1/512),places=15)
        self.assertLess(core.reference_upper(0,512),.05)
        for k,n in ((-1,512),(513,512),(1.5,512),(0,0)):
            with self.assertRaises(ValueError):
                core.reference_upper(k,n)

    def test_invalid_inputs_fail_closed(self):
        for probability in (np.nan, np.inf, 1.01, 0.):
            with self.assertRaises(ValueError):
                core.ONSCalibrator(2).update(np.array([.1,.2]),1,.5,3,probability)
        for estimate in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                core.dual_update(1,estimate,.05,10,1,0)
        with self.assertRaises(ValueError):
            core.lure_weights(8,2.5)
        with self.assertRaises(ValueError):
            core.centered_geometry(self.gram,self.f,[])
        vv,vc,cc=core.centered_geometry(self.gram,self.f,[0,1,2])
        with self.assertRaises(ValueError):
            core.joint_design(vv,vc,cc,self.f,.2,3,tolerance=0)
        with self.assertRaises(ValueError):
            core.joint_design(vv,vc,cc,self.f,0,3,uniform=True)
        with self.assertRaises(ValueError):
            core.certification_range(self.f,[0,1,2],{},np.nan,np.full(3,1/3))
        with self.assertRaises(ValueError):
            core.certification_range(self.f,[0,1,2],{},.5,np.ones(3))
        with self.assertRaises(ValueError):
            core.certify_upper([0],[.5],np.nan)

    def test_unclipped_cost_and_dual(self):
        cost = core.corrected_costs(self.f,[0,1,2],{},0,1,1,np.array([.1,.4,.5]))
        self.assertGreater(cost.max(),1)
        self.assertEqual(core.dual_update(1.,-5.,.05,10.,1.,0),0)
        self.assertEqual(core.dual_update(1.,100.,.05,10.,1.,0),10)


if __name__=='__main__':
    unittest.main()
