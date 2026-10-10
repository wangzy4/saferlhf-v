"""Algorithm v2.1: NumPy-only acquisition, ONS, LURE and certification.

All quantities are float64 and label-predictable. Corrected costs are NOT
probabilities: callers must neither clip nor normalize them. Geometry must be
that of actual trainable policy scores to claim full-gradient variance results.
"""
from __future__ import annotations

import numpy as np


def _array(x):
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite input')
    return a


def allocation(a, epsilon):
    """Eq. (7), water filling with a lower bound (not mixture sampling)."""
    a = _array(a)
    if a.ndim != 1 or not len(a) or np.any(a < -1e-10) or not 0 < epsilon <= 1:
        raise ValueError('Invalid allocation input')
    m = len(a)
    if epsilon == 1 or np.max(a) <= 0:
        return np.full(m, 1 / m)
    roots = np.sqrt(np.maximum(a, 0))
    q = np.zeros(m)
    free = np.ones(m, dtype=bool)
    floor = epsilon / m
    while free.any():
        mass = 1 - q[~free].sum()
        total = roots[free].sum()
        trial = mass * roots[free] / total if total else np.full(free.sum(), mass / free.sum())
        low = trial < floor
        ids = np.flatnonzero(free)
        if not low.any():
            q[ids] = trial
            break
        q[ids[low]] = floor
        free[ids[low]] = False
    if not np.isclose(q.sum(), 1) or np.any(q < floor - 1e-14):
        raise ArithmeticError('Allocation normalization failed')
    return q


def centered_geometry(gram, f, remaining):
    """Compute ||v||², <v,c>, ||c||² from exact score Gram, without storing c."""
    gram, f = _array(gram), _array(f)
    ids = np.asarray(remaining, dtype=int)
    if (f.ndim != 1 or gram.shape != (len(f), len(f)) or not len(ids)
            or len(set(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(f)))
            or np.any((f < 0) | (f > 1))):
        raise ValueError('Invalid Gram/pool geometry')
    g = gram[np.ix_(ids, ids)]
    fu = f[ids]
    vfmean = g @ fu / len(ids)
    mean2 = float(fu @ vfmean / len(ids))
    vv = g.diagonal().copy()
    vc = fu * vv - vfmean
    cc = fu**2 * vv - 2 * fu * vfmean + mean2
    if np.min(cc) < -1e-7 * max(1, np.max(vv)):
        raise ValueError('Gram matrix has invalid centered geometry')
    return vv, vc, np.maximum(cc, 0)


def joint_design(vv, vc, cc, mhat, epsilon, n, fixed_beta=None, uniform=False, tolerance=1e-9):
    """Global one-dimensional convex search, including both endpoints.

    Returns a numerical bracket width, not a claim of zero optimization error.
    """
    vv, vc, cc, mhat = map(_array, (vv, vc, cc, mhat))
    if not (vv.shape == vc.shape == cc.shape == mhat.shape) or vv.ndim != 1 or not len(vv):
        raise ValueError('Geometry shapes differ')
    if (np.any((mhat < 0) | (mhat > 1)) or np.any(vv < 0) or np.any(cc < 0)
            or not np.isfinite(n) or n != int(n) or n < len(vv)
            or not 0 < epsilon <= 1 or not 1e-14 <= tolerance < 1):
        raise ValueError('Invalid prediction/pool size/solver configuration')
    def evaluate(beta):
        a = mhat * vv - 2 * beta * mhat * vc + beta**2 * cc
        if a.min() < -1e-8 * max(1, vv.max()):
            raise ValueError('Negative second moment')
        a = np.maximum(a, 0)
        q = np.full(len(a), 1 / len(a)) if uniform else allocation(a, epsilon)
        return float(np.sum(a / q) / n**2), q
    if fixed_beta is not None:
        if not 0 <= fixed_beta <= 1:
            raise ValueError('beta outside [0,1]')
        value, q = evaluate(fixed_beta)
        return float(fixed_beta), q, {'objective': value, 'beta_bracket_width': 0.0, 'fixed_beta': True}
    lo, hi = 0., 1.
    ratio = (np.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fl, _ = evaluate(left)
    fr, _ = evaluate(right)
    while hi - lo > tolerance:
        if fl <= fr:
            hi, right, fr = right, left, fl
            left = hi - ratio * (hi - lo)
            fl, _ = evaluate(left)
        else:
            lo, left, fl = left, right, fr
            right = lo + ratio * (hi - lo)
            fr, _ = evaluate(right)
    beta = min([0., 1., (lo + hi) / 2], key=lambda b: evaluate(b)[0])
    value, q = evaluate(beta)
    return beta, q, {'objective': value, 'beta_bracket_width': hi - lo, 'fixed_beta': False}


def lure_weights(n, b):
    if not np.isfinite([n, b]).all() or n != int(n) or b != int(b) or not 1 <= b <= n:
        raise ValueError('Require integral 1 <= b <= N')
    if b == n:
        out = np.zeros(b)
        out[-1] = 1
        return out
    j = np.arange(1, b + 1)
    return n * (n - b) / (b * (n - j) * (n - j + 1))


def corrected_costs(f, remaining, revealed, selected, label, beta, q):
    """Eq. (9). revealed excludes the current query; q follows remaining order."""
    f, q = _array(f), _array(q)
    ids = np.asarray(remaining, dtype=int)
    if (f.ndim != 1 or not len(f) or np.any((f < 0) | (f > 1))
            or len(set(ids)) != len(ids) or set(ids) & set(revealed)
            or set(ids) | set(revealed) != set(range(len(f)))):
        raise ValueError('Remaining/revealed do not partition a bounded proxy pool')
    if label not in (0, 1) or not 0 <= beta <= 1 or q.shape != ids.shape or np.any(q <= 0) or not np.isclose(q.sum(), 1):
        raise ValueError('Invalid query')
    pos = np.flatnonzero(ids == selected)
    if len(pos) != 1:
        raise ValueError('Selected candidate not remaining')
    qi = q[pos[0]]
    out = np.zeros(len(f))
    for i, c in revealed.items():
        if c not in (0, 1):
            raise ValueError('Invalid revealed label')
        out[i] = c
    out[ids] = beta * f[ids] / (len(ids) * qi)
    out[selected] += (label - beta * f[selected]) / qi
    return out


class ONSCalibrator:
    def __init__(self, dimension=16, epsilon=.2):
        if dimension < 1 or not 0 < epsilon <= 1:
            raise ValueError('Invalid ONS configuration')
        self.epsilon = epsilon
        self.u = np.zeros(dimension)
        self.A = np.eye(dimension) / epsilon**2
        self.observations = 0

    def predict(self, features):
        phi = _array(features)
        if phi.ndim < 1 or phi.shape[-1] != len(self.u) or np.any(np.linalg.norm(phi, axis=-1) > 1 + 1e-10):
            raise ValueError('Features must be fixed and bounded by one')
        return (1 + phi @ self.u) / 2

    def update(self, phi, label, k, remaining_count, probability):
        phi = _array(phi)
        if (phi.ndim != 1 or label not in (0, 1) or not 0 <= k <= 1
                or not np.isfinite(remaining_count) or remaining_count != int(remaining_count)
                or remaining_count < 1 or not np.isfinite(probability) or probability > 1
                or probability < self.epsilon / remaining_count - 1e-12):
            raise ValueError('Invalid ONS observation')
        prediction = float(self.predict(phi))
        a = k / (remaining_count * probability)
        g = a * (prediction - label) * phi
        self.A += np.outer(g, g)
        target = self.u - (2 / self.epsilon) * np.linalg.solve(self.A, g)
        if np.linalg.norm(target) <= 1:
            self.u = target
        else:
            eig, basis = np.linalg.eigh(self.A)
            coordinates = basis.T @ target
            def project(lam):
                return basis @ (eig * coordinates / (eig + lam))
            lo, hi = 0., 1.
            while np.linalg.norm(project(hi)) > 1:
                hi *= 2
            for _ in range(80):
                mid = (lo + hi) / 2
                if np.linalg.norm(project(mid)) > 1:
                    lo = mid
                else:
                    hi = mid
            self.u = project(hi)
        self.observations += 1
        return {'prediction_before_label': prediction, 'importance_loss_weight': a}

    def state(self):
        return {'u': self.u.tolist(), 'A': self.A.tolist(), 'epsilon': self.epsilon, 'observations': self.observations}


def dual_update(nu, estimate, delta, ceiling, gamma, iteration):
    _array([nu, estimate, delta, ceiling, gamma, iteration])
    if not 0 <= delta <= 1 or not 0 <= nu <= ceiling or gamma < 0 or iteration < 0 or iteration != int(iteration):
        raise ValueError('Invalid dual configuration')
    return float(np.clip(nu + gamma / (iteration + 1) * estimate - gamma / (iteration + 1) * delta, 0, ceiling))


def certification_range(f, remaining, revealed, beta, q):
    f, q = _array(f), _array(q)
    ids = np.asarray(remaining, dtype=int)
    if (f.ndim != 1 or not len(f) or np.any((f < 0) | (f > 1))
            or not len(ids) or q.shape != ids.shape or np.any(q <= 0)
            or not np.isclose(q.sum(), 1) or not 0 <= beta <= 1
            or len(set(ids)) != len(ids) or set(ids) & set(revealed)
            or set(ids) | set(revealed) != set(range(len(f)))
            or any(c not in (0, 1) for c in revealed.values())):
        raise ValueError('Invalid certification query')
    center = f[ids] - f[ids].mean()
    base = sum(revealed.values()) / len(f)
    low = base - beta * center / (len(f) * q)
    high = base + (1 - beta * center) / (len(f) * q)
    upper = float(high.max())
    return float(low.min()), upper, 1 / (2 * max(1, upper))


def certify_upper(estimates, bets, n, zeta_a=.02, zeta_p=.02):
    x, bets = _array(estimates), _array(bets)
    _array([n, zeta_a, zeta_p])
    if (x.ndim != 1 or x.shape != bets.shape or not len(x) or n != int(n) or n < len(x)
            or np.any(bets <= 0) or not 0 < zeta_a < 1 or not 0 < zeta_p < 1):
        raise ValueError('Invalid certification observations')
    def log_e(u):
        factors = 1 + bets * (u - x)
        if np.any(factors <= 0):
            raise ValueError('Bets violate positivity on [0,1]')
        return float(np.log(factors).sum())
    threshold = np.log(1 / zeta_a)
    if log_e(0) > threshold:
        pool = 0.
    elif log_e(1) <= threshold:
        pool = 1.
    else:
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo + hi) / 2
            if log_e(mid) <= threshold:
                lo = mid
            else:
                hi = mid
        pool = hi
    margin = np.sqrt(np.log(1 / zeta_p) / (2 * n))
    return {'pool_upper': pool, 'population_margin': float(margin), 'risk_upper': float(min(1, pool + margin)), 'failure_probability': zeta_a + zeta_p}


def reference_upper(unsafe_count, audit_count, zeta=.01):
    """One-sided exact binomial (Clopper-Pearson) upper bound.

    Caller must supply independent fresh labels on a FIXED reference policy,
    using the target annotation process. AI-proxy counts do not certify humans.
    Log-CDF inversion avoids requiring SciPy in the NumPy-only math module.
    """
    import math
    _array([unsafe_count, audit_count, zeta])
    if (audit_count != int(audit_count) or audit_count < 1
            or unsafe_count != int(unsafe_count) or not 0 <= unsafe_count <= audit_count
            or not 0 < zeta < 1):
        raise ValueError('Invalid independent reference audit')
    n, k = int(audit_count), int(unsafe_count)
    if k == n:
        return 1.
    if k == 0:
        return float(-np.expm1(np.log(zeta) / n))
    j = np.arange(k + 1)
    coefficients = np.array([math.lgamma(n + 1) - math.lgamma(int(i) + 1)
                             - math.lgamma(n - int(i) + 1) for i in j])
    threshold = np.log(zeta)
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        log_cdf = np.logaddexp.reduce(coefficients + j * np.log(mid) + (n - j) * np.log1p(-mid))
        if log_cdf > threshold:
            lo = mid
        else:
            hi = mid
    return float(hi)


def mixture_weight(candidate_upper, reference_upper, delta):
    if not 0 <= reference_upper < delta <= 1 or not 0 <= candidate_upper <= 1:
        raise ValueError('A valid independently certified reference with U0 < delta is required')
    return 0. if candidate_upper <= delta else (candidate_upper - delta) / (candidate_upper - reference_upper)
