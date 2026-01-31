"""
截断贝叶斯推断 + MCMC 采样（带时间平滑）。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    total = np.sum(vec)
    return vec / total if total > 0 else vec


def _within_bounds(sample: Dict[str, float], bounds: Dict[str, Tuple[float, float]], tol: float = 1e-12) -> bool:
    for name, (lower, upper) in bounds.items():
        if name not in sample:
            continue
        val = sample[name]
        if val < lower - tol or val > upper + tol:
            return False
    return True


def _initial_point(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    mid = {k: (v[0] + v[1]) / 2 for k, v in bounds.items()}
    arr = np.array(list(mid.values()))
    arr = _normalize(arr)
    return {k: arr[i] for i, k in enumerate(mid.keys())}


def 计算_hdi(samples: np.ndarray, cred: float = 0.95) -> Tuple[float, float]:
    """计算一维样本的 HDI 区间。"""
    if samples.size == 0:
        return 0.0, 1.0
    sorted_vals = np.sort(samples)
    n = len(sorted_vals)
    interval = int(np.floor(cred * n))
    if interval < 1:
        return float(sorted_vals[0]), float(sorted_vals[-1])
    widths = sorted_vals[interval:] - sorted_vals[: n - interval]
    idx = int(np.argmin(widths))
    return float(sorted_vals[idx]), float(sorted_vals[idx + interval])


def 采样_单周(
    interval_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 2000,
    burnin: int = 500,
    thin: int = 5,
    smooth_lambda: float = 10.0,
    prev_sample: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, float]]:
    """
    单周 MCMC 采样：在区间内采样，并用平滑先验约束相邻周变化。
    """
    rng = rng or np.random.default_rng()

    names = list(interval_bounds.keys())
    current = _initial_point(interval_bounds)
    if prev_sample:
        prev_vec = np.array([prev_sample.get(n, 0.0) for n in names])
        prev_vec = _normalize(prev_vec)
    else:
        prev_vec = None

    samples: List[Dict[str, float]] = []
    max_attempts = (n_samples + burnin) * 20
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        cur_vec = np.array([current[n] for n in names])
        alpha = np.maximum(cur_vec * 50.0, 1e-3)
        proposal_vec = rng.dirichlet(alpha)
        proposal = {n: proposal_vec[i] for i, n in enumerate(names)}

        if not _within_bounds(proposal, interval_bounds):
            continue

        if prev_vec is None:
            accept = True
        else:
            cur_penalty = -smooth_lambda * np.sum((cur_vec - prev_vec) ** 2)
            prop_penalty = -smooth_lambda * np.sum((proposal_vec - prev_vec) ** 2)
            log_ratio = prop_penalty - cur_penalty
            accept = np.log(rng.random()) < log_ratio

        if accept:
            current = proposal

        if attempts > burnin and (attempts - burnin) % thin == 0:
            samples.append(current.copy())

    # 低接受率时补充均匀采样
    if len(samples) < max(1, n_samples // 2):
        for _ in range(n_samples - len(samples)):
            sample = _sample_uniform_within_bounds(interval_bounds, rng)
            if sample:
                samples.append(sample)

    return samples[:n_samples]


def _sample_uniform_within_bounds(
    interval_bounds: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
) -> Optional[Dict[str, float]]:
    names = list(interval_bounds.keys())
    remaining = 1.0
    sample = {}

    for i, name in enumerate(names[:-1]):
        lower, upper = interval_bounds[name]
        lower = max(lower, 0.0)
        upper = min(upper, remaining)
        if upper < lower:
            return None
        val = rng.uniform(lower, upper)
        sample[name] = val
        remaining -= val

    last = names[-1]
    lower, upper = interval_bounds[last]
    if remaining < lower or remaining > upper:
        return None
    sample[last] = remaining

    return sample


def 汇总后验(samples: List[Dict[str, float]], cred: float = 0.95) -> Dict[str, Tuple[float, float, float]]:
    """
    返回每位选手的 (mean, hdi_low, hdi_high)。
    """
    if not samples:
        return {}
    names = list(samples[0].keys())
    arr = {name: [] for name in names}
    for s in samples:
        for name in names:
            arr[name].append(s.get(name, 0.0))

    summary = {}
    for name in names:
        values = np.array(arr[name])
        mean_val = float(np.mean(values))
        hdi_low, hdi_high = 计算_hdi(values, cred)
        summary[name] = (mean_val, hdi_low, hdi_high)

    return summary
