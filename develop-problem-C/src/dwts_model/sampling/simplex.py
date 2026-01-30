"""
单纯形工具：投票比例的投影与采样
"""
import numpy as np


class SimplexProjection:
    """单纯形相关操作"""

    @staticmethod
    def project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        投影到概率单纯形（Duchi 等算法）。
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)

    @staticmethod
    def sample_uniform_simplex(n: int, rng=None) -> np.ndarray:
        """
        在 n 维单纯形上均匀采样。
        """
        if rng is None:
            rng = np.random.default_rng()
        e = rng.exponential(scale=1.0, size=n)
        return e / np.sum(e)
