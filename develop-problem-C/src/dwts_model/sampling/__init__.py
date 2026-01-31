# 采样模块：截断贝叶斯 + MCMC
from .bayes_mcmc import 采样_单周, 汇总后验, 计算_hdi

__all__ = [
    "采样_单周",
    "汇总后验",
    "计算_hdi",
]
