# 与星共舞粉丝投票反演系统
# 美赛 2026 C 题：与星共舞数据集
#
# 架构（主流程保留部分）：
# - etl/：防御式数据工程（FSM、活跃集）
# - engines/：双核反演（百分制 LP、排名制 MILP）
# - sampling/：截断贝叶斯 + MCMC 采样（含时间平滑）
# - analysis/：反事实评估、特征归因、机制设计
__version__ = "1.0.0"
__author__ = "MCM Team 2026"
