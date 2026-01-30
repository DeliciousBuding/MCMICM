"""
DWTS 模型全局配置
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

# 自动创建目录
for d in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class SeasonConfig:
    """赛季规则配置（依据题目描述）"""

    # 规则分段
    RANK_SEASONS_EARLY: Tuple[int, ...] = (1, 2)  # 早期排名制
    PERCENT_SEASONS: Tuple[int, ...] = tuple(range(3, 28))  # S3-S27 百分制
    RANK_SEASONS_LATE: Tuple[int, ...] = tuple(range(28, 35))  # S28-S34 排名 + 评委拯救

    # 评委拯救开始赛季
    JUDGES_SAVE_START: int = 28

    # 全明星赛季
    ALLSTAR_SEASON: int = 15

    # 争议案例（用于分析）
    CONTROVERSY_CASES: Dict[int, List[str]] = field(
        default_factory=lambda: {
            2: ["Jerry Rice"],      # 亚军但评委分偏低
            4: ["Billy Ray Cyrus"], # 多周垫底仍高位
            11: ["Bristol Palin"],  # 多次评委最低
            27: ["Bobby Bones"],    # 低分夺冠
        }
    )

    def get_voting_method(self, season: int) -> str:
        """判断赛季的投票机制"""
        if season in self.RANK_SEASONS_EARLY or season in self.RANK_SEASONS_LATE:
            return "rank"
        return "percent"

    def has_judges_save(self, season: int) -> bool:
        """判断是否含评委拯救"""
        return season >= self.JUDGES_SAVE_START


@dataclass
class ModelConfig:
    """模型超参数配置"""

    # 线性规划设置
    LP_SLACK_WEIGHT: float = 1.0  # 松弛变量 L1 权重
    LP_TOLERANCE: float = 1e-6    # 数值容差

    # 约束规划设置
    CP_TIME_LIMIT: int = 60      # 秒
    CP_NUM_WORKERS: int = 4      # 并行工作线程

    # 蒙特卡洛马尔可夫链设置
    MCMC_SAMPLES: int = 10000
    MCMC_BURNIN: int = 2000
    MCMC_THIN: int = 10

    # 狄利克雷先验
    DIRICHLET_ALPHA_BASE: float = 1.0
    DIRICHLET_JUDGE_CORRELATION: float = 0.3

    # 自助法设置
    BOOTSTRAP_SAMPLES: int = 100

    # 区间估计
    INTERVAL_CONFIDENCE: float = 0.95


# 全局实例
SEASON_CONFIG = SeasonConfig()
MODEL_CONFIG = ModelConfig()
