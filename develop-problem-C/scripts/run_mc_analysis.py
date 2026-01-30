"""
运行蒙特卡洛鲁棒性分析。
输出概率型“人气偏离”指标（而非二值分类）。
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "2026_MCM_Problem_C_Data.csv"
sys.path.insert(0, str(SRC_DIR))

from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, RankCPEngine
from dwts_model.sampling import MonteCarloRobustnessAnalyzer
from dwts_model.config import OUTPUT_DIR


def _tighten_rank_intervals(
    interval_bounds: Dict[str, Tuple[float, float]],
    week_context,
    tightening_factor: float = 0.12,
) -> Dict[str, Tuple[float, float]]:
    """
    启发式正则化：对排名制赛季进行区间收缩。

    排名制的可行域更宽（粉丝排名是潜变量），
    适度收缩可稳定估计，用于敏感性分析。
    """
    tightened = {}
    judge_ranks = week_context.judge_ranks
    all_contestants = list(week_context.active_set)

    for contestant, (lower, upper) in interval_bounds.items():
        width = upper - lower

        if contestant in judge_ranks:
            contestant_judge_rank = judge_ranks[contestant]
            n_contestants = len(all_contestants)
            relative_rank = contestant_judge_rank / n_contestants
            adaptive_factor = tightening_factor * (0.5 + 0.5 * relative_rank)
        else:
            adaptive_factor = tightening_factor

        midpoint = (lower + upper) / 2
        new_width = width * (1 - adaptive_factor)
        new_lower = max(0.001, midpoint - new_width / 2)
        new_upper = min(0.999, midpoint + new_width / 2)
        tightened[contestant] = (new_lower, new_upper)

    return tightened


def run_mc_robustness_analysis(
    seasons=None,
    n_samples=10000,
    output_file="mc_robustness_results.csv",
    use_regularization: bool = False,
    tightening_factor: float = 0.12,
):
    """
    对指定赛季执行蒙特卡洛鲁棒性分析。

    Args:
        seasons: 赛季列表（None 表示全量）
        n_samples: 每次淘汰的有效样本数
        output_file: 结果 CSV 文件名
        use_regularization: 是否启用“区间收缩”（仅排名制）
        tightening_factor: 收缩比例
    """
    print("=" * 60)
    print("蒙特卡洛鲁棒性分析")
    print("=" * 60)
    print(f"每次淘汰样本数: {n_samples}")
    print(f"启发式正则化: {'开启' if use_regularization else '关闭'}")
    print()

    print("加载数据...")
    loader = DWTSDataLoader(str(DATA_PATH))
    loader.load()

    manager = ActiveSetManager(loader)
    manager.build_all_contexts()

    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()
    mc_analyzer = MonteCarloRobustnessAnalyzer(
        n_samples=n_samples,
        burnin=1000,
        thin=5,
    )

    all_results = []
    season_list = seasons if seasons else manager.get_all_seasons()

    for season in tqdm(season_list, desc="分析赛季"):
        context = manager.get_season_context(season)

        if context.voting_method == "percent":
            inversion_result = lp_engine.solve(context)
            method = "percent"
        else:
            inversion_result = cp_engine.solve(context)
            method = "rank"

        for week, week_ctx in context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            eliminated_list = week_ctx.eliminated
            if not eliminated_list:
                continue

            for eliminated in (
                eliminated_list if isinstance(eliminated_list, list) else [eliminated_list]
            ):
                week_estimates = inversion_result.week_results.get(week, {})
                interval_bounds = {}

                for contestant in week_ctx.active_set:
                    est = week_estimates.get(contestant)
                    if est:
                        interval_bounds[contestant] = (est.lower_bound, est.upper_bound)
                    else:
                        interval_bounds[contestant] = (0.01, 0.99)

                if method == "rank" and use_regularization:
                    interval_bounds = _tighten_rank_intervals(
                        interval_bounds=interval_bounds,
                        week_context=week_ctx,
                        tightening_factor=tightening_factor,
                    )

                try:
                    mc_result = mc_analyzer.analyze_elimination(
                        season=season,
                        week=week,
                        eliminated=eliminated,
                        week_context=week_ctx,
                        interval_bounds=interval_bounds,
                        voting_method=method,
                        has_judges_save=context.has_judges_save,
                    )

                    all_results.append(
                        {
                            "season": season,
                            "week": week,
                            "contestant": eliminated,
                            "voting_method": method,
                            "p_wrongful": mc_result.p_wrongful,
                            "p_correct": mc_result.p_correct,
                            "ci_lower": mc_result.ci_lower,
                            "ci_upper": mc_result.ci_upper,
                            "n_samples": mc_result.n_samples,
                            "wrongful_count": mc_result.wrongful_count,
                            "correct_count": mc_result.correct_count,
                            "attempts": mc_result.attempts,
                            "acceptance_rate": mc_result.acceptance_rate,
                            "classification": mc_result.get_classification(threshold=0.05),
                            "fan_vote_lower": mc_result.fan_vote_lower,
                            "fan_vote_upper": mc_result.fan_vote_upper,
                            "fan_vote_mean": mc_result.mean_fan_vote,
                            "fan_vote_median": mc_result.median_fan_vote,
                        }
                    )
                except Exception as exc:
                    print(f"  [警告] S{season} W{week} {eliminated}: {exc}")
                    continue

    df = pd.DataFrame(all_results)
    output_path = OUTPUT_DIR / output_file
    df.to_csv(output_path, index=False)

    print(f"\n完成：结果已保存到 {output_path}")

    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)

    print(f"分析淘汰数: {len(df)}")
    print(f"平均 P(Wrongful): {df['p_wrongful'].mean():.3f}")
    print(f"中位数 P(Wrongful): {df['p_wrongful'].median():.3f}")

    print("\n分类分布:")
    print(df["classification"].value_counts())

    print("\n最高 10 个可疑淘汰:")
    top_wrongful = df.nlargest(10, "p_wrongful")[
        ["season", "week", "contestant", "p_wrongful", "ci_lower", "ci_upper"]
    ]
    print(top_wrongful.to_string(index=False))

    print("\n按赛季汇总（平均 P(Wrongful)）:")
    season_summary = df.groupby("season")["p_wrongful"].agg(["mean", "median", "count"])
    season_summary = season_summary.sort_values("mean", ascending=False).head(10)
    print(season_summary)

    return df


def compare_with_interval_robust():
    """与旧版区间鲁棒分类进行对比"""
    print("\n" + "=" * 60)
    print("对比：蒙特卡洛 vs 区间鲁棒")
    print("=" * 60)

    mc_df = pd.read_csv(OUTPUT_DIR / "mc_robustness_results.csv")
    mc_counts = mc_df["classification"].value_counts()

    print("\n蒙特卡洛分类（阈值=5%）：")
    for cls, count in mc_counts.items():
        pct = count / len(mc_df) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    print("\n区间鲁棒旧方法（参考）:")
    print("  Definite-Wrongful: ~40 (17.4%)")
    print("  Possible-Wrongful: ~37 (16.1%)")
    print("  Definite-Safe: ~190 (66.5%)")

    print("\n要点：")
    print("  - 蒙特卡洛给出连续概率而非硬分类")
    print("  - 可调阈值以权衡精确率/召回率")
    print("  - 置信区间刻画不确定性")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo Robustness Analysis")
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=5000,
        help="Number of Monte Carlo samples (default: 5000)",
    )
    parser.add_argument(
        "--seasons",
        "-s",
        type=str,
        default=None,
        help='Seasons to analyze (e.g., "1-10" or "32,33")',
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with interval-robust method",
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="Apply heuristic regularization (interval tightening) for rank seasons",
    )
    parser.add_argument(
        "--tightening-factor",
        type=float,
        default=0.12,
        help="Interval tightening factor when --regularize is set (default: 0.12)",
    )

    args = parser.parse_args()

    seasons = None
    if args.seasons:
        if "-" in args.seasons:
            start, end = map(int, args.seasons.split("-"))
            seasons = list(range(start, end + 1))
        else:
            seasons = [int(s) for s in args.seasons.split(",")]

    results_df = run_mc_robustness_analysis(
        seasons=seasons,
        n_samples=args.samples,
        use_regularization=args.regularize,
        tightening_factor=args.tightening_factor,
    )

    if args.compare:
        compare_with_interval_robust()

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)
