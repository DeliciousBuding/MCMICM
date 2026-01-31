"""
一键执行新版框架：
1) 硬约束反演（LP/MILP）
2) 截断贝叶斯 + MCMC（含时间平滑）
3) 反事实评估与指标计算
4) 机制设计与帕累托前沿
5) 特征分析（XGBoost + SHAP）
6) 生成图表并编译论文
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "2026_MCM_Problem_C_Data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(SRC_DIR))

from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, RankCPEngine
from dwts_model.sampling import 采样_单周, 汇总后验
from dwts_model.analysis import 运行反事实评估, 运行帕累托优化, 运行特征分析

# 统一配色
PALETTE = {
    "light_blue": "#90C9E7",
    "cyan_blue": "#219EBC",
    "deep_cyan": "#136783",
    "navy": "#02304A",
    "yellow": "#FEB705",
    "orange": "#FF9E02",
    "dark_orange": "#FA8600",
}


def _run_cmd(cmd, cwd: Path):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _规则一致性(week_ctx, fan_votes: Dict[str, float], method: str, has_judges_save: bool) -> bool:
    contestants = list(week_ctx.active_set)
    if not contestants:
        return False
    if method == "percent":
        combined = {c: 0.5 * week_ctx.judge_percentages.get(c, 0.0) + 0.5 * fan_votes.get(c, 0.0) for c in contestants}
        eliminated = min(combined.items(), key=lambda x: x[1])[0]
        return eliminated in week_ctx.eliminated
    # 排名制
    sorted_fans = sorted(contestants, key=lambda x: fan_votes.get(x, 0.0), reverse=True)
    fan_rank = {c: i + 1 for i, c in enumerate(sorted_fans)}
    combined = {c: fan_rank.get(c, len(contestants)) + week_ctx.judge_ranks.get(c, len(contestants)) for c in contestants}
    required = max(2, len(week_ctx.eliminated)) if has_judges_save else len(week_ctx.eliminated)
    bottom = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:required]
    bottom_set = {c for c, _ in bottom}
    return any(e in bottom_set for e in week_ctx.eliminated)


def _动态权重(week: int, total_weeks: int, start: float = 0.7, end: float = 0.4, pivot: int = 5) -> float:
    """动态自适应权重：前期偏评委，后期偏观众。"""
    if total_weeks <= 1:
        return start
    if week <= pivot:
        return start
    if total_weeks <= pivot:
        return end
    progress = min(max((week - pivot) / (total_weeks - pivot), 0.0), 1.0)
    return start + (end - start) * progress


def run_inversion_and_posterior(
    manager: ActiveSetManager,
    mcmc_samples: int,
    burnin: int,
    thin: int,
    smooth_lambda: float,
) -> pd.DataFrame:
    """硬约束反演 + 截断贝叶斯采样，输出后验汇总。"""
    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()

    interval_records = []
    posterior_records = []

    for season in manager.get_all_seasons():
        context = manager.get_season_context(season)
        engine = lp_engine if context.voting_method == "percent" else cp_engine
        inversion_result = engine.solve(context)

        prev_mean: Optional[Dict[str, float]] = None
        for week, week_ctx in context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            interval_bounds = {}
            for contestant in week_ctx.active_set:
                est = inversion_result.week_results.get(week, {}).get(contestant)
                if est:
                    interval_bounds[contestant] = (est.lower_bound, est.upper_bound)
                else:
                    interval_bounds[contestant] = (0.01, 0.99)

                interval_records.append(
                    {
                        "season": season,
                        "week": week,
                        "contestant": contestant,
                        "voting_method": context.voting_method,
                        "lower": interval_bounds[contestant][0],
                        "upper": interval_bounds[contestant][1],
                        "point": est.point_estimate if est else np.nan,
                    }
                )

            # MCMC 采样
            samples = 采样_单周(
                interval_bounds=interval_bounds,
                n_samples=mcmc_samples,
                burnin=burnin,
                thin=thin,
                smooth_lambda=smooth_lambda,
                prev_sample=prev_mean,
            )

            # 强制规则一致性过滤（硬约束）
            filtered = [s for s in samples if _规则一致性(week_ctx, s, context.voting_method, context.has_judges_save)]
            summary = 汇总后验(filtered if filtered else samples)

            for contestant, (mean_val, hdi_low, hdi_high) in summary.items():
                posterior_records.append(
                    {
                        "season": season,
                        "week": week,
                        "contestant": contestant,
                        "fan_mean": mean_val,
                        "fan_hdi_low": hdi_low,
                        "fan_hdi_high": hdi_high,
                    }
                )

            prev_mean = {k: v[0] for k, v in summary.items()} if summary else None

    intervals_df = pd.DataFrame(interval_records)
    posterior_df = pd.DataFrame(posterior_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals_df.to_csv(OUTPUT_DIR / "fan_vote_intervals.csv", index=False)
    posterior_df.to_csv(OUTPUT_DIR / "fan_vote_posterior_summary.csv", index=False)

    return posterior_df


def generate_rule_transparency_figure(manager: ActiveSetManager):
    """规则透明度审计图"""
    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()

    records = []
    for season in manager.get_all_seasons():
        context = manager.get_season_context(season)
        engine = lp_engine if context.voting_method == "percent" else cp_engine
        result = engine.solve(context)

        week_totals: Dict[int, float] = {}
        for (week, key_a, key_b), value in result.slack_values.items():
            if key_a == "__total__" and key_b == "__total__":
                week_totals[week] = max(week_totals.get(week, 0.0), float(value))

        season_score = max(week_totals.values()) if week_totals else 0.0
        records.append((season, season_score, context.voting_method))

    records.sort(key=lambda x: x[0])
    seasons = [r[0] for r in records]
    scores = [r[1] for r in records]
    methods = [r[2] for r in records]

    colors = [PALETTE["cyan_blue"] if m == "percent" else PALETTE["navy"] for m in methods]

    plt.figure(figsize=(12, 4.5))
    plt.bar(seasons, scores, color=colors, edgecolor=PALETTE["navy"], linewidth=0.4)

    for season, score in zip(seasons, scores):
        if score > 0:
            plt.plot(season, score, "o", color=PALETTE["dark_orange"], markersize=5)
            plt.text(season, score + 0.02, f"{score:.2f}", ha="center", fontsize=8)

    plt.xlabel("Season")
    plt.ylabel("Mismatch Score (S*)")
    plt.title("Rule Transparency Audit")
    plt.grid(True, axis="y", alpha=0.2, linewidth=0.6)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig_anomaly_detection.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"OK Saved: {out_path}")


def generate_bobby_bones_figure(posterior_df: Optional[pd.DataFrame] = None):
    """Bobby Bones 后验带状图"""
    if posterior_df is None or posterior_df.empty:
        return

    df = posterior_df[posterior_df["contestant"].str.lower() == "bobby bones"].copy()
    if df.empty:
        return

    df = df.sort_values("week")
    weeks = df["week"].tolist()
    means = df["fan_mean"].tolist()
    lows = df["fan_hdi_low"].tolist()
    highs = df["fan_hdi_high"].tolist()

    plt.figure(figsize=(9, 4.5))
    plt.fill_between(weeks, lows, highs, color=PALETTE["light_blue"], alpha=0.6, label="95% HDI")
    plt.plot(weeks, means, color=PALETTE["cyan_blue"], linewidth=2, label="Posterior Mean")
    plt.xlabel("Week")
    plt.ylabel("Fan Vote Share")
    plt.title("Bobby Bones (S27) Posterior Fan Vote")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.2, linewidth=0.6)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "bobby_bones_survival.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OK Saved: {out_path}")


def generate_dynamic_weight_figure(manager: ActiveSetManager):
    """动态权重曲线示意"""
    total_weeks = max((manager.get_season_context(s).num_weeks for s in manager.get_all_seasons()), default=1)
    weeks = list(range(1, total_weeks + 1))
    alphas = [_动态权重(w, total_weeks) for w in weeks]

    plt.figure(figsize=(7.5, 4.2))
    plt.plot(weeks, alphas, color=PALETTE["deep_cyan"], linewidth=2)
    plt.scatter(weeks, alphas, color=PALETTE["cyan_blue"], s=18)
    plt.xlabel("Week")
    plt.ylabel("Judge Weight (alpha)")
    plt.title("Dynamic Adaptive Weighting Schedule")
    plt.ylim(0.3, 0.8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    out_path = FIGURES_DIR / "fig_dynamic_weight_schedule.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"OK Saved: {out_path}")


def sync_paper_assets():
    """同步图表与表格到论文目录，并清理旧图"""
    ai_keep = {
        "fig_dual_engine.jpg",
        "fig_dwts_flowchart.jpeg",
        "fig_forest_plot.jpg",
        "fig_pareto_frontier.jpg",
        "figure1.jpg",
    }
    generated = {
        "fig_anomaly_detection.pdf",
        "bobby_bones_survival.png",
        "fig_pareto_frontier.pdf",
        "fig_shap_summary.pdf",
        "fig_shap_age_dependence.pdf",
        "fig_dynamic_weight_schedule.pdf",
    }

    paper_targets = [
        PROJECT_ROOT / "paper" / "en" / "PaperC",
        PROJECT_ROOT / "paper" / "zh" / "PaperC - Chinese",
    ]

    for paper_dir in paper_targets:
        fig_dir = paper_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for file in fig_dir.iterdir():
            if file.name in ai_keep:
                continue
            if file.name in generated:
                continue
            if file.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".eps"}:
                file.unlink()

        for name in generated:
            src = FIGURES_DIR / name
            if src.exists():
                shutil.copy2(src, fig_dir / name)

        # 同步汇总表
        for summary_name in [
            "fan_vote_posterior_summary.csv",
            "counterfactual_summary.csv",
            "counterfactual_summary_dynamic.csv",
            "pareto_frontier.csv",
            "feature_importance.csv",
        ]:
            src = OUTPUT_DIR / summary_name
            if src.exists():
                shutil.copy2(src, paper_dir / "sections" / summary_name)


def compile_paper(paper_dir: Path, main_file: str):
    cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-file-line-error", main_file]
    try:
        _run_cmd(cmd, paper_dir)
    except Exception as exc:
        print(f"[WARN] latexmk 失败：{exc}")
        _run_cmd(["xelatex", "-interaction=nonstopmode", main_file], paper_dir)


def main():
    parser = argparse.ArgumentParser(description="Run full DWTS framework pipeline")
    parser.add_argument("--mcmc-samples", type=int, default=2000, help="MCMC 样本数")
    parser.add_argument("--burnin", type=int, default=500, help="MCMC burn-in")
    parser.add_argument("--thin", type=int, default=5, help="MCMC 抽稀")
    parser.add_argument("--smooth-lambda", type=float, default=10.0, help="时间平滑强度")
    parser.add_argument("--skip-ml", action="store_true", help="跳过特征分析")
    parser.add_argument("--skip-compile", action="store_true", help="跳过 PDF 编译")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = DWTSDataLoader(str(DATA_PATH))
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()

    posterior_df = run_inversion_and_posterior(
        manager,
        mcmc_samples=args.mcmc_samples,
        burnin=args.burnin,
        thin=args.thin,
        smooth_lambda=args.smooth_lambda,
    )

    generate_rule_transparency_figure(manager)
    generate_bobby_bones_figure(posterior_df)
    generate_dynamic_weight_figure(manager)

    # 反事实评估（固定权重）
    counter = 运行反事实评估(manager, posterior_df, alpha=0.6)
    counter.逐周结果.to_csv(OUTPUT_DIR / "counterfactual_weekly.csv", index=False)
    counter.汇总结果.to_csv(OUTPUT_DIR / "counterfactual_summary.csv", index=False)

    # 反事实评估（动态权重）
    counter_dynamic = 运行反事实评估(manager, posterior_df, alpha=0.6, dynamic_alpha_fn=_动态权重)
    counter_dynamic.逐周结果.to_csv(OUTPUT_DIR / "counterfactual_weekly_dynamic.csv", index=False)
    counter_dynamic.汇总结果.to_csv(OUTPUT_DIR / "counterfactual_summary_dynamic.csv", index=False)

    # 机制设计（帕累托）
    alphas = [round(a, 2) for a in np.linspace(0.3, 0.8, 11)]
    pareto = 运行帕累托优化(
        manager,
        posterior_df,
        alphas=alphas,
        fig_path=str(FIGURES_DIR / "fig_pareto_frontier.pdf"),
    )
    pareto.结果表.to_csv(OUTPUT_DIR / "pareto_frontier.csv", index=False)

    # 特征分析
    if not args.skip_ml:
        运行特征分析(loader, output_dir=str(OUTPUT_DIR), fig_dir=str(FIGURES_DIR))

    sync_paper_assets()

    if not args.skip_compile:
        compile_paper(PROJECT_ROOT / "paper" / "en" / "PaperC", "main.tex")
        compile_paper(PROJECT_ROOT / "paper" / "zh" / "PaperC - Chinese", "main.tex")

    print("\n流程完成。")


if __name__ == "__main__":
    main()
