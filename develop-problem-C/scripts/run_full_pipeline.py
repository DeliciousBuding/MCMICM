"""
一键执行模型一主流程：
1) 运行蒙特卡洛分析
2) 生成论文级图表
3) 生成规则透明度与案例图
4) 同步到中英文论文并编译 PDF
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "2026_MCM_Problem_C_Data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(SRC_DIR))

from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, RankCPEngine

# 统一配色（与可视化脚本一致）
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


def run_mc(samples: int, seasons: str, regularize: bool, tightening: float):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_mc_analysis.py"), "--samples", str(samples)]
    if seasons:
        cmd += ["--seasons", seasons]
    if regularize:
        cmd += ["--regularize", "--tightening-factor", str(tightening)]
    _run_cmd(cmd, PROJECT_ROOT)


def run_visuals():
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "visualize_mc_results.py")]
    _run_cmd(cmd, PROJECT_ROOT)


def generate_rule_transparency_figure():
    """生成规则透明度（不一致度）图"""
    loader = DWTSDataLoader(str(DATA_PATH))
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()

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


def generate_bobby_bones_figure():
    """生成 Bobby Bones 个案区间图"""
    loader = DWTSDataLoader(str(DATA_PATH))
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()

    season = 27
    context = manager.get_season_context(season)
    engine = PercentLPEngine()
    result = engine.solve(context)

    # 选手名称匹配
    target_name = "Bobby Bones"
    all_names = list(context.fsm.lifecycles.keys()) if context.fsm else []
    name_map = {c.lower(): c for c in all_names}
    if target_name.lower() not in name_map:
        raise ValueError("未找到 Bobby Bones，请检查数据名称")
    actual_name = name_map[target_name.lower()]

    weeks = []
    lowers = []
    uppers = []
    points = []

    for week in sorted(result.week_results.keys()):
        est = result.week_results.get(week, {}).get(actual_name)
        if not est:
            continue
        weeks.append(week)
        lowers.append(est.lower_bound)
        uppers.append(est.upper_bound)
        points.append(est.point_estimate)

    if not weeks:
        raise ValueError("Bobby Bones 无有效周次数据")

    plt.figure(figsize=(9, 4.5))
    plt.fill_between(weeks, lowers, uppers, color=PALETTE["light_blue"], alpha=0.6, label="Feasible Interval")
    plt.plot(weeks, points, color=PALETTE["cyan_blue"], linewidth=2, label="Point Estimate")
    plt.plot(weeks, lowers, color=PALETTE["deep_cyan"], linestyle="--", linewidth=1)
    plt.plot(weeks, uppers, color=PALETTE["deep_cyan"], linestyle="--", linewidth=1)

    plt.xlabel("Week")
    plt.ylabel("Fan Vote Share")
    plt.title("Bobby Bones (S27) Feasible Interval")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.2, linewidth=0.6)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "bobby_bones_survival.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
        "mc_probability_distribution.pdf",
        "mc_season_evolution.pdf",
        "mc_confidence_intervals.pdf",
        "mc_voting_method_comparison.pdf",
        "mc_classification_breakdown.pdf",
        "mc_interval_width_analysis.pdf",
        "fig_anomaly_detection.pdf",
        "bobby_bones_survival.png",
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
        summary_src = OUTPUT_DIR / "mc_summary_statistics.tex"
        if summary_src.exists():
            shutil.copy2(summary_src, paper_dir / "sections" / "mc_summary_statistics.tex")


def compile_paper(paper_dir: Path, main_file: str):
    """编译论文 PDF"""
    cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-file-line-error", main_file]
    try:
        _run_cmd(cmd, paper_dir)
    except Exception as exc:
        print(f"[WARN] latexmk 失败：{exc}")
        # 兜底：尝试直接运行 xelatex
        _run_cmd(["xelatex", "-interaction=nonstopmode", main_file], paper_dir)


def main():
    parser = argparse.ArgumentParser(description="Run full Model-1 pipeline")
    parser.add_argument("--samples", type=int, default=5000, help="MC 样本数")
    parser.add_argument("--seasons", type=str, default=None, help="赛季范围，例如 '1-10' 或 '32,33'")
    parser.add_argument("--regularize", action="store_true", help="排名制启用区间收缩")
    parser.add_argument("--tightening-factor", type=float, default=0.12, help="收缩比例")
    parser.add_argument("--skip-compile", action="store_true", help="跳过 PDF 编译")
    args = parser.parse_args()

    run_mc(samples=args.samples, seasons=args.seasons, regularize=args.regularize, tightening=args.tightening_factor)
    run_visuals()
    generate_rule_transparency_figure()
    generate_bobby_bones_figure()
    sync_paper_assets()

    if not args.skip_compile:
        compile_paper(PROJECT_ROOT / "paper" / "en" / "PaperC", "main.tex")
        compile_paper(PROJECT_ROOT / "paper" / "zh" / "PaperC - Chinese", "main.tex")

    print("\n流程完成。")


if __name__ == "__main__":
    main()
