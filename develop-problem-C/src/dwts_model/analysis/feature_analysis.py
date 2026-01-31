"""
特征影响分析：XGBoost + SHAP。
若依赖缺失则回退到传统树模型与置换重要度。
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..paper_palette import PALETTE


@dataclass
class 特征分析结果:
    结果表: pd.DataFrame
    图像目录: str


def _构建特征表(loader) -> Tuple[pd.DataFrame, pd.Series]:
    df = loader.processed_df.copy()
    scores = loader.score_matrix.copy()

    # 计算每个选手的评委分统计
    score_stats = (
        scores[scores["all_na"] == False]
        .groupby(["season", "contestant"])["total_score"]
        .agg(["mean", "std", "max", "min", "count"])
        .reset_index()
    )
    score_stats.columns = ["season", "contestant", "judge_mean", "judge_std", "judge_max", "judge_min", "judge_count"]

    df = df.merge(score_stats, left_on=["season", "celebrity_name"], right_on=["season", "contestant"], how="left")

    # 目标：存活周数（淘汰周 or 该赛季最大周）
    season_max_week = scores.groupby("season")["week"].max().to_dict()
    df["survival_weeks"] = df.apply(
        lambda r: r["elimination_week"] if pd.notna(r["elimination_week"]) else season_max_week.get(r["season"], np.nan),
        axis=1,
    )

    # 处理舞伴高基数：仅保留频率最高的 10 位
    top_partners = df["ballroom_partner"].value_counts().head(10).index
    df["partner_bucket"] = df["ballroom_partner"].where(df["ballroom_partner"].isin(top_partners), "Other")

    # 构建特征
    features = df[[
        "celebrity_age_during_season",
        "celebrity_industry",
        "partner_bucket",
        "judge_mean",
        "judge_std",
        "judge_max",
        "judge_min",
        "judge_count",
    ]].copy()

    features = pd.get_dummies(features, columns=["celebrity_industry", "partner_bucket"], drop_first=True)
    features["celebrity_age_during_season"] = features["celebrity_age_during_season"].fillna(
        features["celebrity_age_during_season"].median()
    )

    target = df["survival_weeks"].fillna(df["survival_weeks"].median())

    return features, target


def _生成配色映射():
    return LinearSegmentedColormap.from_list(
        "dwts_cmap",
        [PALETTE["baseline"], PALETTE["proposed"], PALETTE["accent"]],
        N=256,
    )


def _绘制重要性条形图(result_df: pd.DataFrame, fig_path: str):
    top = result_df.sort_values("importance", ascending=False).head(12)
    plt.figure(figsize=(7.2, 4.8))
    plt.barh(top["feature"], top["importance"], color=PALETTE["proposed"], edgecolor=PALETTE["baseline"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance (Fallback)")
    plt.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def _绘制年龄依赖(age: np.ndarray, shap_age: np.ndarray, fig_path: str):
    cmap = _生成配色映射()
    mask = np.isfinite(age) & np.isfinite(shap_age)
    age = age[mask]
    shap_age = shap_age[mask]

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(age, shap_age, c=age, cmap=cmap, s=18, alpha=0.7, edgecolors="none")

    # 分箱均值曲线
    bins = np.linspace(age.min(), age.max(), 12)
    centers = (bins[:-1] + bins[1:]) / 2
    digit = np.digitize(age, bins) - 1
    means = []
    for i in range(len(centers)):
        sel = shap_age[digit == i]
        means.append(np.mean(sel) if sel.size else np.nan)
    plt.plot(centers, means, color=PALETTE["aux"], linewidth=2)

    plt.xlabel("Celebrity Age")
    plt.ylabel("SHAP Value")
    plt.title("SHAP Dependence: Age")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def 运行特征分析(loader, output_dir: str, fig_dir: str) -> 特征分析结果:
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y = _构建特征表(loader)

    result_df = pd.DataFrame({"feature": X.columns})

    # 训练模型
    model = None
    use_shap = False
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X, y)
        use_shap = True
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)

    # 输出特征重要性
    if hasattr(model, "feature_importances_"):
        result_df["importance"] = model.feature_importances_
    else:
        result_df["importance"] = np.nan

    result_df.sort_values("importance", ascending=False).to_csv(
        os.path.join(output_dir, "feature_importance.csv"), index=False
    )

    # 生成 SHAP 图
    summary_path = os.path.join(fig_dir, "fig_shap_summary.pdf")
    age_path = os.path.join(fig_dir, "fig_shap_age_dependence.pdf")

    if use_shap:
        try:
            import shap
            cmap = _生成配色映射()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap.summary_plot(shap_values, X, show=False, color=cmap, plot_size=(7, 4.8))
            plt.tight_layout()
            plt.savefig(summary_path, bbox_inches="tight")
            plt.close()

            if "celebrity_age_during_season" in X.columns:
                age_idx = list(X.columns).index("celebrity_age_during_season")
                _绘制年龄依赖(X["celebrity_age_during_season"].to_numpy(), shap_values[:, age_idx], age_path)
        except Exception:
            _绘制重要性条形图(result_df, summary_path)
    else:
        _绘制重要性条形图(result_df, summary_path)

    return 特征分析结果(结果表=result_df, 图像目录=fig_dir)
