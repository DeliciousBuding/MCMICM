"""
活跃集管理器

维护所有赛季/周的活跃选手集合，用于约束生成与反演。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .fsm import ContestantFSM, WeekType
from .data_loader import DWTSDataLoader


@dataclass
class WeekContext:
    """单周上下文（约束生成所需全部信息）"""

    season: int
    week: int
    week_type: WeekType
    active_set: Set[str]
    eliminated: List[str]
    survivors: Set[str]
    judge_scores: Dict[str, float]
    judge_percentages: Dict[str, float]
    judge_ranks: Dict[str, int]

    def get_pairwise_constraints(self) -> List[Tuple[str, str]]:
        """返回 (淘汰者, 幸存者) 对"""
        pairs = []
        for e in self.eliminated:
            for s in self.survivors:
                pairs.append((e, s))
        return pairs

    def has_valid_elimination(self) -> bool:
        """是否存在有效淘汰约束"""
        return len(self.eliminated) > 0 and len(self.survivors) > 0


@dataclass
class SeasonContext:
    """赛季上下文"""

    season: int
    voting_method: str
    has_judges_save: bool
    num_weeks: int
    num_contestants: int
    weeks: Dict[int, WeekContext] = field(default_factory=dict)
    fsm: Optional[ContestantFSM] = None

    def get_valid_weeks(self) -> List[int]:
        """返回有有效淘汰约束的周"""
        return [w for w, ctx in self.weeks.items() if ctx.has_valid_elimination()]

    def get_all_constraints(self) -> List[Tuple[int, str, str]]:
        """返回所有约束 (week, eliminated, survivor)"""
        constraints = []
        for week, ctx in self.weeks.items():
            for e, s in ctx.get_pairwise_constraints():
                constraints.append((week, e, s))
        return constraints


class ActiveSetManager:
    """
    活跃集管理器（ETL 与反演引擎的桥梁）
    """

    def __init__(self, loader: DWTSDataLoader):
        self.loader = loader
        self.season_contexts: Dict[int, SeasonContext] = {}
        self.fsm_cache: Dict[int, ContestantFSM] = {}

        from ..config import SEASON_CONFIG

        self.config = SEASON_CONFIG

    def build_season_context(self, season: int) -> SeasonContext:
        """构建赛季上下文"""
        if season in self.season_contexts:
            return self.season_contexts[season]

        season_data = self.loader.get_season_data(season)
        score_matrix = self.loader.score_matrix[
            self.loader.score_matrix["season"] == season
        ]

        fsm = ContestantFSM(season_data, score_matrix)
        self.fsm_cache[season] = fsm

        context = SeasonContext(
            season=season,
            voting_method=self.config.get_voting_method(season),
            has_judges_save=self.config.has_judges_save(season),
            num_weeks=season_data["num_weeks"],
            num_contestants=season_data["num_contestants"],
            fsm=fsm,
        )

        for week in range(1, season_data["num_weeks"] + 1):
            week_ctx = self._build_week_context(season, week, fsm, score_matrix)
            if week_ctx:
                context.weeks[week] = week_ctx

        self.season_contexts[season] = context
        return context

    def _build_week_context(
        self,
        season: int,
        week: int,
        fsm: ContestantFSM,
        score_matrix: pd.DataFrame,
    ) -> Optional[WeekContext]:
        """构建单周上下文"""
        week_type = fsm.get_week_type(week)
        if week_type == WeekType.SKIPPED:
            return None

        active_set = fsm.get_active_set(week)
        eliminated = fsm.get_eliminated_this_week(week)
        survivors = fsm.get_survivors(week)

        week_scores = score_matrix[score_matrix["week"] == week]
        judge_scores = {}
        for contestant in active_set:
            contestant_score = week_scores[week_scores["contestant"] == contestant]
            if len(contestant_score) > 0:
                total = contestant_score.iloc[0]["total_score"]
                judge_scores[contestant] = total if pd.notna(total) else 0.0
            else:
                judge_scores[contestant] = 0.0

        total_sum = sum(judge_scores.values())
        judge_percentages = {}
        if total_sum > 0:
            for c, s in judge_scores.items():
                judge_percentages[c] = s / total_sum
        else:
            n = len(active_set)
            for c in active_set:
                judge_percentages[c] = 1.0 / n if n > 0 else 0.0

        sorted_contestants = sorted(
            judge_scores.items(), key=lambda x: x[1], reverse=True
        )
        judge_ranks = {}
        for rank, (contestant, _) in enumerate(sorted_contestants, 1):
            judge_ranks[contestant] = rank

        return WeekContext(
            season=season,
            week=week,
            week_type=week_type,
            active_set=active_set,
            eliminated=eliminated,
            survivors=survivors,
            judge_scores=judge_scores,
            judge_percentages=judge_percentages,
            judge_ranks=judge_ranks,
        )

    def get_season_context(self, season: int) -> SeasonContext:
        """获取或构建赛季上下文"""
        if season not in self.season_contexts:
            return self.build_season_context(season)
        return self.season_contexts[season]

    def get_all_seasons(self) -> List[int]:
        """返回所有赛季列表"""
        return sorted(self.loader.processed_df["season"].unique())

    def build_all_contexts(self) -> Dict[int, SeasonContext]:
        """构建全部赛季上下文"""
        for season in self.get_all_seasons():
            self.build_season_context(season)
        return self.season_contexts

    def get_constraint_matrix(self, season: int) -> pd.DataFrame:
        """
        生成约束矩阵（用于分析/导出）
        """
        context = self.get_season_context(season)

        records = []
        for week, ctx in context.weeks.items():
            for e, s in ctx.get_pairwise_constraints():
                e_score = ctx.judge_scores.get(e, 0)
                s_score = ctx.judge_scores.get(s, 0)
                e_pct = ctx.judge_percentages.get(e, 0)
                s_pct = ctx.judge_percentages.get(s, 0)
                e_rank = ctx.judge_ranks.get(e, 0)
                s_rank = ctx.judge_ranks.get(s, 0)

                judge_score_diff = s_score - e_score
                min_fan_diff = e_pct - s_pct

                records.append(
                    {
                        "week": week,
                        "eliminated": e,
                        "survivor": s,
                        "elim_judge_score": e_score,
                        "surv_judge_score": s_score,
                        "elim_judge_pct": e_pct,
                        "surv_judge_pct": s_pct,
                        "elim_judge_rank": e_rank,
                        "surv_judge_rank": s_rank,
                        "judge_score_diff": judge_score_diff,
                        "min_fan_diff_needed": min_fan_diff,
                    }
                )

        return pd.DataFrame(records)

    def get_active_matrix(self, season: int) -> pd.DataFrame:
        """导出活跃矩阵（周 × 选手）"""
        context = self.get_season_context(season)
        fsm = context.fsm
        return fsm.to_dataframe()

    def generate_summary_report(self) -> pd.DataFrame:
        """生成所有赛季统计摘要"""
        records = []
        for season in self.get_all_seasons():
            ctx = self.get_season_context(season)
            total_constraints = len(ctx.get_all_constraints())
            valid_weeks = len(ctx.get_valid_weeks())
            no_elim_weeks = sum(
                1 for w in ctx.weeks.values() if w.week_type == WeekType.NO_ELIM
            )
            multi_elim_weeks = sum(
                1
                for w in ctx.weeks.values()
                if w.week_type in [WeekType.MULTI_ELIM, WeekType.DOUBLE_ELIM]
            )

            records.append(
                {
                    "season": season,
                    "voting_method": ctx.voting_method,
                    "has_judges_save": ctx.has_judges_save,
                    "num_contestants": ctx.num_contestants,
                    "num_weeks": ctx.num_weeks,
                    "valid_weeks": valid_weeks,
                    "no_elim_weeks": no_elim_weeks,
                    "multi_elim_weeks": multi_elim_weeks,
                    "total_constraints": total_constraints,
                }
            )

        return pd.DataFrame(records)

    def export_all(self, output_dir: str):
        """导出所有结构化数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary = self.generate_summary_report()
        summary.to_csv(output_path / "season_summary.csv", index=False)

        for season in self.get_all_seasons():
            constraints = self.get_constraint_matrix(season)
            constraints.to_csv(output_path / f"constraints_s{season}.csv", index=False)

            active = self.get_active_matrix(season)
            active.to_csv(output_path / f"active_matrix_s{season}.csv", index=False)
