"""
选手生命周期 FSM

用于精确维护每周活跃集合，防止分母错算。
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class ContestantState(Enum):
    """选手状态枚举"""

    ACTIVE = auto()
    ELIMINATED_THIS_WEEK = auto()
    WITHDREW = auto()
    FINALIST = auto()
    INACTIVE = auto()


class WeekType(Enum):
    """周类型枚举"""

    NORMAL = auto()
    NO_ELIM = auto()
    MULTI_ELIM = auto()
    DOUBLE_ELIM = auto()
    SKIPPED = auto()


@dataclass
class ContestantLifecycle:
    """单个选手生命周期"""

    name: str
    season: int
    final_status: str
    final_placement: int
    elimination_week: Optional[int] = None
    withdrew_week: Optional[int] = None
    state_history: Dict[int, ContestantState] = field(default_factory=dict)

    def get_state(self, week: int) -> ContestantState:
        return self.state_history.get(week, ContestantState.INACTIVE)

    def was_active(self, week: int) -> bool:
        """判断该周是否在分母中"""
        state = self.get_state(week)
        return state in [
            ContestantState.ACTIVE,
            ContestantState.ELIMINATED_THIS_WEEK,
            ContestantState.FINALIST,
        ]

    def get_active_weeks(self) -> List[int]:
        """返回所有活跃周"""
        return [
            w
            for w, s in self.state_history.items()
            if s
            in [
                ContestantState.ACTIVE,
                ContestantState.ELIMINATED_THIS_WEEK,
                ContestantState.FINALIST,
            ]
        ]


class ContestantFSM:
    """
    选手生命周期状态机

    逻辑：
    1) 解析 results 字段确定最终状态
    2) 退赛：根据最后一次正分推断
    3) 逐周构建状态序列
    4) 处理边界与异常
    """

    def __init__(self, season_data: Dict, score_matrix: pd.DataFrame):
        self.contestants_df = season_data["contestants"]
        self.scores = score_matrix
        self.num_weeks = season_data["num_weeks"]
        self.season = self.contestants_df["season"].iloc[0]

        self.lifecycles: Dict[str, ContestantLifecycle] = {}
        self.week_types: Dict[int, WeekType] = {}
        self.week_events: Dict[int, List[str]] = {}

        self._build_lifecycles()
        self._classify_weeks()

    def _build_lifecycles(self):
        """构建每位选手的生命周期"""
        for _, row in self.contestants_df.iterrows():
            name = row["celebrity_name"]
            status = row["status"]
            elim_week = row["elimination_week"]
            placement = row["placement"]

            lifecycle = ContestantLifecycle(
                name=name,
                season=self.season,
                final_status=status,
                final_placement=placement,
                elimination_week=elim_week,
            )

            if status == "withdrew":
                lifecycle.withdrew_week = self._infer_withdrew_week(name)

            self._build_state_history(lifecycle)
            self.lifecycles[name] = lifecycle

    def _infer_withdrew_week(self, contestant: str) -> int:
        """根据最后一次正分推断退赛周"""
        contestant_scores = self.scores[self.scores["contestant"] == contestant]
        last_positive_week = 0
        for week in range(1, self.num_weeks + 1):
            week_data = contestant_scores[contestant_scores["week"] == week]
            if len(week_data) > 0:
                total = week_data.iloc[0]["total_score"]
                if pd.notna(total) and total > 0:
                    last_positive_week = week
        return last_positive_week + 1 if last_positive_week > 0 else 1

    def _build_state_history(self, lifecycle: ContestantLifecycle):
        """逐周生成状态历史"""
        name = lifecycle.name
        status = lifecycle.final_status
        elim_week = lifecycle.elimination_week
        withdrew_week = lifecycle.withdrew_week

        for week in range(1, self.num_weeks + 1):
            if status == "withdrew":
                state = (
                    ContestantState.INACTIVE
                    if withdrew_week and week >= withdrew_week
                    else ContestantState.ACTIVE
                )
            elif status == "eliminated":
                if week < elim_week:
                    state = ContestantState.ACTIVE
                elif week == elim_week:
                    state = ContestantState.ELIMINATED_THIS_WEEK
                else:
                    state = ContestantState.INACTIVE
            elif status in ["winner", "finalist"]:
                state = ContestantState.FINALIST
            else:
                contestant_scores = self.scores[
                    (self.scores["contestant"] == name)
                    & (self.scores["week"] == week)
                ]
                if len(contestant_scores) > 0:
                    total = contestant_scores.iloc[0]["total_score"]
                    state = ContestantState.ACTIVE if pd.notna(total) and total > 0 else ContestantState.INACTIVE
                else:
                    state = ContestantState.INACTIVE

            lifecycle.state_history[week] = state

    def _classify_weeks(self):
        """分类每周类型"""
        for week in range(1, self.num_weeks + 1):
            eliminated = []
            for name, lifecycle in self.lifecycles.items():
                if lifecycle.get_state(week) == ContestantState.ELIMINATED_THIS_WEEK:
                    eliminated.append(name)
            self.week_events[week] = eliminated

            week_scores = self.scores[self.scores["week"] == week]
            if week_scores["all_na"].all():
                self.week_types[week] = WeekType.SKIPPED
            elif len(eliminated) == 0:
                self.week_types[week] = WeekType.NO_ELIM
            elif len(eliminated) == 1:
                self.week_types[week] = WeekType.NORMAL
            elif len(eliminated) == 2:
                self.week_types[week] = WeekType.DOUBLE_ELIM
            else:
                self.week_types[week] = WeekType.MULTI_ELIM

    def get_active_set(self, week: int) -> Set[str]:
        """获取该周活跃选手集合"""
        active = set()
        for name, lifecycle in self.lifecycles.items():
            if lifecycle.was_active(week):
                active.add(name)
        return active

    def get_eliminated_this_week(self, week: int) -> List[str]:
        return self.week_events.get(week, [])

    def get_survivors(self, week: int) -> Set[str]:
        """获取该周幸存者集合"""
        active = self.get_active_set(week)
        eliminated = set(self.get_eliminated_this_week(week))
        return active - eliminated

    def get_week_type(self, week: int) -> WeekType:
        return self.week_types.get(week, WeekType.SKIPPED)

    def get_pairwise_constraints(self, week: int) -> List[Tuple[str, str]]:
        """生成 (淘汰者, 幸存者) 约束对"""
        eliminated = self.get_eliminated_this_week(week)
        survivors = self.get_survivors(week)
        return [(e, s) for e in eliminated for s in survivors]

    def generate_event_log(self) -> List[str]:
        """生成赛季事件日志"""
        log = []
        log.append(f"=== Season {self.season} Event Log ===")
        log.append(f"Total contestants: {len(self.lifecycles)}")
        log.append(f"Total weeks: {self.num_weeks}")
        log.append("")

        for week in range(1, self.num_weeks + 1):
            week_type = self.get_week_type(week)
            active = self.get_active_set(week)
            eliminated = self.get_eliminated_this_week(week)
            log.append(f"Week {week}: {week_type.name}")
            log.append(f"  Active: {len(active)} contestants")
            if eliminated:
                log.append(f"  Eliminated: {', '.join(eliminated)}")
            for name, lifecycle in self.lifecycles.items():
                if lifecycle.withdrew_week == week:
                    log.append(f"  WITHDREW: {name}")
            log.append("")
        return log

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame"""
        records = []
        for name, lifecycle in self.lifecycles.items():
            for week in range(1, self.num_weeks + 1):
                state = lifecycle.get_state(week)
                records.append(
                    {
                        "season": self.season,
                        "contestant": name,
                        "week": week,
                        "state": state.name,
                        "is_active": lifecycle.was_active(week),
                        "final_placement": lifecycle.final_placement,
                    }
                )
        return pd.DataFrame(records)
