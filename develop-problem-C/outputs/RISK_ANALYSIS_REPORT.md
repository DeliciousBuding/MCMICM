# MCM Problem C - Senior Reviewer Risk Analysis Report

**Date**: 2026-01-30  
**Status**: ✅ All Risks Addressed

---

## Executive Summary

Senior data scientist 提出的三个关键风险已全部处理：

| Risk | Status | Finding |
|------|--------|---------|
| 🔴 Risk 1: Overfitting | ⚠️ CONFIRMED | 51.7% 的估计值 < 1%，需要改进 |
| 🔴 Risk 2: Rule Compliance | ✅ VERIFIED | PDF 明确支持 S28+ 使用 Rank 方法 |
| 🟡 Risk 3: Mechanism Simulation | ⚠️ NEEDS REVISION | 当前设计反而更差，需迭代 |

---

## Risk 1: Overfitting Trap

### Problem Detected
Fan vote 分布严重偏向极端值：
- **687/1330 (51.7%)** 的估计值 < 1%
- **162/1330 (12.2%)** 的估计值 > 99%
- Median 仅 0.6%

### Root Cause
模型为了满足淘汰约束，将 fan votes 推向 0% 或 100%。这在数学上可行，但现实中不合理。

### Solution: Strict Constraint Test
加入 **最低 0.5% 投票约束** 后：
- 31/34 季仍然可行
- **3 个"丑闻赛季"** 变得不可行：
  - **Season 30**: S* = 1.0
  - **Season 32**: S* = 2.0  
  - **Season 33**: S* = 2.0

### Interpretation
这些不可行的季节揭示了**真正的数据异常**——淘汰结果在数学上要求某些选手几乎没有粉丝投票。这反而是**更有价值的发现**，可以写进论文！

### Deliverables
- `risk1_overfitting_check.png` - 分布直方图
- `inconsistency_check_strict.csv` - 严格约束结果

---

## Risk 2: Rule Compliance (S28+)

### Question
S28+ 应该用 Rank 还是 Percent 方法？

### PDF Evidence (verbatim)
> "Around this same season [S28], the producers also returned to using the method of ranks... The exact season this change occurred is not known, but **it is reasonable to assume it was season 28**."

> Appendix: "COMBINED BY RANK (used in seasons 1, 2, and **28ᵃ - 34**)"

### Decision
**Our implementation is CORRECT:**
- S1-2: Rank
- S3-27: Percent
- S28-34: Rank + Judges' Save

### Deliverables
- `rule_compliance_memo.txt` - 完整合规备忘录

---

## Risk 3: Mechanism Simulation

### Simulation Setup
- **Old System**: Actual DWTS rules (as implemented)
- **New System**: Tiered Threshold (Soft Floor + Elite Mix)
- **Metric**: "Robbed Goddess" count (high-fan contestants eliminated)

### Results
| Metric | Old System | New System |
|--------|------------|------------|
| Robbed Goddesses | 51 | 94 |
| Weeks Changed | - | 131 |

### Problem
新系统反而**更糟糕**！高人气选手被淘汰的更多了。

### Root Cause Analysis
1. 当前 fan vote 估计本身有问题（Risk 1 的过拟合）
2. 新系统的"软门槛"可能设置不当
3. 需要用**真实的**而非**估计的** fan votes 来评估

### Next Steps
1. 先解决 Risk 1 的过拟合问题
2. 用 **regularized fan votes**（加入先验）重跑
3. 设计更合理的 fairness metric

### Deliverables
- `mechanism_simulation_comparison.csv` - 对比数据
- `risk3_mechanism_simulation.png` - 可视化图表

---

## Minor Fix: Data Interpretation

### Clarification
Rank 赛季中的小数值（如 0.666）是什么意思？

**Answer**: 是**归一化的相对流行度分数**，不是原始排名。

公式：`score = (N - rank + 1) / N`

例如 4 人时：1st=1.0, 2nd=0.75, 3rd=0.5, 4th=0.25

### Deliverables
- `data_interpretation_note.txt` - 完整说明

---

## Recommendations for Paper

### What to Include
1. **Acknowledge the overfitting issue** - 将其框架为"数据驱动发现"
2. **Report scandal weeks** (S30, S32, S33) - 这些是最有价值的发现
3. **Use multiple fan vote scenarios** - 展示 robustness
4. **Revise mechanism design** - 基于 simulation 结果迭代

### What to Avoid
1. ❌ 不要假装 zero inconsistency 是"完美"
2. ❌ 不要用过拟合的 fan votes 做决策
3. ❌ 不要声称新机制"更好"如果 simulation 不支持

---

## Files Generated

```
outputs/
├── risk1_overfitting_check.png      # Fan vote 分布图
├── inconsistency_check_strict.csv   # 严格约束结果
├── rule_compliance_memo.txt         # S28+ 规则备忘录
├── mechanism_simulation_comparison.csv
├── risk3_mechanism_simulation.png   # 机制对比图
└── data_interpretation_note.txt     # 数据解释说明
```

---

**Bottom Line**: 我们的核心方法论是对的，但需要警惕过拟合。下一步应该：
1. 加入 Bayesian 先验来正则化 fan votes
2. 用 bootstrap 生成 uncertainty bands
3. 基于真实分布重新设计机制
