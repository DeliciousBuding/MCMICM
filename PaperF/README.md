# PaperF - ICM Problem F (Policy Science) 论文骨架

## 项目概述

本目录包含 **ICM 2026 Problem F (政策科学题)** 的 O奖级论文模板。

**题型特点**：F题聚焦于复杂政策问题的定量分析与建议，需要：
- 📋 **政策备忘录 (Memo)**：面向决策者的单页摘要
- 🔄 **机制建模**：揭示政策干预如何传导至结果
- ⚖️ **多目标权衡**：效率、公平、成本的 Pareto 分析
- 🌐 **溢出效应**：跨领域影响与长期系统稳定性
- 📊 **鲁棒性验证**：灵敏度分析确保建议可靠

---

## 文件结构

```
PaperF/
├── main.tex                    # 主控文件
├── ref.bib                     # 参考文献数据库
├── README.md                   # 本说明文档
├── .gitignore                  # Git忽略规则
│
├── sections/
│   ├── _shared_macros.tex      # 共享宏定义（协作必备）
│   ├── 00_abstract.tex         # 摘要 (Summary Sheet)
│   ├── 00_memo.tex             # 政策备忘录 ★F题核心★
│   ├── 01_introduction.tex     # 引言
│   ├── 02_project_and_assumptions.tex    # 项目定义与假设
│   ├── 03_data_and_notation.tex          # 数据与符号
│   ├── 04_model1_mechanism.tex           # 模型一：系统机制
│   ├── 05_model2_policy.tex              # 模型二：政策优化
│   ├── 06_solution_and_algorithm.tex     # 求解算法
│   ├── 07_results_and_validation.tex     # 结果与验证
│   ├── 08_sensitivity_and_robustness.tex # 灵敏度分析
│   ├── 09_policy_impacts_and_discussion.tex  # 政策影响（溢出效应）
│   ├── 10_strengths_and_weaknesses.tex   # 优缺点分析
│   └── 11_conclusion.tex       # 结论
│
├── appendices/
│   ├── appendix_code.tex       # 代码附录
│   ├── appendix_figures.tex    # 补充图表
│   └── 12_ai_tool_report.tex   # AI使用报告
│
├── figures/                    # 图片目录（PDF/PNG）
│
└── ai_log/
    └── README.md               # AI使用日志说明
```

---

## 编译方式

### 推荐：使用 latexmk 自动编译

```bash
cd PaperF
latexmk -xelatex -bibtex main.tex
```

### 手动编译

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

### 清理临时文件

```bash
latexmk -c
```

---

## 协作规范

### 1. 分工建议

| 角色 | 负责文件 | 说明 |
|------|---------|------|
| **建模手** | `04_model1_mechanism.tex`, `05_model2_policy.tex` | 核心模型构建 |
| **算法手** | `06_solution_and_algorithm.tex`, `appendix_code.tex` | 算法实现 |
| **分析手** | `07_results`, `08_sensitivity`, `09_impacts` | 结果分析 |
| **写作手** | `00_memo.tex`, `01_intro`, `11_conclusion` | 政策叙事 |

### 2. 宏定义使用

所有共享符号定义在 `_shared_macros.tex`，例如：

```latex
\policy     % 政策变量 π
\outcome    % 结果变量 Y
\objective  % 目标函数 J
\welfare    % 社会福利 W
\TODO{...}  % 待填标记（红色加粗）
```

### 3. TODO 标记

所有待完成内容使用 `\TODO{说明}` 标记，便于：
- 编译后可视化定位
- 比赛前全文搜索确认无遗漏

---

## F题写作要点

### 政策备忘录 (Memo) - 最关键！

- **格式**：To / From / Date / Subject
- **内容**：执行摘要、核心建议（3-5条）、预期成效、风险、资源需求、下一步
- **语言**：专业但可被非技术决策者理解
- **篇幅**：严格1页以内

### 模型叙事结构

```
政策背景 → 机制建模 → 政策优化 → 情景比较 → 灵敏度测试 → 溢出效应 → 结论建议
```

### 评审关注点

1. **问题重述**：是否准确理解政策问题？
2. **机制透明**：能否解释"为什么"政策有效？
3. **权衡分析**：是否展示多目标冲突与平衡？
4. **鲁棒性**：建议在不确定性下是否可靠？
5. **可操作性**：建议是否具体、分阶段、可评估？

---

## AI 使用追踪

按 MCM/ICM 2025+ 要求，所有 AI 工具使用需记录在：
- `ai_log/` 目录：原始交互记录
- `appendices/12_ai_tool_report.tex`：正式披露报告

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-01 | 初始 O奖级骨架 |

---

## 联系与支持

- **模板作者**：Team #2617892
- **LaTeX 文档类**：mcmthesis
- **问题反馈**：\TODO{GitHub Issues 或邮箱}

---

*Good luck with ICM 2026! 🏆*
