"""
DWTS Visualization Script
生成论文所需的核心图表
使用统一论文级配色方案
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Add dwts_model to path
sys.path.insert(0, str(Path(__file__).parent))

# 导入统一调色板
from dwts_model.paper_palette import (
    PALETTE, VOTING_METHODS, MECHANISMS, DATA_STATES,
    get_season_color, get_season_colors, apply_paper_style,
    LEGEND_LABELS, LINE_STYLES, BAR_STYLES, FILL_STYLES
)

# 设置中文字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

OUTPUT_DIR = Path('outputs/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 确保数据文件存在
def ensure_data_files():
    """运行分析确保数据文件存在"""
    from dwts_model.etl import DWTSDataLoader, ActiveSetManager
    from dwts_model.engines import PercentLPEngine, RankCPEngine
    from dwts_model.sampling import CounterfactualSimulator
    
    loader = DWTSDataLoader('2026_MCM_Problem_C_Data.csv')
    loader.load()
    
    # 保存清洗数据 (跳过已存在的)
    try:
        loader.processed_df.to_csv('outputs/cleaned_contestants.csv', index=False)
    except:
        pass
    try:
        loader.score_matrix.to_csv('outputs/cleaned_scores.csv', index=False)
    except:
        pass
    
    # 构建分析
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()
    
    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()
    simulator = CounterfactualSimulator()
    
    # 运行反演和反事实分析
    inversion_results = {}
    counterfactual_results = {}
    inconsistency_data = []
    
    for season in manager.get_all_seasons():
        context = manager.get_season_context(season)
        
        if context.voting_method == 'percent':
            result = lp_engine.solve(context)
        else:
            result = cp_engine.solve(context)
        
        inversion_results[season] = result
        inconsistency_data.append({
            'season': season,
            'inconsistency_score': result.inconsistency_score,
            'voting_method': context.voting_method
        })
        
        # 反事实
        fan_votes = result.get_point_estimates_matrix()
        cf = simulator.compare_methods(context, fan_votes)
        counterfactual_results[season] = cf
    
    # 保存不一致性数据
    pd.DataFrame(inconsistency_data).to_csv('outputs/inconsistency_spectrum.csv', index=False)
    
    # 保存反事实数据
    cf_records = []
    for season, cf in counterfactual_results.items():
        cf_records.append({
            'season': season,
            'reversal_rate': cf.reversal_rate,
            'n_reversal_weeks': len(cf.reversal_weeks)
        })
    
    pd.DataFrame(cf_records).to_csv('outputs/counterfactual_results.csv', index=False)
    
    # 保存逆转热力图数据 - 从模拟结果中获取
    reversal_records = []
    for season, cf in counterfactual_results.items():
        context = manager.get_season_context(season)
        for week in context.weeks.keys():
            reversal_records.append({
                'season': season,
                'week': week,
                'would_reverse': week in cf.reversal_weeks
            })
    
    pd.DataFrame(reversal_records).to_csv('outputs/reversal_heatmap.csv', index=False)
    
    # 保存粉丝投票估计
    vote_records = []
    for season, result in inversion_results.items():
        for week, estimates in result.week_results.items():
            for contestant, est in estimates.items():
                vote_records.append({
                    'season': season,
                    'week': week,
                    'contestant': contestant,
                    'fan_vote_estimate': est.point_estimate,
                    'certainty': est.certainty
                })
    pd.DataFrame(vote_records).to_csv('outputs/fan_vote_estimates.csv', index=False)
    
    print("✓ Data files generated")
    return True


def plot_inconsistency_spectrum():
    """
    Figure 1: 各季节不一致性得分 (S*)
    显示模型与实际淘汰结果的契合度
    """
    df = pd.read_csv('outputs/inconsistency_spectrum.csv')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 按投票方法着色 - 使用统一调色板
    colors = get_season_colors(df['season'].tolist())
    
    bars = ax.bar(df['season'], df['inconsistency_score'], color=colors, 
                  edgecolor=PALETTE['aux'], linewidth=0.8)
    
    # 标记高不一致性季节
    for i, row in df.iterrows():
        if row['inconsistency_score'] > 0:
            ax.annotate(f"S*={row['inconsistency_score']:.1f}", 
                       (row['season'], row['inconsistency_score']),
                       textcoords="offset points", xytext=(0, 5),
                       ha='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Inconsistency Score (S*)', fontsize=12)
    ax.set_title('Model Fit: Inconsistency Score by Season', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 35))
    ax.set_xlim(0.5, 34.5)
    ax.axhline(y=0, color=PALETTE['aux'], linestyle='--', alpha=0.5)
    apply_paper_style(ax)
    
    # 图例 - 使用统一调色板
    rank_patch = mpatches.Patch(color=PALETTE['baseline'], label=LEGEND_LABELS['rank'])
    pct_patch = mpatches.Patch(color=PALETTE['proposed'], label=LEGEND_LABELS['percent'])
    ax.legend(handles=[rank_patch, pct_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_inconsistency_spectrum.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_inconsistency_spectrum.pdf', bbox_inches='tight')
    print("✓ Figure 1: Inconsistency Spectrum saved")
    plt.close()


def plot_reversal_heatmap():
    """
    Figure 2: 方法切换逆转热力图
    显示如果使用另一种投票方法，哪些周的淘汰结果会不同
    """
    df = pd.read_csv('outputs/reversal_heatmap.csv')
    
    # 创建热力图矩阵
    seasons = sorted(df['season'].unique())
    max_week = 11
    
    matrix = np.zeros((len(seasons), max_week))
    for _, row in df.iterrows():
        s_idx = seasons.index(row['season'])
        w_idx = int(row['week']) - 1
        if w_idx < max_week:
            matrix[s_idx, w_idx] = 1 if row['would_reverse'] else 0
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 使用统一调色板: 浅色=无逆转, 警示色=逆转
    cmap = plt.cm.colors.ListedColormap([PALETTE['fill'], PALETTE['warning']])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    ax.set_xticks(range(max_week))
    ax.set_xticklabels([f'W{i+1}' for i in range(max_week)])
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels([f'S{s}' for s in seasons])
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Season', fontsize=12)
    ax.set_title('Elimination Reversals Under Alternative Voting Method', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.set_xticks(np.arange(-0.5, max_week, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(seasons), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # 图例 - 使用统一调色板
    no_rev = mpatches.Patch(color=PALETTE['fill'], label='Same Outcome')
    rev = mpatches.Patch(color=PALETTE['warning'], label='Would Reverse')
    ax.legend(handles=[no_rev, rev], loc='upper right', bbox_to_anchor=(1.15, 1))
    apply_paper_style(ax, grid_alpha=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_reversal_heatmap.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_reversal_heatmap.pdf', bbox_inches='tight')
    print("✓ Figure 2: Reversal Heatmap saved")
    plt.close()


def plot_counterfactual_comparison():
    """
    Figure 3: 反事实分析 - 各季节的逆转率
    """
    df = pd.read_csv('outputs/counterfactual_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 按投票方法着色 - 使用统一调色板
    colors = get_season_colors(df['season'].tolist())
    
    bars = ax.bar(df['season'], df['reversal_rate'] * 100, color=colors, 
                  edgecolor=PALETTE['aux'], linewidth=0.8, alpha=0.9)
    
    # 平均线 - 使用基准色
    avg_rate = df['reversal_rate'].mean() * 100
    ax.axhline(y=avg_rate, color=PALETTE['baseline'], linestyle='--', linewidth=2, 
               label=f'Average: {avg_rate:.1f}%')
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Reversal Rate (%)', fontsize=12)
    ax.set_title('Counterfactual Analysis: What If We Used the Other Voting Method?', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(df['season'])
    ax.set_ylim(0, 105)
    
    # 标注争议季节 - 使用警示色
    controversy_seasons = [2, 4, 11, 27]  # Jerry Rice, Billy Ray, Bristol Palin, Bobby Bones
    for s in controversy_seasons:
        if s in df['season'].values:
            rate = df[df['season'] == s]['reversal_rate'].values[0] * 100
            ax.annotate(f'S{s}', (s, rate), textcoords="offset points", 
                       xytext=(0, 8), ha='center', fontsize=9, fontweight='bold',
                       color=PALETTE['warning'])
    
    ax.legend(loc='upper right')
    apply_paper_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_counterfactual_rates.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_counterfactual_rates.pdf', bbox_inches='tight')
    print("✓ Figure 3: Counterfactual Comparison saved")
    plt.close()


def plot_fan_vote_distribution():
    """
    Figure 4: 粉丝投票估计分布 (Ghost Data)
    显示模型推断的隐藏粉丝投票
    """
    df = pd.read_csv('outputs/fan_vote_estimates.csv')
    
    # 只取有意义的估计
    df = df[df['fan_vote_estimate'] > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图 - 使用统一调色板
    ax1 = axes[0]
    ax1.hist(df['fan_vote_estimate'], bins=50, color=PALETTE['proposed'], 
             edgecolor=PALETTE['aux'], alpha=0.7, density=True)
    ax1.axvline(x=df['fan_vote_estimate'].mean(), color=PALETTE['warning'], linestyle='--', 
                linewidth=2, label=f"Mean: {df['fan_vote_estimate'].mean():.3f}")
    ax1.axvline(x=df['fan_vote_estimate'].median(), color=PALETTE['baseline'], linestyle='--', 
                linewidth=2, label=f"Median: {df['fan_vote_estimate'].median():.3f}")
    ax1.set_xlabel('Estimated Fan Vote Share', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Inferred Fan Votes', fontsize=14, fontweight='bold')
    ax1.legend()
    apply_paper_style(ax1)
    
    # 右图：按周的箱线图
    ax2 = axes[1]
    week_data = [df[df['week'] == w]['fan_vote_estimate'].values for w in range(1, 12)]
    week_data = [d for d in week_data if len(d) > 0]
    
    bp = ax2.boxplot(week_data, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(PALETTE['proposed'])
        patch.set_alpha(0.7)
        patch.set_edgecolor(PALETTE['aux'])
    for whisker in bp['whiskers']:
        whisker.set_color(PALETTE['aux'])
    for cap in bp['caps']:
        cap.set_color(PALETTE['aux'])
    for median in bp['medians']:
        median.set_color(PALETTE['warning'])
        median.set_linewidth(2)
    
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Estimated Fan Vote Share', fontsize=12)
    ax2.set_title('Fan Vote Distribution by Week', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([f'W{i}' for i in range(1, len(week_data)+1)])
    apply_paper_style(ax2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_fan_vote_distribution.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_fan_vote_distribution.pdf', bbox_inches='tight')
    print("✓ Figure 4: Fan Vote Distribution saved")
    plt.close()


def plot_pro_dancer_effect():
    """
    Figure 5: 职业舞者效应
    显示不同职业舞者搭档的选手表现
    """
    df = pd.read_csv('outputs/cleaned_contestants.csv')
    
    # 统计每位舞者的选手平均名次
    dancer_stats = df.groupby('ballroom_partner').agg({
        'placement': ['mean', 'std', 'count'],
        'elimination_week': 'mean'
    }).reset_index()
    dancer_stats.columns = ['ballroom_partner', 'avg_placement', 'std_placement', 
                            'n_partners', 'avg_elim_week']
    
    # 只取有足够数据的舞者 (至少3位搭档)
    dancer_stats = dancer_stats[dancer_stats['n_partners'] >= 3].sort_values('avg_placement')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 水平条形图 - 使用辅助色
    y_pos = range(len(dancer_stats))
    bars = ax.barh(y_pos, dancer_stats['avg_placement'], 
                   xerr=dancer_stats['std_placement'] / np.sqrt(dancer_stats['n_partners']),
                   color=PALETTE['aux'], alpha=0.85, edgecolor=PALETTE['baseline'], 
                   capsize=3, linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dancer_stats['ballroom_partner'])
    ax.set_xlabel('Average Placement (lower is better)', fontsize=12)
    ax.set_title('Professional Dancer Effect on Partner Placement', fontsize=14, fontweight='bold')
    ax.invert_xaxis()  # 反转x轴，让更好的名次在右边
    apply_paper_style(ax)
    
    # 添加样本量注释 - 使用标注色
    for i, (_, row) in enumerate(dancer_stats.iterrows()):
        ax.annotate(f'n={int(row["n_partners"])}', 
                   (row['avg_placement'] - 0.3, i),
                   fontsize=8, va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_pro_dancer_effect.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_pro_dancer_effect.pdf', bbox_inches='tight')
    print("✓ Figure 5: Pro Dancer Effect saved")
    plt.close()


def plot_voting_method_comparison():
    """
    Figure 6: 投票方法对比分析
    百分比制 vs 排名制的特性对比
    """
    # 读取数据
    cf_df = pd.read_csv('outputs/counterfactual_results.csv')
    inc_df = pd.read_csv('outputs/inconsistency_spectrum.csv')
    
    # 分组
    percent_seasons = list(range(3, 28))
    rank_seasons = [1, 2] + list(range(28, 35))
    
    pct_reversals = cf_df[cf_df['season'].isin(percent_seasons)]['reversal_rate'].mean()
    rank_reversals = cf_df[cf_df['season'].isin(rank_seasons)]['reversal_rate'].mean()
    
    pct_inconsistency = inc_df[inc_df['season'].isin(percent_seasons)]['inconsistency_score'].mean()
    rank_inconsistency = inc_df[inc_df['season'].isin(rank_seasons)]['inconsistency_score'].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：逆转率对比 - 使用统一调色板
    ax1 = axes[0]
    methods = ['Percent\n(S3-27)', 'Rank\n(S1-2, S28+)']
    reversals = [pct_reversals * 100, rank_reversals * 100]
    bars1 = ax1.bar(methods, reversals, color=[PALETTE['proposed'], PALETTE['baseline']], 
                    edgecolor=PALETTE['aux'], linewidth=1.2)
    ax1.set_ylabel('Average Reversal Rate (%)', fontsize=12)
    ax1.set_title('Sensitivity to Method Change', fontsize=14, fontweight='bold')
    for bar, val in zip(bars1, reversals):
        ax1.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(reversals) * 1.2)
    apply_paper_style(ax1)
    
    # 右图：不一致性对比 - 使用统一调色板
    ax2 = axes[1]
    inconsistencies = [pct_inconsistency, rank_inconsistency]
    bars2 = ax2.bar(methods, inconsistencies, color=[PALETTE['proposed'], PALETTE['baseline']],
                    edgecolor=PALETTE['aux'], linewidth=1.2)
    ax2.set_ylabel('Average Inconsistency Score (S*)', fontsize=12)
    ax2.set_title('Model Fit Quality', fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, inconsistencies):
        ax2.annotate(f'{val:.2f}', (bar.get_x() + bar.get_width()/2, val + 0.02),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(inconsistencies) * 1.3 + 0.1)
    apply_paper_style(ax2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_method_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_method_comparison.pdf', bbox_inches='tight')
    print("✓ Figure 6: Voting Method Comparison saved")
    plt.close()


def plot_controversy_case_study():
    """
    Figure 7: 争议案例研究
    Bristol Palin (S11), Bobby Bones (S27) 等
    """
    df = pd.read_csv('outputs/cleaned_contestants.csv')
    scores_df = pd.read_csv('outputs/cleaned_scores.csv')
    
    # 选取争议案例
    controversy_cases = [
        ('Bristol Palin', 11, 'Finished 3rd despite low scores'),
        ('Bobby Bones', 27, 'Won despite lowest average score'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, season, description) in enumerate(controversy_cases):
        ax = axes[idx]
        
        # 获取该季所有选手
        season_df = df[df['season'] == season].copy()
        season_scores = scores_df[scores_df['season'] == season].copy()
        
        # 计算每位选手的平均分
        avg_scores = season_scores.groupby('contestant')['total_score'].mean().reset_index()
        avg_scores.columns = ['celebrity_name', 'avg_score']
        
        season_df = season_df.merge(avg_scores, on='celebrity_name', how='left')
        season_df = season_df.sort_values('placement')
        
        # 绘制 - 使用统一调色板: 争议选手用警示色, 其他用主色
        colors = [PALETTE['warning'] if name.lower() in n.lower() else PALETTE['proposed'] 
                  for n in season_df['celebrity_name']]
        
        bars = ax.barh(range(len(season_df)), season_df['avg_score'], color=colors,
                       edgecolor=PALETTE['aux'], alpha=0.85, linewidth=0.8)
        
        ax.set_yticks(range(len(season_df)))
        ax.set_yticklabels([f"{row['placement']}. {row['celebrity_name'][:15]}" 
                          for _, row in season_df.iterrows()], fontsize=9)
        ax.set_xlabel('Average Judge Score', fontsize=11)
        ax.set_title(f'Season {season}: {description}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        apply_paper_style(ax)
        
        # 标注争议选手 - 使用警示色
        controversy_row = season_df[season_df['celebrity_name'].str.contains(name.split()[0], case=False)]
        if len(controversy_row) > 0:
            placement = controversy_row['placement'].values[0]
            ax.annotate('← Controversy', 
                       (controversy_row['avg_score'].values[0], placement - 1),
                       fontsize=10, color=PALETTE['warning'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_controversy_cases.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig7_controversy_cases.pdf', bbox_inches='tight')
    print("✓ Figure 7: Controversy Case Studies saved")
    plt.close()


def plot_survival_curve():
    """
    Figure 8: 生存曲线
    按职业舞者分组的"存活"概率
    """
    df = pd.read_csv('outputs/cleaned_contestants.csv')
    
    # 按职业舞者分组，计算存活曲线
    top_dancers = df.groupby('ballroom_partner')['placement'].mean().nsmallest(5).index.tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用统一调色板的颜色序列
    survival_colors = [PALETTE['proposed'], PALETTE['baseline'], PALETTE['warning'], 
                       PALETTE['aux'], PALETTE['warning2']]
    
    for i, dancer in enumerate(top_dancers):
        dancer_df = df[df['ballroom_partner'] == dancer]
        
        # 计算每周的存活率
        max_week = 11
        survival = []
        for week in range(1, max_week + 1):
            survived = (dancer_df['elimination_week'] >= week).sum() / len(dancer_df)
            survival.append(survived)
        
        ax.step(range(1, max_week + 1), survival, where='mid', 
                label=f'{dancer} (n={len(dancer_df)})', color=survival_colors[i], linewidth=2)
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Survival Curves by Professional Dancer Partner', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=9)
    apply_paper_style(ax)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_survival_curves.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig8_survival_curves.pdf', bbox_inches='tight')
    print("✓ Figure 8: Survival Curves saved")
    plt.close()


if __name__ == '__main__':
    print("=" * 50)
    print("Generating DWTS Visualizations...")
    print("=" * 50)
    
    # 确保所有数据文件存在
    required_files = [
        'outputs/inconsistency_spectrum.csv',
        'outputs/reversal_heatmap.csv', 
        'outputs/counterfactual_results.csv',
        'outputs/fan_vote_estimates.csv',
        'outputs/cleaned_contestants.csv'
    ]
    
    if not all(Path(f).exists() for f in required_files):
        print("Generating data files first...")
        ensure_data_files()
    
    plot_inconsistency_spectrum()
    plot_reversal_heatmap()
    plot_counterfactual_comparison()
    plot_fan_vote_distribution()
    plot_pro_dancer_effect()
    plot_voting_method_comparison()
    plot_controversy_case_study()
    plot_survival_curve()
    
    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 50)
