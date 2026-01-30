"""
诊断退化样本（区间宽度接近1.0）
识别哪些赛季/周次存在规则约束不足的问题
"""
import pandas as pd
import numpy as np

# 加载MC结果
df = pd.read_csv('outputs/mc_robustness_results.csv')

# 计算区间宽度
df['interval_width'] = df['fan_vote_upper'] - df['fan_vote_lower']

# 找出区间宽度接近1.0的样本（退化情况）
degenerate = df[df['interval_width'] > 0.95].sort_values(['season', 'week'])

print("\n" + "="*80)
print("【警报】区间宽度接近1.0（完全无约束）的样本诊断")
print("="*80)
print(f"\n总数: {len(degenerate)}/{len(df)} ({100*len(degenerate)/len(df):.1f}%)")

if len(degenerate) > 0:
    print("\n【按赛季分组统计】")
    season_counts = degenerate['season'].value_counts().sort_index()
    print("-" * 50)
    for season, count in season_counts.items():
        pct = 100 * count / len(df[df['season'] == season])
        print(f"  S{season:2d}: {count:3d}个案例 ({pct:5.1f}% 该赛季)")
    
    print("\n【详细列表】所有退化样本（interval_width > 0.95）")
    print("-" * 80)
    cols_show = ['season', 'week', 'contestant', 'interval_width', 'p_wrongful', 'voting_method']
    for col in cols_show:
        if col not in degenerate.columns:
            cols_show.remove(col)
    
    display_df = degenerate[cols_show].copy()
    display_df['season'] = display_df['season'].astype(int)
    display_df['week'] = display_df['week'].astype(int)
    display_df['interval_width'] = display_df['interval_width'].apply(lambda x: f"{x:.4f}")
    display_df['p_wrongful'] = display_df['p_wrongful'].apply(lambda x: f"{x:.3f}")
    
    print(display_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("【诊断：这些赛季的规则约束不足】")
    print("="*80)
    
    # 按赛季检查规则
    for season in sorted(degenerate['season'].unique()):
        subset = degenerate[degenerate['season'] == season]
        all_in_season = df[df['season'] == season]
        voting_method = subset['voting_method'].iloc[0]
        
        # 识别特殊赛季
        if season == 28:
            note = "⚠️  引入Judge Save的第一个赛季"
        elif season in [32, 33]:
            note = "⚠️  高度模型-数据不匹配（S* > 0）"
        elif season in [1, 2]:
            note = "⚠️  初代赛季，规则不成熟"
        elif season >= 28:
            note = "⚠️  Judge Save时代（S28+）"
        else:
            note = ""
        
        print(f"\n  S{season} ({voting_method}): {len(subset)}/{len(all_in_season)}个样本（{100*len(subset)/len(all_in_season):.1f}%）{note}")
        
        # 显示周次
        weeks = sorted(subset['week'].unique())
        print(f"    周次: {weeks}")
        
        # 显示该赛季的均值
        print(f"    退化样本的平均P(W): {subset['p_wrongful'].mean():.3f}")
        print(f"    全赛季平均P(W):     {all_in_season['p_wrongful'].mean():.3f}")

else:
    print("\n✓ 未发现退化样本（所有样本 interval_width < 0.95）")

# 额外分析：区间宽度分布
print("\n" + "="*80)
print("【区间宽度分布统计】")
print("="*80)

print(f"\n  最小值: {df['interval_width'].min():.4f}")
print(f"  25分位: {df['interval_width'].quantile(0.25):.4f}")
print(f"  中位数: {df['interval_width'].median():.4f}")
print(f"  75分位: {df['interval_width'].quantile(0.75):.4f}")
print(f"  最大值: {df['interval_width'].max():.4f}")
print(f"  平均值: {df['interval_width'].mean():.4f}")
print(f"  标准差: {df['interval_width'].std():.4f}")

# 按阈值分组
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
print(f"\n区间宽度分组：")
print("-" * 50)
for i in range(len(thresholds) - 1):
    lower = thresholds[i]
    upper = thresholds[i + 1]
    count = len(df[(df['interval_width'] >= lower) & (df['interval_width'] < upper)])
    pct = 100 * count / len(df)
    print(f"  [{lower:.1f}, {upper:.1f}): {count:3d}个 ({pct:5.1f}%)")

count_max = len(df[df['interval_width'] >= 1.0])
pct_max = 100 * count_max / len(df)
print(f"  [{thresholds[-1]:.1f}, ∞):   {count_max:3d}个 ({pct_max:5.1f}%)")

print("\n" + "="*80)
print("【结论与建议】")
print("="*80)

if len(degenerate) > 0:
    pct_deg = 100 * len(degenerate) / len(df)
    print(f"""
1️⃣  约{pct_deg:.1f}%的样本出现完全退化（宽度≥0.95）
    → 说明这些周次的约束信息不足，可能原因：
       • Judge Save等规则变化导致约束减少
       • 数据缺失（无底部信息、无周次详情）
       • 模型未能捕捉隐含的选择逻辑

2️⃣  重灾区在 S{list(season_counts.index)}
    → 这些赛季需要在论文中单独标注或补充规则建模

3️⃣  建议论文中添加一句话：
    "对于约{pct_deg:.1f}%出现区间退化的样本，我们在后续敏感性分析中
     进行了单独处理，或在讨论中标注为'规则信息不足周次'。"

4️⃣  下一步行动：
    • 检查S{list(season_counts.index)}这些赛季的规则定义是否完整
    • 补充Judge Save、双淘汰等高阶规则的约束
    • 若数据确实缺失，则在论文中声明局限性
""")
else:
    print("""
✓ 所有样本的区间宽度都 < 0.95，说明：
  • 模型约束很紧密
  • 观众票反演整体确定性高
  • 无明显建模缺陷

建议论文中添加一句话：
"所有样本均显示有效的约束（区间宽度 < 0.95），表明所采规则框架
 充分捕捉了主要的淘汰逻辑。"
""")

print("="*80)
