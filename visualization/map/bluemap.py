import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. 数据源加载
# ==========================================
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
print("数据加载中，请稍候...")
world = gpd.read_file(url)

# ==========================================
# 2. 优化后的配色方案 (结合你喜欢的简约版 + ACADEMIC 强调色)
# ==========================================
COLORS = {
    'ocean': '#FFFFFF',          # 纯白背景
    'land_base': '#E2E8F0',      # 更清透的淡蓝灰 (参考陆地底色)
    'land_highlight': '#1f77b4', # 采用你提供的 ACADEMIC[0] (明亮学术蓝)
    'border': '#FFFFFF',         # 呼吸感白边
    'text': '#1E293B'            # 衬线体文字颜色
}

# ==========================================
# 3. 标记目标国家
# ==========================================
target_iso = ['BOL', 'CAF', 'TCD', 'MLI', 'NER', 'SOM', 'SSD', 'YEM', 'PNG', 'LAO', 'KHM', 'BIH', 'ALB', 'VNM']
world['is_target'] = world['ISO_A3'].isin(target_iso)

# ==========================================
# 4. 绘图核心流程
# ==========================================
# 关键步：Robinson 投影确保地图不拉伸变形
world_robin = world.to_crs("+proj=robin")

fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['ocean'])
ax.set_facecolor(COLORS['ocean'])

# A. 绘制基础陆地
world_robin.plot(
    ax=ax, 
    color=COLORS['land_base'], 
    edgecolor=COLORS['border'], 
    linewidth=0.5
)

# B. 绘制高亮国家 (使用 zorder 确保在顶层)
world_robin[world_robin['is_target']].plot(
    ax=ax, 
    color=COLORS['land_highlight'], 
    edgecolor=COLORS['border'], 
    linewidth=0.7,
    zorder=3
)

# C. 细节处理
ax.set_axis_off()

# D. 简约图例 (移除边框)
highlight_patch = mpatches.Patch(color=COLORS['land_highlight'], label='Target Countries')
base_patch = mpatches.Patch(color=COLORS['land_base'], label='Other Countries')
plt.legend(
    handles=[highlight_patch, base_patch],
    loc='lower left',
    bbox_to_anchor=(0.05, 0.1),
    frameon=False,
    fontsize=12,
    prop={'family': 'serif'}
)

# E. 标题 (符合 MCM/ICM 格式要求)
plt.title('Figure 1: Global Distribution of Target Countries', 
          fontsize=18, pad=20, fontfamily='serif', fontweight='bold', color=COLORS['text'])

# ==========================================
# 5. 论文级导出
# ==========================================
plt.tight_layout()
# 导出 600DPI 确保在 Word 或 LaTeX 中缩放不失真
plt.savefig('mcm_final_clean_map.png', dpi=600, bbox_inches='tight')
print("绘制成功！已生成 mcm_final_clean_map.png")
plt.show()