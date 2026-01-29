import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. 配置数据源 (解决 GeoPandas 1.0 兼容性)
# ==========================================
# 推荐使用 URL 自动下载。如果网络不畅，请手动下载 .zip 并将路径替换到 read_file 中
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
print("正在构建地理坐标系统...")
world = gpd.read_file(url)

# ==========================================
# 2. 核心配色方案 (清新学术风)
# ==========================================
COLORS = {
    'ocean': '#FFFFFF',         # 纯白背景 (海洋)
    'land_base': '#D1D9E0',     # 淡蓝灰色 (基础陆地，低调高级)
    'land_highlight': '#003366', # 深午夜蓝 (强调国家，视觉核心)
    'border': '#FFFFFF',        # 细白边框 (增加呼吸感)
    'text': '#1E293B'           # 深色文字
}

# ==========================================
# 3. 标记目标国家 (输入 ISO 代码)
# ==========================================
target_iso = ['BOL', 'CAF', 'TCD', 'MLI', 'NER', 'SOM', 'SSD', 'YEM', 'PNG', 'LAO', 'KHM', 'BIH', 'ALB', 'VNM']
world['is_target'] = world['ISO_A3'].isin(target_iso)

# ==========================================
# 4. 绘图流程
# ==========================================
# 转换为 Robinson 投影 (美赛论文加分项：视觉平衡感远超经纬度投影)
world_robin = world.to_crs("+proj=robin")

# 创建画布
fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['ocean'])
ax.set_facecolor(COLORS['ocean'])

# A. 绘制基础陆地 (淡蓝色)
world_robin.plot(
    ax=ax, 
    color=COLORS['land_base'], 
    edgecolor=COLORS['border'], 
    linewidth=0.5
)

# B. 绘制强调国家 (深蓝色)
world_robin[world_robin['is_target']].plot(
    ax=ax, 
    color=COLORS['land_highlight'], 
    edgecolor=COLORS['border'], 
    linewidth=0.7,
    zorder=3
)

# C. 细节隐藏：移除坐标轴和多余边框
ax.set_axis_off()

# D. 设置专业图例
highlight_patch = mpatches.Patch(color=COLORS['land_highlight'], label='Target Countries')
base_patch = mpatches.Patch(color=COLORS['land_base'], label='Other Countries')
legend = plt.legend(
    handles=[highlight_patch, base_patch],
    loc='lower left',
    bbox_to_anchor=(0.05, 0.1),
    frameon=False,
    fontsize=12,
    prop={'family': 'serif'}
)

# E. 图表标题 (符合国际学术规范)
plt.title('Figure 1: Global Distribution based on Specified Criteria', 
          fontsize=18, pad=20, fontfamily='serif', fontweight='bold', color=COLORS['text'])

# ==========================================
# 5. 高保真输出
# ==========================================
plt.tight_layout()
# 建议导出为 PDF 或 600DPI 的 PNG 放入 LaTeX 论文
plt.savefig('academic_blue_map.png', dpi=600, bbox_inches='tight')
print("【高级地图生成成功】文件保存为: academic_blue_map.png")
plt.show()