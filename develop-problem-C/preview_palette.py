"""
DWTS Paper Palette Preview Generator
生成颜色预览图，展示论文配色方案
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Import unified palette
import sys
sys.path.insert(0, '.')
from dwts_model.paper_palette import PALETTE, VOTING_METHODS, MECHANISMS

def create_palette_preview():
    """生成调色板预览图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('DWTS Paper Palette - 论文级配色方案', fontsize=16, fontweight='bold', y=0.98)
    
    # ============ 上图: 核心调色板 ============
    ax1 = axes[0]
    ax1.set_xlim(0, len(PALETTE))
    ax1.set_ylim(0, 1.5)
    ax1.set_title('核心配色 (Core Palette)', fontsize=14, pad=10)
    
    palette_items = list(PALETTE.items())
    for i, (name, color) in enumerate(palette_items):
        # 绘制色块
        rect = patches.FancyBboxPatch((i + 0.1, 0.3), 0.8, 0.9, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax1.add_patch(rect)
        
        # 颜色名称
        ax1.text(i + 0.5, 1.35, name, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 十六进制值
        ax1.text(i + 0.5, 0.1, color, ha='center', va='top', fontsize=9, 
                 family='monospace', color='#666666')
        
        # 使用场景说明
        usage_map = {
            'proposed': '新机制/推荐',
            'baseline': '基准/原机制', 
            'warning': '警示/失败',
            'warning2': '次级警示',
            'fill': '填充/区间',
            'accent': '小标注/箭头',
            'aux': '辅助/第三方'
        }
        ax1.text(i + 0.5, 0.75, usage_map.get(name, ''), ha='center', va='center', 
                 fontsize=8, color='white' if name in ['baseline', 'aux'] else 'black')
    
    ax1.axis('off')
    
    # ============ 下图: 应用示例 ============
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    ax2.set_title('应用示例 (Usage Examples)', fontsize=14, pad=10)
    
    # 投票方法对比柱状图示例
    methods = ['Percent', 'Rank']
    x_bar = [1, 2]
    heights = [0.7, 0.5]
    colors_bar = [PALETTE['proposed'], PALETTE['baseline']]
    for x, h, c in zip(x_bar, heights, colors_bar):
        rect = patches.Rectangle((x - 0.3, 0), 0.6, h * 2.5, facecolor=c, 
                                  edgecolor=PALETTE['aux'], linewidth=1)
        ax2.add_patch(rect)
    ax2.text(1.5, 2.8, '机制对比柱状图', ha='center', fontsize=10, fontweight='bold')
    ax2.text(1, -0.15, 'Percent', ha='center', fontsize=8)
    ax2.text(2, -0.15, 'Rank', ha='center', fontsize=8)
    
    # 热力图示例 - Mismatch检测
    for i, alpha in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        rect = patches.Rectangle((3.5 + i*0.5, 0.5), 0.45, 0.45, 
                                  facecolor=PALETTE['fill'], alpha=alpha)
        ax2.add_patch(rect)
        rect2 = patches.Rectangle((3.5 + i*0.5, 1.0), 0.45, 0.45, 
                                   facecolor=PALETTE['warning'], alpha=alpha)
        ax2.add_patch(rect2)
    ax2.text(4.75, 2.8, 'Mismatch热力图', ha='center', fontsize=10, fontweight='bold')
    ax2.text(4.75, 1.6, 'Warning', ha='center', fontsize=8, color=PALETTE['warning'])
    ax2.text(4.75, 0.2, 'Fill', ha='center', fontsize=8, color=PALETTE['aux'])
    
    # 曲线图示例
    x_line = np.linspace(7, 9.5, 50)
    y_proposed = 1 + 0.5 * np.sin(2 * np.pi * (x_line - 7) / 2.5)
    y_baseline = 0.8 + 0.3 * np.sin(2 * np.pi * (x_line - 7) / 2.5 + 0.5)
    
    ax2.plot(x_line, y_proposed, color=PALETTE['proposed'], linewidth=2.5, label='Proposed')
    ax2.plot(x_line, y_baseline, color=PALETTE['baseline'], linewidth=2, linestyle='--', label='Baseline')
    ax2.fill_between(x_line, y_proposed - 0.2, y_proposed + 0.2, 
                     color=PALETTE['fill'], alpha=0.3)
    ax2.scatter([8.2], [y_proposed[24]], s=60, color=PALETTE['warning'], zorder=5)
    ax2.annotate('Alert!', (8.2, y_proposed[24] + 0.1), fontsize=8, 
                 color=PALETTE['accent'], fontweight='bold')
    ax2.text(8.25, 2.8, '时序曲线图', ha='center', fontsize=10, fontweight='bold')
    
    ax2.axis('off')
    
    # ============ 底部说明 ============
    fig.text(0.5, 0.02, 
             '原则: 全文只用 3 个高饱和色 (baseline/proposed/warning)，其余用浅色+透明度+线型表达',
             ha='center', fontsize=10, style='italic', color='#666666')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # 保存
    plt.savefig('outputs/figures/palette_preview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('outputs/figures/palette_preview.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Palette preview saved to outputs/figures/")
    plt.close()


if __name__ == '__main__':
    create_palette_preview()
