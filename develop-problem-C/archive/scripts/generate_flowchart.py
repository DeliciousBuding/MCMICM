
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import sys
from pathlib import Path

# Add dwts_model to path to import palette
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dwts_model.paper_palette import PALETTE
except ImportError:
    # Fallback palette if import fails
    PALETTE = {
        "proposed": "#219EBC",
        "baseline": "#02304A",
        "warning":  "#FA8600",
        "warning2": "#FF9E02",
        "fill":     "#90C9E7",
        "aux":      "#136783",
        "accent":   "#FEB705"
    }

# Setup
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['figure.dpi'] = 300

def draw_node(ax, x, y, width, height, text, color, header=None, subtext=None, style='process'):
    """Draw a professional node with consistent styling"""
    
    # Styles
    if style == 'input':
        fc = 'white'
        ec = color
        alpha=1.0
        text_color = '#333'
    elif style == 'process':
        fc = 'white'
        ec = color
        alpha=1.0
        text_color = '#333'
    elif style == 'decision':
        fc = color
        ec = 'none'
        alpha=1.0
        text_color = 'white'
    elif style == 'terminator':
        fc = color
        ec = 'none'
        alpha=1.0
        text_color = 'white'
    
    # Box
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         fc=fc, ec=ec, lw=1.5, zorder=3, alpha=alpha)
    
    # Shadow effect
    shadow = FancyBboxPatch((x - width/2 + 0.05, y - height/2 - 0.05), width, height,
                            boxstyle="round,pad=0.05,rounding_size=0.15",
                            fc='#ddd', ec='none', zorder=2, alpha=0.5)
    
    ax.add_patch(shadow)
    ax.add_patch(box)
    
    # Text
    if header:
        ax.text(x, y + height/4, header, ha='center', va='center', fontweight='bold', fontsize=10, color=text_color, zorder=4)
        ax.text(x, y - height/4, subtext, ha='center', va='center', fontsize=8.5, color=text_color, zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=10, color=text_color, zorder=4)
        
    return box

def draw_arrow(ax, start, end, color='#555', style='curved'):
    """Draw arrows safely"""
    x1, y1 = start
    x2, y2 = end
    
    # Use straight lines for maximum stability but styled nicely
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle="->", color=color, lw=2.0, 
                             shrinkA=2, shrinkB=2,
                             connectionstyle="arc3,rad=0"))

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # --- Background Zones ---
    # We define 3 zones: Data, Dual-Engine, Outcome
    zones = [
        (0.5, 4, "DATA INPUTS", "#f8f9fa"),
        (4.5, 5, "MODELING ENGINE", "#f0f7fb"),
        (9.5, 4, "AUDIT & OUTCOMES", "#fff8f0")
    ]
    
    # Draw zone separators vertically
    plt.axvline(x=4.25, color='#ddd', linestyle='--', lw=1)
    plt.axvline(x=9.25, color='#ddd', linestyle='--', lw=1)
    
    # Zone Headers
    ax.text(2.1, 8.0, "PHASE 1: DATA INPUTS", ha='center', fontweight='bold', fontsize=12, color='#666')
    ax.text(6.8, 8.0, "PHASE 2: DUAL-CORE ENGINE", ha='center', fontweight='bold', fontsize=12, color=PALETTE['aux'])
    ax.text(11.7, 8.0, "PHASE 3: AUDIT & OUTCOME", ha='center', fontweight='bold', fontsize=12, color=PALETTE['warning'])

    # --- Nodes ---
    
    # 1. Inputs
    node_judge = dict(x=2.1, y=6.0, w=2.4, h=1.2)
    draw_node(ax, node_judge['x'], node_judge['y'], node_judge['w'], node_judge['h'], 
              None, PALETTE['baseline'], header="JUDGE SCORES", subtext="Observed (0-30/40)", style='input')
              
    node_fan = dict(x=2.1, y=3.0, w=2.4, h=1.2)
    draw_node(ax, node_fan['x'], node_fan['y'], node_fan['w'], node_fan['h'], 
              None, PALETTE['baseline'], header="FAN VOTES", subtext="Latent / Hidden %", style='input')
              
    # 2. Engines
    # Percent Engine
    node_lp = dict(x=6.8, y=6.0, w=3.2, h=1.4)
    draw_node(ax, node_lp['x'], node_lp['y'], node_lp['w'], node_lp['h'], 
              None, PALETTE['aux'], header="LINEAR PROGRAMMING", subtext="Percent Era (S3-S27)\nDirect Sum Constraints", style='process')
    
    # Rank Engine
    node_milp = dict(x=6.8, y=3.0, w=3.2, h=1.4)
    draw_node(ax, node_milp['x'], node_milp['y'], node_milp['w'], node_milp['h'], 
              None, PALETTE['baseline'], header="MILP SOLVER", subtext="Rank Era (S1-2, S28+)\nOrder-Based Logic", style='process')

    # 3. Decision
    # Bottom Two (Diamond) - Moved slightly right to make room
    bx, by = 10.6, 4.5
    # Shadow for diamond
    diamond_shadow = mpatches.RegularPolygon((bx + 0.05, by - 0.05), numVertices=4, radius=0.9, 
                                    fc='#ddd', ec='none', zorder=2, alpha=0.5)
    ax.add_patch(diamond_shadow)
    
    diamond = mpatches.RegularPolygon((bx, by), numVertices=4, radius=0.9, 
                                    fc=PALETTE['warning'], ec='none', zorder=3)
    ax.add_patch(diamond)
    ax.text(bx, by, "Bottom\nTwo?", ha='center', va='center', color='white', fontweight='bold', fontsize=10, zorder=5)
    
    # Outcomes
    node_elim = dict(x=12.8, y=6.5, w=2.0, h=0.8)
    draw_node(ax, node_elim['x'], node_elim['y'], node_elim['w'], node_elim['h'], 
              "ELIMINATION", PALETTE['warning'], style='terminator')
              
    node_safe = dict(x=12.8, y=2.5, w=2.0, h=0.8)
    draw_node(ax, node_safe['x'], node_safe['y'], node_safe['w'], node_safe['h'], 
              "SURVIVES", PALETTE['proposed'], style='terminator')
              
    # Save Logic - Increased size
    node_save = dict(x=10.6, y=6.5, w=1.6, h=0.7)
    draw_node(ax, node_save['x'], node_save['y'], node_save['w'], node_save['h'], 
              "Judge Save?", PALETTE['accent'], style='input')

    # --- Connections (The "Advanced" look comes from clean orthogonal lines) ---
    
    # Inputs to Engines
    # Judge -> LP
    draw_arrow(ax, (node_judge['x'] + node_judge['w']/2, node_judge['y']), (node_lp['x'] - node_lp['w']/2, node_lp['y']), style='ortho')
    # Judge -> MILP
    draw_arrow(ax, (node_judge['x'] + node_judge['w']/2, node_judge['y']), (node_milp['x'] - node_milp['w']/2, node_milp['y']), style='ortho')
    
    # Fan -> LP
    draw_arrow(ax, (node_fan['x'] + node_fan['w']/2, node_fan['y']), (node_lp['x'] - node_lp['w']/2, node_lp['y']), style='ortho')
    # Fan -> MILP
    draw_arrow(ax, (node_fan['x'] + node_fan['w']/2, node_fan['y']), (node_milp['x'] - node_milp['w']/2, node_milp['y']), style='ortho')
    
    # Engines to Decision
    # LP -> Decision
    draw_arrow(ax, (node_lp['x'] + node_lp['w']/2, node_lp['y']), (bx - 0.8, by + 0.2), style='ortho')
    # MILP -> Decision
    draw_arrow(ax, (node_milp['x'] + node_milp['w']/2, node_milp['y']), (bx - 0.8, by - 0.2), style='ortho')

    # Decision Logic
    # No -> Safe (Bottom tip ~3.6)
    ax.annotate("No", xy=(10.6, 2.8), xytext=(10.6, 3.4), ha='center', fontsize=10, color='#555',
               arrowprops=dict(arrowstyle="->", color=PALETTE['proposed'], lw=2))
    
    # Yes -> Save? (Top tip ~5.4)
    # Target is bottom of save node (y=6.5 - 0.35 = 6.15). Tip is 5.4.
    ax.annotate("Yes", xy=(10.6, 6.15), xytext=(10.6, 5.6), ha='center', fontsize=10, color='#555',
               arrowprops=dict(arrowstyle="->", color=PALETTE['warning'], lw=2))

    # Save logic
    # Save -> Safe (Loop around)
    # Save node right edge is x=10.6+0.8=11.4. y=6.5
    # Use arc3 for safe curved connection
    ax.annotate("Saved", xy=(12.8, 2.9), xytext=(11.5, 6.5), ha='left', fontsize=10, color=PALETTE['proposed'],
               arrowprops=dict(arrowstyle="->", color=PALETTE['proposed'], lw=1.5, connectionstyle="arc3,rad=-0.4"))
    
    # Save -> Elim
    # Elim node left edge x=12.8-1.0=11.8.
    ax.annotate("Not Saved", xy=(11.8, 6.5), xytext=(11.5, 6.8), ha='left', fontsize=9, color=PALETTE['warning'],
               arrowprops=dict(arrowstyle="->", color=PALETTE['warning'], lw=1.5))


    # Legend / Info
    plt.text(7, 1.2, "Dual-Core Inversion Engine detects mismatches (S*) > 0", 
             ha='center', va='center', fontsize=11, style='italic', color='#666',
             bbox=dict(fc='#f8f9fa', ec='#ddd', pad=0.8, boxstyle='round'))

    # Save
    output_path = Path("outputs/figures/fig_dwts_flowchart_vector.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(str(output_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Generated advanced flowchart at {output_path}")

if __name__ == "__main__":
    draw_flowchart()
