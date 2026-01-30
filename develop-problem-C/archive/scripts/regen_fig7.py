#!/usr/bin/env python
"""
Regenerate Figure 7 with unified color palette
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')
from dwts_model.paper_palette import PALETTE, apply_paper_style

OUTPUT_DIR = Path('outputs/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv('outputs/cleaned_contestants.csv')
scores_df = pd.read_csv('outputs/cleaned_scores.csv')

# Figure 7: 争议案例研究
controversy_cases = [
    ('Bristol Palin', 11, 'Finished 3rd despite low scores'),
    ('Bobby Bones', 27, 'Won despite lowest average score'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, season, description) in enumerate(controversy_cases):
    ax = axes[idx]
    
    season_df = df[df['season'] == season].copy()
    season_scores = scores_df[scores_df['season'] == season].copy()
    
    avg_scores = season_scores.groupby('contestant')['total_score'].mean().reset_index()
    avg_scores.columns = ['celebrity_name', 'avg_score']
    
    season_df = season_df.merge(avg_scores, on='celebrity_name', how='left')
    season_df = season_df.sort_values('placement')
    
    # 使用统一调色板: 争议选手用警示色, 其他用主色
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
    
    # 标注争议选手
    controversy_row = season_df[season_df['celebrity_name'].str.contains(name.split()[0], case=False)]
    if len(controversy_row) > 0:
        placement = controversy_row['placement'].values[0]
        ax.annotate('<- Controversy', 
                   (controversy_row['avg_score'].values[0], placement - 1),
                   fontsize=10, color=PALETTE['warning'], fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_controversy_cases.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fig7_controversy_cases.pdf', bbox_inches='tight')
print('✓ Figure 7 regenerated with unified palette')
print(f'  proposed (cyan): {PALETTE["proposed"]}')
print(f'  warning (orange): {PALETTE["warning"]}')
print(f'  aux (dark cyan): {PALETTE["aux"]}')
