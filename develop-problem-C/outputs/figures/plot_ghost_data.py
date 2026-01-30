import pandas as pd
df = pd.read_csv('ghost_data.csv')

import matplotlib.pyplot as plt
import seaborn as sns

# Ghost in the Data Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Create boxplot of interval widths by season
sns.boxplot(data=df, x='season', y='interval_width', hue='rule_era', ax=ax)

# Add vertical lines at rule change points
ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.5, label='Percent Era Start')
ax.axvline(x=27.5, color='blue', linestyle='--', alpha=0.5, label='Judges Save Start')

ax.set_xlabel('Season')
ax.set_ylabel('Fan Vote Interval Width')
ax.set_title('Ghost in the Data: Fan Vote Estimation Uncertainty by Season')
ax.legend()
plt.tight_layout()
plt.savefig('ghost_data.pdf', dpi=300, bbox_inches='tight')
