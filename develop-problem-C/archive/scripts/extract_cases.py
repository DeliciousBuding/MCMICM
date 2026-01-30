"""提取 Jerry Rice, Bobby Bones 等案例数据用于 Case Study"""
import pandas as pd

df = pd.read_csv('2026_MCM_Problem_C_Data.csv')

def get_week_judge_total(row, week):
    """获取某周的裁判总分"""
    scores = []
    for j in range(1, 5):
        col = f'week{week}_judge{j}_score'
        if col in df.columns:
            val = row.get(col)
            if pd.notna(val) and val != 'N/A':
                try:
                    scores.append(float(val))
                except:
                    pass
    return sum(scores) if scores else None

print("=" * 60)
print("CASE STUDY 1: Season 2 (Rank Rules) - Jerry Rice Controversy")
print("=" * 60)
s2 = df[df['season'] == 2].copy()

# 计算每位选手的平均分
for _, row in s2.iterrows():
    scores = [get_week_judge_total(row, w) for w in range(1, 11)]
    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"{row['celebrity_name']:20s} | Place: {row['placement']} | Avg Judge: {avg:.1f}")

print("\n" + "=" * 60)
print("CASE STUDY 2: Season 27 (Percent Rules) - Bobby Bones Upset Win")
print("=" * 60)
s27 = df[df['season'] == 27].copy()

for _, row in s27.iterrows():
    scores = [get_week_judge_total(row, w) for w in range(1, 11)]
    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"{row['celebrity_name']:20s} | Place: {row['placement']} | Avg Judge: {avg:.1f}")

print("\n" + "=" * 60)
print("CASE STUDY 3: Season 32 (Rank + Judge Save) - Anomalous Season")
print("=" * 60)
s32 = df[df['season'] == 32].copy()

for _, row in s32.iterrows():
    scores = [get_week_judge_total(row, w) for w in range(1, 11)]
    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"{row['celebrity_name']:20s} | Place: {row['placement']} | Avg Judge: {avg:.1f}")

print("\n" + "=" * 60)
print("DATA CHECK: Season Types")
print("=" * 60)
print("Percent Seasons (S3-S27): Fan vote PERCENTAGES disclosed")
print("Rank Seasons (S1-S2, S28+): Only RANKINGS disclosed (fan rank is UNKNOWN)")
