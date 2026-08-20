import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

# read cleaned data
df = pd.read_csv("./data/processed/MIMIC/df_mimic_cleaned.csv")

print("=" * 60)
print("Dataset info")
print("=" * 60)
print(f"Total N: {len(df)}")
print(f"30-day mortality: {df['death_within_icu_30days'].mean()*100:.2f}%")
print(f"90-day mortality: {df['death_within_icu_90days'].mean()*100:.2f}%")
print(f"365-day mortality: {df['death_within_icu_365days'].mean()*100:.2f}%")

# ----- train/val split (80/20, stratified by 30-day) -----
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df['death_within_icu_30days']
)

print("\n" + "=" * 60)
print("Split results")
print("=" * 60)
print(f"Training set: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")

# check stratification
print("\nStratification check:")
for outcome in ['death_within_icu_30days', 'death_within_icu_90days', 'death_within_icu_365days']:
    train_rate = train_df[outcome].mean() * 100
    val_rate = val_df[outcome].mean() * 100
    print(f"  {outcome}: train {train_rate:.2f}%, val {val_rate:.2f}%")

# ----- save -----
os.makedirs("./data/splits/", exist_ok=True)
train_df.to_csv("./data/splits/train_set.csv", index=False)
val_df.to_csv("./data/splits/val_set.csv", index=False)

print("\nDone.")
print(f"Saved to ./data/splits/")