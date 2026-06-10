from pathlib import Path
import pandas as pd

PATH = Path("data/processed/training_positions.parquet")

df = pd.read_parquet(PATH)

# display fields + precise field info
print(df.head())
print()
df.info()
print()

print(f"Rows: {len(df)}")
print(f"Games: {df['game_uuid'].nunique()}")
print()

# display training rows/positions for each time control
print("Time classes:")
print(df["time_class"].value_counts(dropna=False)) # include NaNs
print()

# display move-time buckets
print("Move-time buckets:")
print(df["move_time_bucket"].value_counts(dropna=False))
print()

# display player colours
print("My colour:")
print(df["my_colour"].value_counts(dropna=False))
print()

# display rating stats
print("Ratings:")
desc_df = df[["my_rating", "opponent_rating", "rating_diff"]].describe()
print(desc_df)
print()

#! 6 NaNs in move-time buckets