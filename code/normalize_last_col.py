import pandas as pd
from pathlib import Path

file_path = Path(r"d:\2026-repo\data\task3_dataset_full.csv")

# Read the CSV
df = pd.read_csv(file_path)

# Retrieve the last column name
last_col = df.columns[-1]
print(f"Normalizing column: {last_col}")

# Get statistics before normalization
print("Before normalization:")
print(df[last_col].describe())

# Apply Z-score Standardization (StandardScaler)
col_mean = df[last_col].mean()
col_std = df[last_col].std()
df[last_col] = (df[last_col] - col_mean) / col_std

# Get statistics after standardization
print("\nAfter Z-score standardization:")
print(df[last_col].describe())

# Save to a new file to avoid PermissionError if the original is open
output_path = file_path.with_name("task3_dataset_full_zscored.csv")
df.to_csv(output_path, index=False)
print(f"File saved to {output_path}")
