import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

FILE_2017 = BASE / "Impr2017_stratified.csv"
FILE_2018 = BASE / "Impr2018_stratified.csv"

OUT = BASE / "ImprTotal.csv"

print("Reading Impr2017...")
df17 = pd.read_csv(FILE_2017)

print("Reading Impr2018...")
df18 = pd.read_csv(FILE_2018)

print("\n=== INPUT ===")
print("Impr2017:", df17.shape)
print("Impr2018:", df18.shape)

# Kiểm tra schema
assert list(df17.columns) == list(df18.columns), \
    "2017 và 2018 không cùng columns!"

# QUAN TRỌNG:
# KHÔNG shuffle sau khi nối
impr_total = pd.concat(
    [df17, df18],
    ignore_index=True
)

print("\n=== IMPRTOTAL ===")
print("Shape:", impr_total.shape)

print("\n2017 starts at row: 0")
print("2017 ends at row:", len(df17) - 1)

print("2018 starts at row:", len(df17))
print("2018 ends at row:", len(impr_total) - 1)

# kiểm tra label tổng
print("\n=== TOTAL LABEL DISTRIBUTION ===")
print(impr_total["attack_flag"].value_counts())

print(
    "\nTotal attack ratio:",
    f"{impr_total['attack_flag'].mean() * 100:.2f}%"
)

impr_total.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)