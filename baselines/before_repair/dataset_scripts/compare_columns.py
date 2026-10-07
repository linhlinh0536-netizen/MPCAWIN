import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

FILE_2017 = BASE / "cicids2017-imp" / "Reformatted_CIC-IDS2017-Imp.csv"
FILE_2018 = BASE / "cicids2018-imp" / "Reformatted_CSE-CIC-IDS2018-Imp.csv"

cols17 = pd.read_csv(FILE_2017, nrows=0).columns.tolist()
cols18 = pd.read_csv(FILE_2018, nrows=0).columns.tolist()

print("=== 2017 ===")
print("Number of columns:", len(cols17))

print("\n=== 2018 ===")
print("Number of columns:", len(cols18))

print("\n=== EXACTLY SAME COLUMNS? ===")
print(cols17 == cols18)

only17 = [c for c in cols17 if c not in cols18]
only18 = [c for c in cols18 if c not in cols17]

print("\n=== ONLY IN 2017 ===")
print(only17)

print("\n=== ONLY IN 2018 ===")
print(only18)

common_cols = [c for c in cols17 if c in cols18]

print("\n=== COMMON COLUMNS ===")
print("Number:", len(common_cols))
print(common_cols)