import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
FILE = BASE / "ImprTotal.csv"

BATCH_SIZE = 2000
DRIFT_ROW = 160000

print("Reading ImprTotal...")
df = pd.read_csv(FILE)

print("\n=== DATASET ===")
print("Shape:", df.shape)

total_rows = len(df)
num_batches = total_rows // BATCH_SIZE

print("\n=== BATCH INFO ===")
print("Batch size:", BATCH_SIZE)
print("Number of full batches:", num_batches)

# Điểm chuyển 2017 -> 2018
drift_batch = DRIFT_ROW // BATCH_SIZE

print("\n=== DRIFT BOUNDARY ===")
print("2018 starts at row:", DRIFT_ROW)
print("Theoretical drift batch:", drift_batch)

summary = []

for batch_id in range(num_batches):

    start = batch_id * BATCH_SIZE
    end = start + BATCH_SIZE

    batch = df.iloc[start:end]

    # Vì mình biết boundary ở row 160000
    if end <= DRIFT_ROW:
        source = "Impr2017"
    elif start >= DRIFT_ROW:
        source = "Impr2018"
    else:
        source = "Mixed"

    attack_count = int((batch["attack_flag"] == 1).sum())
    benign_count = int((batch["attack_flag"] == 0).sum())

    summary.append({
        "batch_id": batch_id,
        "start_row": start,
        "end_row": end - 1,
        "source": source,
        "benign": benign_count,
        "attack": attack_count,
        "attack_ratio": attack_count / len(batch)
    })

summary_df = pd.DataFrame(summary)

print("\n=== FIRST 5 BATCHES ===")
print(summary_df.head())

print("\n=== AROUND DRIFT ===")
print(
    summary_df[
        (summary_df["batch_id"] >= drift_batch - 2) &
        (summary_df["batch_id"] <= drift_batch + 2)
    ]
)

print("\n=== LAST 5 BATCHES ===")
print(summary_df.tail())

OUT = BASE / "batch_summary.csv"
summary_df.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)