import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

from mpcawin import MPCAWIN

DATA_FILE = (
    Path.home()
    / "Downloads"
    / "hoids"
    / "datasets-main"
    / "ImprTotal.csv"
)

BATCH_SIZE = 2000
DRIFT_BATCH = 80

print("Reading data...")
df = pd.read_csv(DATA_FILE)

DROP_COLUMNS = [
    "id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Attempted Category",
    "attack_name",
    "attack_flag",
    "attack_step",
]

X = df.drop(
    columns=[c for c in DROP_COLUMNS if c in df.columns]
)

X = X.select_dtypes(include=[np.number])

print("Shape:", X.shape)

summary = []

for c_value in [2, 3, 4, 5, 6]:

    print("\n" + "=" * 60)
    print("TEST c =", c_value)
    print("=" * 60)

    detector = MPCAWIN(
        min_window_size=10,
        max_window_size=10,
        coef_std=c_value
    )

    alerts_before = []
    alerts_after = []
    detections = []
    p_values = []

    num_batches = len(X) // BATCH_SIZE

    for batch_id in range(num_batches):

        start = batch_id * BATCH_SIZE
        end = start + BATCH_SIZE

        batch = X.iloc[start:end]

        threshold_before = detector.threshold

        drift = detector.add_element_and_detected_change(batch)

        angle = detector.last_angle

        if (
            angle is not None
            and threshold_before is not None
        ):
            alert = int(angle > threshold_before)
        else:
            alert = None

        if alert is not None:

            if 10 <= batch_id < DRIFT_BATCH:
                alerts_before.append(alert)

            if DRIFT_BATCH <= batch_id < DRIFT_BATCH + 20:
                alerts_after.append(alert)

        p = getattr(
            detector.detector,
            "p_value",
            None
        )

        if p is not None:
            p_values.append(p)

        if drift:
            detections.append(batch_id)

    before_ones = sum(alerts_before)
    after_ones = sum(alerts_after)

    min_p = min(p_values) if p_values else None

    print("Pre-drift alerts  :", before_ones, "/", len(alerts_before))
    print("Post-drift alerts :", after_ones, "/", len(alerts_after))
    print("Minimum p-value   :", min_p)
    print("Detected batches  :", detections)

    summary.append({
        "c": c_value,
        "pre_drift_alerts": before_ones,
        "post_drift_alerts": after_ones,
        "min_p_value": min_p,
        "detected_batches": str(detections)
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

summary_df.to_csv(
    BASE / "c_sensitivity_results.csv",
    index=False
)

print("\nSaved: c_sensitivity_results.csv")