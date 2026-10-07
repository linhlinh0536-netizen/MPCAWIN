import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. SETUP
# ============================================================

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

from mpcawin import MPCAWIN


# Tắt warning SciPy khi KS-test tự chuyển exact -> asymptotic
warnings.filterwarnings(
    "ignore",
    message=".*Exact calculation unsuccessful.*",
    category=RuntimeWarning,
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = (
    Path.home()
    / "Downloads"
    / "hoids"
    / "datasets-main"
    / "ImprTotal.csv"
)

OUT_FILE = BASE / "mpcawin_results.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

BATCH_SIZE = 2000

# Impr2017 = 160,000 records
# 160,000 / 2,000 = 80 batch
# => Impr2018 bắt đầu từ batch 80
GROUND_TRUTH_DRIFT_BATCH = 80


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("READING IMPRTOTAL")
print("=" * 70)

print("File:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

# Không cho identifier / metadata / label vào PCA
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

existing_drop = [
    col
    for col in DROP_COLUMNS
    if col in df.columns
]

X = df.drop(columns=existing_drop)

# PCA chỉ dùng numerical features
X = X.select_dtypes(include=[np.number])


print("\n" + "=" * 70)
print("FEATURES USED BY MPCAWIN")
print("=" * 70)

print("Dropped columns:")
print(existing_drop)

print("\nNumber of numerical features:")
print(X.shape[1])

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# 5. DATA VALIDATION
# ============================================================

nan_count = int(X.isna().sum().sum())

X_numpy = X.to_numpy(dtype=float)

inf_count = int(np.isinf(X_numpy).sum())


print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("NaN:", nan_count)
print("Inf:", inf_count)


if nan_count > 0 or inf_count > 0:

    raise ValueError(
        "\nDataset contains NaN or Inf.\n"
        "Stop here before PCA."
    )


# ============================================================
# 6. CREATE MPCAWIN
# ============================================================

detector = MPCAWIN(
    min_window_size=10,
    max_window_size=10,
    coef_std=3
)


num_batches = len(X) // BATCH_SIZE

remainder = len(X) % BATCH_SIZE


print("\n" + "=" * 70)
print("MPCAWIN")
print("=" * 70)

print("Batch size:", BATCH_SIZE)
print("Number of full batches:", num_batches)
print("Remaining records:", remainder)

print(
    "Ground-truth drift starts at batch:",
    GROUND_TRUTH_DRIFT_BATCH
)
print("\n=== MPCAWIN CONFIG ===")
print("min_window_size =", detector.min_window_size)
print("max_window_size =", detector.max_window_size)
print("coef_std =", detector.coef_std)
print("KSWIN alpha =", detector.detector.alpha)
print("KSWIN window_size =", detector.detector.window_size)
print("KSWIN stat_size =", detector.detector.stat_size)

# ============================================================
# HELPER
# ============================================================

def array_to_string(arr):
    """
    Chuyển numpy array thành text đẹp để lưu CSV/debug.
    """

    if arr is None:
        return None

    arr = np.asarray(arr)

    return "[" + ",".join(
        str(int(x)) if float(x).is_integer()
        else str(round(float(x), 4))
        for x in arr
    ) + "]"


# ============================================================
# 7. STREAM DATA THROUGH MPCAWIN
# ============================================================

results = []

detected_batches = []


print("\n" + "=" * 70)
print("STREAM START")
print("=" * 70)


for batch_id in range(num_batches):

    # --------------------------------------------------------
    # Create batch
    # --------------------------------------------------------

    start = batch_id * BATCH_SIZE
    end = start + BATCH_SIZE

    batch = X.iloc[start:end]


    # --------------------------------------------------------
    # Dataset source
    # --------------------------------------------------------

    if batch_id < GROUND_TRUTH_DRIFT_BATCH:

        source = "Impr2017"

    else:

        source = "Impr2018"


    # --------------------------------------------------------
    # Threshold BEFORE evaluating current batch
    # --------------------------------------------------------

    threshold_before = detector.threshold


    # --------------------------------------------------------
    # Run official MPCAWIN function
    # --------------------------------------------------------

    drift = detector.add_element_and_detected_change(
        batch
    )


    # --------------------------------------------------------
    # MPC / SAD information
    # --------------------------------------------------------

    angle = detector.last_angle

    noise_angle = detector.last_noise_angle


    # SAD binary output:
    #
    # 0 = normal dissimilarity
    # 1 = angle vượt threshold
    #
    # Batch đầu hoặc lúc chưa đủ SAD window:
    # threshold = None
    # => alert = None

    if (
        angle is not None
        and threshold_before is not None
    ):

        alert = int(
            angle > threshold_before
        )

    else:

        alert = None


    # --------------------------------------------------------
    # KSWIN DEBUG
    # --------------------------------------------------------

    ks_detector = detector.detector

    ks_statistic = getattr(
        ks_detector,
        "statistic",
        None
    )

    ks_p_value = getattr(
        ks_detector,
        "p_value",
        None
    )

    ks_reference = getattr(
        ks_detector,
        "last_reference",
        None
    )

    ks_recent = getattr(
        ks_detector,
        "last_recent",
        None
    )

    ks_window = getattr(
        ks_detector,
        "window",
        None
    )


    if ks_window is not None:

        ks_window_length = len(
            ks_window
        )

    else:

        ks_window_length = None


    # --------------------------------------------------------
    # Save detection
    # --------------------------------------------------------

    if drift:

        detected_batches.append(
            batch_id
        )


    # --------------------------------------------------------
    # Save result row
    # --------------------------------------------------------

    results.append({

        "batch_id":
            batch_id,

        "start_row":
            start,

        "end_row":
            end - 1,

        "source":
            source,

        "angle":
            angle,

        "threshold_used":
            threshold_before,

        "alert":
            alert,

        "noise_angle":
            noise_angle,

        "ks_statistic":
            ks_statistic,

        "ks_p_value":
            ks_p_value,

        "ks_window_length":
            ks_window_length,

        "ks_reference":
            array_to_string(
                ks_reference
            ),

        "ks_recent":
            array_to_string(
                ks_recent
            ),

        "drift_detected":
            drift,
    })


    # --------------------------------------------------------
    # PRINT IMPORTANT BATCHES
    # --------------------------------------------------------

    # In:
    # - batch đầu
    # - batch 60 -> 100 để soi drift
    # - batch detect drift
    # - 3 batch cuối

    if (
        batch_id < 5
        or 60 <= batch_id <= 100
        or drift
        or batch_id >= num_batches - 3
    ):

        print(
            f"Batch {batch_id:3d} | "
            f"{source:8s} | "
            f"angle={str(angle):>6s} | "
            f"thr={str(threshold_before):>6s} | "
            f"alert={str(alert):>4s} | "
            f"KS={str(ks_statistic):>6s} | "
            f"p={str(ks_p_value):>12s} | "
            f"W={str(ks_window_length):>3s} | "
            f"drift={drift}"
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUT_FILE,
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Ground-truth drift batch:",
    GROUND_TRUTH_DRIFT_BATCH
)

print(
    "Detected drift batches:",
    detected_batches
)


# ------------------------------------------------------------
# First detection after real drift
# ------------------------------------------------------------

valid_detections = [
    batch
    for batch in detected_batches
    if batch >= GROUND_TRUTH_DRIFT_BATCH
]


if len(valid_detections) > 0:

    first_detection = valid_detections[0]

    delay = (
        first_detection
        - GROUND_TRUTH_DRIFT_BATCH
    )

    print(
        "\nFirst valid drift detection:",
        first_detection
    )

    print(
        "Detection delay:",
        delay,
        "batches"
    )

    print(
        "Detection delay in records:",
        delay * BATCH_SIZE
    )

else:

    print(
        "\nNo drift was detected "
        "after ground-truth boundary."
    )


# ------------------------------------------------------------
# False detections before drift
# ------------------------------------------------------------

false_detections = [
    batch
    for batch in detected_batches
    if batch < GROUND_TRUTH_DRIFT_BATCH
]


print(
    "\nFalse detections before drift:",
    false_detections
)


# ============================================================
# 10. ALERT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SAD ALERT DISTRIBUTION")
print("=" * 70)


before_alerts = results_df.loc[
    (
        results_df["batch_id"] >= 10
    )
    &
    (
        results_df["batch_id"]
        < GROUND_TRUTH_DRIFT_BATCH
    ),
    "alert"
]


after_alerts = results_df.loc[
    (
        results_df["batch_id"]
        >= GROUND_TRUTH_DRIFT_BATCH
    )
    &
    (
        results_df["batch_id"]
        < GROUND_TRUTH_DRIFT_BATCH + 20
    ),
    "alert"
]


print("\nBefore drift (batch 10-79):")

print(
    before_alerts.value_counts(
        dropna=False
    )
)


print("\nAfter drift (batch 80-99):")

print(
    after_alerts.value_counts(
        dropna=False
    )
)


# ============================================================
# 11. DEBUG TABLE AROUND DRIFT
# ============================================================

print("\n" + "=" * 70)
print("DEBUG AROUND DRIFT: BATCH 70-100")
print("=" * 70)


debug_cols = [

    "batch_id",

    "source",

    "angle",

    "threshold_used",

    "alert",

    "ks_statistic",

    "ks_p_value",

    "ks_window_length",

    "drift_detected",
]


print(

    results_df.loc[

        (
            results_df["batch_id"]
            >= 70
        )

        &

        (
            results_df["batch_id"]
            <= 100
        ),

        debug_cols

    ].to_string(
        index=False
    )

)


# ============================================================
# 12. LOWEST KSWIN P-VALUES
# ============================================================

print("\n" + "=" * 70)
print("LOWEST KSWIN P-VALUES")
print("=" * 70)


tested = results_df[
    results_df["ks_p_value"].notna()
].copy()


if len(tested) > 0:

    tested = tested.sort_values(
        "ks_p_value"
    )

    print(

        tested[
            [
                "batch_id",
                "source",
                "alert",
                "ks_statistic",
                "ks_p_value",
                "drift_detected",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )

    )

else:

    print(
        "KSWIN has not produced "
        "any p-value."
    )


# ============================================================
# 13. FINISH
# ============================================================

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUT_FILE)

print("\nDONE.")