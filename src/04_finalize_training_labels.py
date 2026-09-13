"""
04_finalize_training_labels.py

Purpose
-------
Produce the final, defensible training label for all 100 candidate
points, using a quantitative rule instead of manual visual inspection.

Why a quantitative rule, not eyeballing images
-------------------------------------------------
Earlier attempts to visually verify the 40 ambiguous cropland/
vegetation_forest points ran into real limits: dry-season imagery made
both classes look the same shade of brown, and even after switching to
peak-season imagery, small 30x30-pixel patches are genuinely hard to
judge reliably by eye. But we already have something more reliable: the
full NDVI time series (5 dates across the year) for every point, saved
in candidate_training_points.csv by script 03.

The rule
--------
Cropland has a distinctive NUMERIC signature: NDVI rises sharply during
the growing season then drops sharply after harvest (high amplitude).
Natural vegetation (forest/shrubland) stays comparatively stable
year-round (lower amplitude), since it isn't cleared and replanted.

For the 40 points suggested as cropland or vegetation_forest by
WorldCover, this script only OVERRIDES the suggestion when the NDVI
amplitude clearly contradicts it:
    - suggested "cropland" but amplitude < LOW_AMPLITUDE_THRESHOLD
      -> relabel as vegetation_forest (doesn't behave like a farmed field)
    - suggested "vegetation_forest" but amplitude > HIGH_AMPLITUDE_THRESHOLD
      -> relabel as cropland (too much seasonal swing to be stable cover)
    - otherwise -> keep WorldCover's original suggestion

The other 60 points (water, built_up, bare_land) are visually
unambiguous classes that WorldCover is reliable for, so their suggested
label is kept as-is.

This rule and its two thresholds are a disclosed simplification — noted
as a limitation in the README — not a claim of perfect ground truth.

Output
------
data/processed/training_data.csv — 100 rows (fewer if any had missing
feature values), each with a final_label column plus all 13 features,
ready for model training in script 05.
"""

import pandas as pd
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CSV_PATH = os.path.join(PROCESSED_DIR, "candidate_training_points.csv")

LOW_AMPLITUDE_THRESHOLD = 0.20
HIGH_AMPLITUDE_THRESHOLD = 0.45

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} candidate points")

AMBIGUOUS_CLASSES = ["cropland", "vegetation_forest"]


def resolve_label(row):
    suggested = row["worldcover_suggested_label"]
    if suggested not in AMBIGUOUS_CLASSES:
        return suggested  # water, built_up, bare_land: trust WorldCover as-is

    amplitude = row["ndvi_amplitude"]
    if pd.isna(amplitude):
        return suggested  # no data to check against, fall back to suggestion

    if suggested == "cropland" and amplitude < LOW_AMPLITUDE_THRESHOLD:
        return "vegetation_forest"
    if suggested == "vegetation_forest" and amplitude > HIGH_AMPLITUDE_THRESHOLD:
        return "cropland"
    return suggested


df["final_label"] = df.apply(resolve_label, axis=1)

# ---------------------------------------------------------------------------
# Report how many points were overridden, for transparency
# ---------------------------------------------------------------------------
overridden = df[df["final_label"] != df["worldcover_suggested_label"]]
print(f"\nPoints overridden by the NDVI-amplitude rule: {len(overridden)}")
if len(overridden) > 0:
    print(overridden[["point_id", "worldcover_suggested_label", "ndvi_amplitude", "final_label"]].to_string(index=False))

print("\nFinal label counts:")
print(df["final_label"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Drop any rows with missing feature values (can't train on NaN),
# and save the final training dataset
# ---------------------------------------------------------------------------
feature_cols = [
    "dryseason_blue", "dryseason_green", "dryseason_red", "dryseason_rededge",
    "dryseason_nir", "dryseason_swir1", "dryseason_swir2",
    "ndvi_s1_land_prep_mar_apr", "ndvi_s2_early_growth_may_jun",
    "ndvi_s3_peak_growth_jun_sep", "ndvi_s4_maturity_sep_oct",
    "ndvi_s5_postharvest_dec_jan", "ndvi_amplitude",
]

before = len(df)
df_clean = df.dropna(subset=feature_cols)
after = len(df_clean)
if before != after:
    print(f"\nDropped {before - after} points with missing feature values")

out_cols = ["point_id", "lon", "lat", "final_label"] + feature_cols
out_path = os.path.join(PROCESSED_DIR, "training_data.csv")
df_clean[out_cols].to_csv(out_path, index=False)

print(f"\nSaved final training dataset: {out_path}")
print(f"Total training points: {len(df_clean)}")