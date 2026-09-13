"""
make_verification_sheets.py

Purpose
-------
For the two classes most likely to be confused by ESA WorldCover
(cropland vs. vegetation_forest), generate zoomed-in true-color image
patches centered on each candidate point, using the PEAK GROWING SEASON
composite (Jun-Sep) rather than the dry-season one.

Why peak season, not dry season
---------------------------------
An earlier version of this script used the dry-season composite, but at
that time of year BOTH harvested cropland and senesced natural vegetation
look similarly brown, making them impossible to tell apart visually. At
peak growing season, actively growing cropland shows up as vivid green
with sharp, geometric plot boundaries, while natural vegetation shows a
continuous, irregular green canopy — a real, visually distinguishable
difference.

Why a global color stretch, not per-patch
--------------------------------------------
An earlier version of this script normalized each small 30x30-pixel
patch independently, which produced unnatural magenta/pink color casts
whenever a patch's local pixel statistics were skewed (e.g. by a shadow
or a stray cloud edge). This version computes ONE brightness stretch
from the whole image and applies it consistently to every patch, so
colors are visually meaningful and comparable across patches.

This targets only the 40 cropland/vegetation_forest points — the other
60 (water, built_up, bare_land) are visually distinctive enough that
WorldCover's label is accepted as-is, per the earlier discussion.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pyproj import Transformer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(PROCESSED_DIR, "candidate_training_points.csv")
PEAK_SEASON_PATH = os.path.join(RAW_DIR, "benue_s3_peak_growth_jun_sep.tif")

PATCH_HALF_SIZE = 15  # pixels each direction -> 30x30 px patch (~300x300m)
TARGET_CLASSES = ["cropland", "vegetation_forest"]

# Band order in the raw seasonal files (from script 01):
# 1:blue 2:green 3:red 4:rededge 5:nir 6:swir1 7:swir2 8:NDVI 9:NDWI
RED_BAND, GREEN_BAND, BLUE_BAND, NDVI_BAND = 3, 2, 1, 8

df = pd.read_csv(CSV_PATH)
target_df = df[df["worldcover_suggested_label"].isin(TARGET_CLASSES)].reset_index(drop=True)
print(f"Found {len(target_df)} points needing visual verification")

with rasterio.open(PEAK_SEASON_PATH) as src:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # ------------------------------------------------------------------
    # Compute ONE global brightness stretch from the whole image (using a
    # downsampled read to keep this fast), so every small patch below
    # uses the same color balance instead of each inventing its own —
    # that per-patch approach is what caused the magenta/pink color casts
    # in the previous version.
    # ------------------------------------------------------------------
    downsample_factor = 4
    out_shape = (src.height // downsample_factor, src.width // downsample_factor)
    r_full = src.read(RED_BAND, out_shape=out_shape)
    g_full = src.read(GREEN_BAND, out_shape=out_shape)
    b_full = src.read(BLUE_BAND, out_shape=out_shape)

    def global_stretch_params(band):
        valid = band[~np.isnan(band)]
        return np.percentile(valid, [2, 98])

    r_lo, r_hi = global_stretch_params(r_full)
    g_lo, g_hi = global_stretch_params(g_full)
    b_lo, b_hi = global_stretch_params(b_full)
    print(f"Global stretch — red: {r_lo:.3f}-{r_hi:.3f}, green: {g_lo:.3f}-{g_hi:.3f}, blue: {b_lo:.3f}-{b_hi:.3f}")

    def apply_stretch(band, lo, hi):
        return np.clip((band - lo) / (hi - lo + 1e-9), 0, 1)

    patches = []
    for _, row in target_df.iterrows():
        x, y = transformer.transform(row["lon"], row["lat"])
        center_row, center_col = src.index(x, y)

        row_start = max(0, center_row - PATCH_HALF_SIZE)
        row_stop = min(src.height, center_row + PATCH_HALF_SIZE)
        col_start = max(0, center_col - PATCH_HALF_SIZE)
        col_stop = min(src.width, center_col + PATCH_HALF_SIZE)

        window = rasterio.windows.Window(
            col_start, row_start, col_stop - col_start, row_stop - row_start
        )
        r = src.read(RED_BAND, window=window)
        g = src.read(GREEN_BAND, window=window)
        b = src.read(BLUE_BAND, window=window)
        ndvi_patch = src.read(NDVI_BAND, window=window)
        center_ndvi = float(np.nanmean(ndvi_patch)) if not np.all(np.isnan(ndvi_patch)) else float("nan")

        rgb_patch = np.dstack([
            apply_stretch(r, r_lo, r_hi),
            apply_stretch(g, g_lo, g_hi),
            apply_stretch(b, b_lo, b_hi),
        ])
        rgb_patch = np.nan_to_num(rgb_patch, nan=0.3)

        patches.append(
            {
                "point_id": int(row["point_id"]),
                "suggested": row["worldcover_suggested_label"],
                "patch": rgb_patch,
                "ndvi": center_ndvi,
            }
        )

# ---------------------------------------------------------------------------
# Build contact sheets: one image per class, grid layout
# ---------------------------------------------------------------------------
for cls in TARGET_CLASSES:
    cls_patches = [p for p in patches if p["suggested"] == cls]
    n = len(cls_patches)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.8))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")
        if idx < n:
            p = cls_patches[idx]
            ax.imshow(p["patch"])
            ax.set_title(f"ID {p['point_id']}\nNDVI={p['ndvi']:.2f}", fontsize=9)

    fig.suptitle(f"PEAK SEASON — candidates suggested as: {cls}", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"verify_{cls}_peakseason.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  ({n} patches)")

print("\nUpload both verify_cropland_peakseason.png and verify_vegetation_forest_peakseason.png here for review.")