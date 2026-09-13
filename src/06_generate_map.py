"""
06_generate_map.py

Purpose
-------
Apply the trained Logistic Regression model (selected over Random Forest
for its higher macro F1 on our small dataset — see script 05's output)
to every pixel in the study area, producing:
  - outputs/maps/classification_map.tif  (GeoTIFF, for GIS use)
  - outputs/maps/classification_map.png  (colored map, for the README)

Handling missing data honestly
----------------------------------
Some pixels have no valid value in one or more features (cloud gaps from
the original composites — up to ~10% in the worst season, per script 01's
sanity check). Rather than guessing a class for these pixels, they are
explicitly marked as "no data" in both outputs — shown as white/transparent
in the map, not silently assigned to whichever class happens to be most
common. This is the honest way to represent a genuine gap in the data.
"""

import rasterio
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

FEATURE_STACK_PATH = os.path.join(PROCESSED_DIR, "feature_stack.tif")
MODEL_PATH = os.path.join(PROCESSED_DIR, "model_logreg.joblib")
SCALER_PATH = os.path.join(PROCESSED_DIR, "feature_scaler.joblib")

FEATURE_ORDER = [
    "dryseason_blue", "dryseason_green", "dryseason_red", "dryseason_rededge",
    "dryseason_nir", "dryseason_swir1", "dryseason_swir2",
    "ndvi_s1_land_prep_mar_apr", "ndvi_s2_early_growth_may_jun",
    "ndvi_s3_peak_growth_jun_sep", "ndvi_s4_maturity_sep_oct",
    "ndvi_s5_postharvest_dec_jan", "ndvi_amplitude",
]

CLASS_TO_CODE = {
    "cropland": 1,
    "vegetation_forest": 2,
    "built_up": 3,
    "bare_land": 4,
    "water": 5,
}
CODE_TO_CLASS = {v: k for k, v in CLASS_TO_CODE.items()}
NODATA_CODE = 0

CLASS_COLORS = {
    "cropland": "#e8b923",       # gold
    "vegetation_forest": "#1a7a3c",  # dark green
    "built_up": "#c0392b",       # brick red
    "bare_land": "#b08968",      # tan
    "water": "#2980b9",          # blue
}

# ---------------------------------------------------------------------------
# 1. Load model, scaler, and feature stack
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Loaded trained Logistic Regression model and scaler")

with rasterio.open(FEATURE_STACK_PATH) as src:
    band_names = list(src.descriptions)
    profile = src.profile.copy()
    height, width = src.height, src.width

    # Read bands in the exact order the model was trained on
    band_indices = [band_names.index(name) + 1 for name in FEATURE_ORDER]
    stack = src.read(band_indices)  # shape (13, H, W)

print(f"Feature stack shape: {stack.shape}")

# ---------------------------------------------------------------------------
# 2. Reshape to (n_pixels, n_features), identify valid pixels
# ---------------------------------------------------------------------------
n_features = stack.shape[0]
flat = stack.reshape(n_features, -1).T  # shape (H*W, 13)

valid_mask = ~np.isnan(flat).any(axis=1)
n_valid = valid_mask.sum()
n_total = flat.shape[0]
print(f"Valid pixels: {n_valid} / {n_total} ({100 * n_valid / n_total:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Predict on valid pixels only
# ---------------------------------------------------------------------------
X_valid = flat[valid_mask]
X_valid_scaled = scaler.transform(X_valid)
predictions = model.predict(X_valid_scaled)  # class name strings

pred_codes = np.array([CLASS_TO_CODE[p] for p in predictions])

# ---------------------------------------------------------------------------
# 4. Reassemble into full grid, with NODATA_CODE for invalid pixels
# ---------------------------------------------------------------------------
full_pred = np.full(n_total, NODATA_CODE, dtype=np.uint8)
full_pred[valid_mask] = pred_codes
classification_map = full_pred.reshape(height, width)

# ---------------------------------------------------------------------------
# 5. Save as GeoTIFF
# ---------------------------------------------------------------------------
out_profile = profile.copy()
out_profile.update(count=1, dtype="uint8", nodata=NODATA_CODE)

tif_path = os.path.join(MAPS_DIR, "classification_map.tif")
with rasterio.open(tif_path, "w", **out_profile) as dst:
    dst.write(classification_map, 1)
    dst.write_colormap(1, {
        NODATA_CODE: (255, 255, 255, 0),
        **{code: tuple(int(CLASS_COLORS[name][i:i+2], 16) for i in (1, 3, 5)) + (255,)
           for name, code in CLASS_TO_CODE.items()}
    })

print(f"\nSaved GeoTIFF: {tif_path}")

# ---------------------------------------------------------------------------
# 6. Save colored PNG map
# ---------------------------------------------------------------------------
ordered_codes = [NODATA_CODE] + list(CLASS_TO_CODE.values())
ordered_colors = ["#ffffff"] + [CLASS_COLORS[CODE_TO_CLASS[c]] for c in CLASS_TO_CODE.values()]
cmap = ListedColormap(ordered_colors)

display_map = np.zeros_like(classification_map)
for i, code in enumerate(ordered_codes):
    display_map[classification_map == code] = i

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(display_map, cmap=cmap, vmin=0, vmax=len(ordered_codes) - 1)
ax.set_title("Agricultural Land-Cover Classification\nMakurdi/Guma, Benue State, Nigeria")
ax.axis("off")

legend_elements = [Patch(facecolor="white", edgecolor="gray", label="No data (cloud gap)")]
legend_elements += [Patch(facecolor=CLASS_COLORS[name], label=name.replace("_", " ").title())
                     for name in CLASS_TO_CODE.keys()]
ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0, -0.15), ncol=2, fontsize=9)

png_path = os.path.join(MAPS_DIR, "classification_map.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved PNG map: {png_path}")

# ---------------------------------------------------------------------------
# 7. Report area per class (agricultural interpretation)
# ---------------------------------------------------------------------------
pixel_area_km2 = (10 * 10) / 1_000_000  # 10m x 10m pixels, in km^2
print("\nEstimated land area by class:")
print(f"{'Class':<20}{'Pixels':>12}{'Area (km2)':>14}{'% of study area':>18}")
for name, code in CLASS_TO_CODE.items():
    count = (classification_map == code).sum()
    area = count * pixel_area_km2
    pct = 100 * count / n_total
    print(f"{name:<20}{count:>12}{area:>14.2f}{pct:>17.1f}%")

nodata_count = (classification_map == NODATA_CODE).sum()
print(f"{'no_data':<20}{nodata_count:>12}{nodata_count * pixel_area_km2:>14.2f}"
      f"{100 * nodata_count / n_total:>17.1f}%")