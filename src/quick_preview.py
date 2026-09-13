"""
quick_preview.py

Purpose
-------
Generate two quick PNG previews from the feature stack so you can visually
confirm the data looks sensible before investing time in training labels:

1. A true-color-ish preview (red/green/blue dry-season bands) — should
   look recognizably like a real satellite photo of a landscape.
2. An NDVI amplitude map — cropland should show up as visibly different
   (typically brighter/higher values) from permanent water or dense urban
   areas, since amplitude measures how much a pixel's greenness swings
   across the year.

These are diagnostic previews only, not the final polished map (that
comes later, after classification).
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

path = os.path.join(PROCESSED_DIR, "feature_stack.tif")

with rasterio.open(path) as src:
    band_names = src.descriptions
    print("Bands in file:", band_names)

    red = src.read(band_names.index("dryseason_red") + 1)
    green = src.read(band_names.index("dryseason_green") + 1)
    blue = src.read(band_names.index("dryseason_blue") + 1)
    ndvi_amp = src.read(band_names.index("ndvi_amplitude") + 1)

# ---------------------------------------------------------------------------
# True-color-ish preview
# ---------------------------------------------------------------------------
def normalize(band, low_pct=2, high_pct=98):
    valid = band[~np.isnan(band)]
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    return out

rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])
rgb = np.nan_to_num(rgb, nan=0.0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb)
ax.set_title("Dry-season true-color preview (Makurdi/Guma study area)")
ax.axis("off")
out1 = os.path.join(FIGURES_DIR, "preview_truecolor.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# NDVI amplitude preview
# Stretched to the actual 2nd-98th percentile of the data (not the
# theoretical 0-2 max, which real-world values rarely approach) so real
# contrast between land types is actually visible.
# ---------------------------------------------------------------------------
valid_amp = ndvi_amp[~np.isnan(ndvi_amp)]
vmin, vmax = np.percentile(valid_amp, [2, 98])
print(f"\nNDVI amplitude 2nd-98th percentile range: {vmin:.3f} to {vmax:.3f}")
print(f"NDVI amplitude mean: {np.nanmean(ndvi_amp):.3f}, median: {np.nanmedian(ndvi_amp):.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(ndvi_amp, cmap="RdYlGn", vmin=vmin, vmax=vmax)
ax.set_title("NDVI amplitude (max - min across the year), stretched to data range")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI amplitude")
out2 = os.path.join(FIGURES_DIR, "preview_ndvi_amplitude.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print("\nOpen these two PNG files in outputs/figures/ to take a look.")