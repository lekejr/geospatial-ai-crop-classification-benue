import rasterio
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

expected_files = [
    "benue_s1_land_prep_mar_apr.tif",
    "benue_s2_early_growth_may_jun.tif",
    "benue_s3_peak_growth_jun_sep.tif",
    "benue_s4_maturity_sep_oct.tif",
    "benue_s5_postharvest_dec_jan.tif",
]

for fname in expected_files:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue

    with rasterio.open(path) as src:
        data = src.read(1)  # first band (blue)
        total_pixels = data.size
        nodata_pixels = np.isnan(data).sum()
        pct_missing = 100 * nodata_pixels / total_pixels

        print(f"{fname}")
        print(f"    size on disk: {os.path.getsize(path) / 1e6:.1f} MB")
        print(f"    dimensions: {src.width} x {src.height} pixels, {src.count} bands")
        print(f"    % pixels missing (cloud-masked): {pct_missing:.1f}%")
        print(f"    band 1 (blue) min/max, ignoring gaps: {np.nanmin(data):.4f} / {np.nanmax(data):.4f}")
        print()