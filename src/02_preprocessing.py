"""
02_preprocessing.py

Purpose
-------
Combine the 5 seasonal Sentinel-2 composites (in data/raw/) into a single,
purposeful feature stack for classification. Rather than using all 45
bands (5 dates x 9 bands each) — which would be redundant and hard to
interpret — this script builds 13 features chosen for a specific reason
each:

1. Seven spectral bands (blue, green, red, red edge, NIR, SWIR1, SWIR2)
   from the DRY SEASON composite (Dec-Jan). Dry-season spectral values are
   the most stable and reliable for telling apart water, built-up areas,
   and bare land, since there's no crop canopy confusing the signal.

2. NDVI at each of the 5 dates (5 features). This is the actual phenology
   signal — how green each pixel is at each point in the year.

3. NDVI amplitude = (max NDVI across the year) - (min NDVI across the year),
   per pixel (1 feature). This is the single feature that most directly
   captures "greened up then died back," which is the signature of
   cultivated cropland, as opposed to permanent forest (stays green
   year-round, low amplitude) or bare/built-up/water (stays low
   year-round, low amplitude).

Total: 7 + 5 + 1 = 13 features.

Handling missing data
----------------------
Each composite has some cloud-masked gaps (0.2% to 10%, per script 01's
sanity check). Where a pixel is missing in any input band, that pixel's
NDVI-amplitude and dry-season-spectral features become NaN too. These
pixels are NOT dropped here — they're carried through as NaN and will be
excluded only where they'd break a specific step (training in script 04,
final prediction in script 07 will show them as "no data" rather than a
guessed class, which is the honest way to handle a real gap in the data).

Output
------
data/processed/feature_stack.tif — 13-band GeoTIFF, same grid/CRS/extent
as the input composites, with descriptive band names.
"""

import rasterio
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEASON_FILES = {
    "s1_land_prep_mar_apr": "benue_s1_land_prep_mar_apr.tif",
    "s2_early_growth_may_jun": "benue_s2_early_growth_may_jun.tif",
    "s3_peak_growth_jun_sep": "benue_s3_peak_growth_jun_sep.tif",
    "s4_maturity_sep_oct": "benue_s4_maturity_sep_oct.tif",
    "s5_postharvest_dec_jan": "benue_s5_postharvest_dec_jan.tif",
}

# Band order in each input file, as written by script 01:
# 1:blue 2:green 3:red 4:rededge 5:nir 6:swir1 7:swir2 8:NDVI 9:NDWI
NDVI_BAND_IDX = 8
SPECTRAL_BAND_IDXS = [1, 2, 3, 4, 5, 6, 7]
SPECTRAL_BAND_NAMES = ["blue", "green", "red", "rededge", "nir", "swir1", "swir2"]
DRY_SEASON_KEY = "s5_postharvest_dec_jan"  # baseline for spectral bands

# ---------------------------------------------------------------------------
# 1. Load NDVI from all 5 dates, and spectral bands from the dry season
# ---------------------------------------------------------------------------
ndvi_stack = []
ndvi_order = []
ref_profile = None
ref_shape = None
dry_spectral = None

for key, fname in SEASON_FILES.items():
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")

    with rasterio.open(path) as src:
        if ref_profile is None:
            ref_profile = src.profile.copy()
            ref_shape = (src.height, src.width)
        else:
            if (src.height, src.width) != ref_shape:
                raise ValueError(
                    f"{fname} has shape {(src.height, src.width)}, "
                    f"expected {ref_shape}. All composites must share the same grid."
                )

        ndvi = src.read(NDVI_BAND_IDX)
        ndvi_stack.append(ndvi)
        ndvi_order.append(key)
        print(f"Loaded NDVI from {fname}")

        if key == DRY_SEASON_KEY:
            dry_spectral = src.read(SPECTRAL_BAND_IDXS)  # shape (7, H, W)
            print(f"Loaded 7 spectral bands from {fname} (dry-season baseline)")

if dry_spectral is None:
    raise RuntimeError(f"Could not find dry-season file for key {DRY_SEASON_KEY}")

ndvi_stack = np.stack(ndvi_stack, axis=0)  # shape (5, H, W)

# ---------------------------------------------------------------------------
# 2. Compute NDVI amplitude (per-pixel max - min across the 5 dates)
#    Using nanmax/nanmin so a single missing date doesn't wipe out the
#    whole pixel; a pixel only ends up NaN here if ALL 5 dates are missing.
# ---------------------------------------------------------------------------
with np.errstate(invalid="ignore"):
    ndvi_max = np.nanmax(ndvi_stack, axis=0)
    ndvi_min = np.nanmin(ndvi_stack, axis=0)
ndvi_amplitude = ndvi_max - ndvi_min

pct_all_missing = 100 * np.isnan(ndvi_amplitude).sum() / ndvi_amplitude.size
print(f"\nNDVI amplitude computed. Pixels missing in ALL 5 dates: {pct_all_missing:.2f}%")

# ---------------------------------------------------------------------------
# 3. Stack everything into the final 13-band feature array
# ---------------------------------------------------------------------------
feature_bands = []
feature_names = []

for i, name in enumerate(SPECTRAL_BAND_NAMES):
    feature_bands.append(dry_spectral[i])
    feature_names.append(f"dryseason_{name}")

for i, key in enumerate(ndvi_order):
    feature_bands.append(ndvi_stack[i])
    feature_names.append(f"ndvi_{key}")

feature_bands.append(ndvi_amplitude)
feature_names.append("ndvi_amplitude")

feature_array = np.stack(feature_bands, axis=0).astype("float32")  # (13, H, W)

print(f"\nFinal feature stack shape: {feature_array.shape} (bands, height, width)")
print("Feature order:")
for i, name in enumerate(feature_names, start=1):
    print(f"  {i}. {name}")

# ---------------------------------------------------------------------------
# 4. Write out as a single GeoTIFF, preserving the original grid/CRS
# ---------------------------------------------------------------------------
out_profile = ref_profile.copy()
out_profile.update(
    count=feature_array.shape[0],
    dtype="float32",
    nodata=np.nan,
)

out_path = os.path.join(PROCESSED_DIR, "feature_stack.tif")
with rasterio.open(out_path, "w", **out_profile) as dst:
    for i in range(feature_array.shape[0]):
        dst.write(feature_array[i], i + 1)
        dst.set_band_description(i + 1, feature_names[i])

print(f"\nSaved feature stack to: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")