"""
03_generate_candidate_points.py

Purpose
-------
Generate a modest set of CANDIDATE training points, spread across 5
target classes (cropland, built-up, water, vegetation/forest, bare land),
using ESA WorldCover as a starting reference for where each class is
likely to be found.

IMPORTANT — these are candidates, not final ground truth
----------------------------------------------------------
ESA WorldCover is itself a machine-learning product (10m resolution),
and it can misclassify small, mixed smallholder farm plots — exactly the
kind of land this project cares about. Using it directly as training
labels would just mean inheriting someone else's model's mistakes.

So this script only proposes WHERE to look. The actual label for each
point still needs a quick human visual check (next script / next step)
before it's used for training. That verification step is what makes
these genuinely defensible training labels rather than borrowed ones.

Method
------
1. Load ESA WorldCover (10m, global) for the exact same study area used
   in script 01 (read from the saved study_area.geojson, so it lines up
   exactly with the downloaded Sentinel-2 data).
2. Reclassify WorldCover's 11 classes down to our 5 target classes:
     10 Tree cover, 20 Shrubland, 30 Grassland  -> vegetation_forest
     40 Cropland                                -> cropland
     50 Built-up                                -> built_up
     60 Bare/sparse vegetation                  -> bare_land
     80 Permanent water, 90 Herbaceous wetland  -> water
   (70 Snow/ice, 95 Mangroves, 100 Moss/lichen are not expected in this
   region and are excluded.)
3. Draw a stratified random sample (~20 points per class, so the rarer
   classes aren't drowned out) from this reclassified map.
4. For each sampled point, extract its 13 feature values from the local
   feature_stack.tif (the file built by script 02).
5. Save everything to a CSV with an empty "verified_label" column, ready
   for the manual visual-check step.

Sample size note
-----------------
~20 points/class (up to ~100 total) is intentionally modest — enough for
a baseline Logistic Regression / Random Forest with only 13 features,
without creating an unreasonable manual-verification workload. This
sample size is disclosed as a limitation in the final README.
"""

import ee
import json
import os
import csv
import numpy as np
import rasterio
from pyproj import Transformer

EE_PROJECT = "benue-crop-project"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

POINTS_PER_CLASS = 20
SEED = 42

CLASS_NAMES = {
    0: "vegetation_forest",
    1: "cropland",
    2: "built_up",
    3: "bare_land",
    4: "water",
}

# ---------------------------------------------------------------------------
# 1. Rebuild the exact same AOI used in script 01, from the saved geojson
# ---------------------------------------------------------------------------
aoi_path = os.path.join(RAW_DIR, "study_area.geojson")
with open(aoi_path) as f:
    aoi_geojson = json.load(f)

geom_dict = aoi_geojson["features"][0]["geometry"]
aoi = ee.Geometry(geom_dict)
print("Loaded study area AOI from study_area.geojson")

# ---------------------------------------------------------------------------
# 2. Load ESA WorldCover and reclassify to our 5 target classes
# ---------------------------------------------------------------------------
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

from_classes = [10, 20, 30, 40, 50, 60, 80, 90]
to_classes = [0, 0, 0, 1, 2, 3, 4, 4]  # see mapping in docstring above

label_img = worldcover.remap(from_classes, to_classes, defaultValue=-1).rename("label")
label_img = label_img.updateMask(label_img.neq(-1))
labelled_img = label_img.addBands(worldcover.rename("worldcover_raw"))

# ---------------------------------------------------------------------------
# 3. Stratified sample
# ---------------------------------------------------------------------------
sample = labelled_img.stratifiedSample(
    numPoints=POINTS_PER_CLASS,
    classBand="label",
    region=aoi,
    scale=10,
    seed=SEED,
    geometries=True,
)

sample_info = sample.getInfo()
features = sample_info["features"]
print(f"Sampled {len(features)} candidate points across {len(CLASS_NAMES)} target classes")

# ---------------------------------------------------------------------------
# 4. Extract feature values for each point from the local feature_stack.tif
# ---------------------------------------------------------------------------
feature_stack_path = os.path.join(PROCESSED_DIR, "feature_stack.tif")

rows_out = []
with rasterio.open(feature_stack_path) as src:
    band_names = src.descriptions
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    for i, feat in enumerate(features):
        lon, lat = feat["geometry"]["coordinates"]
        label_code = feat["properties"]["label"]
        worldcover_raw = feat["properties"]["worldcover_raw"]

        x, y = transformer.transform(lon, lat)
        try:
            row, col = src.index(x, y)
        except Exception:
            continue  # point somehow outside raster extent, skip

        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            continue

        window = rasterio.windows.Window(col, row, 1, 1)
        pixel_values = src.read(window=window)[:, 0, 0]  # shape (13,)

        record = {
            "point_id": i,
            "lon": lon,
            "lat": lat,
            "worldcover_raw_code": worldcover_raw,
            "worldcover_suggested_label": CLASS_NAMES[label_code],
            "verified_label": "",  # to be filled in manually, next step
            "google_maps_check_link": f"https://www.google.com/maps/@{lat},{lon},17z/data=!3m1!1e3",
        }
        for name, val in zip(band_names, pixel_values):
            record[name] = float(val) if not np.isnan(val) else ""

        rows_out.append(record)

print(f"Extracted feature values for {len(rows_out)} points")

# ---------------------------------------------------------------------------
# 5. Save to CSV
# ---------------------------------------------------------------------------
out_path = os.path.join(PROCESSED_DIR, "candidate_training_points.csv")
if rows_out:
    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

print(f"\nSaved: {out_path}")
print("\nWorldCover-suggested label counts:")
from collections import Counter
counts = Counter(r["worldcover_suggested_label"] for r in rows_out)
for name, count in counts.items():
    print(f"  {name}: {count}")