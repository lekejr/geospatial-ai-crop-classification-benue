"""
03b_add_vegetation_points.py

Purpose
-------
The initial 100-point sample only produced 9 usable vegetation_forest
training points after cleaning — too few for either model to learn that
class (both scored 0 recall on it). Since verification is now handled by
the automatic NDVI-amplitude rule (no manual image inspection needed),
getting more candidate points is cheap. This script samples ~25 more
vegetation_forest-only points and appends them to the existing
candidate_training_points.csv.
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

ADDITIONAL_POINTS = 25
SEED = 123  # different from script 03's seed (42), to get different points

# ---------------------------------------------------------------------------
# 1. Rebuild the AOI
# ---------------------------------------------------------------------------
aoi_path = os.path.join(RAW_DIR, "study_area.geojson")
with open(aoi_path) as f:
    aoi_geojson = json.load(f)
geom_dict = aoi_geojson["features"][0]["geometry"]
aoi = ee.Geometry(geom_dict)

# ---------------------------------------------------------------------------
# 2. Build a mask of ONLY the vegetation_forest classes (WorldCover 10/20/30)
# ---------------------------------------------------------------------------
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
veg_mask = worldcover.remap([10, 20, 30], [1, 1, 1], defaultValue=0)
veg_only = worldcover.updateMask(veg_mask)

sample = veg_only.sample(
    region=aoi,
    scale=10,
    numPixels=ADDITIONAL_POINTS * 4,  # oversample, then trim, since sample() isn't exact
    seed=SEED,
    geometries=True,
    dropNulls=True,
)

sample_info = sample.getInfo()
features = sample_info["features"][:ADDITIONAL_POINTS]
print(f"Sampled {len(features)} additional vegetation_forest points")

# ---------------------------------------------------------------------------
# 3. Extract feature values from the local feature_stack.tif
# ---------------------------------------------------------------------------
feature_stack_path = os.path.join(PROCESSED_DIR, "feature_stack.tif")

# Find the current max point_id so new points don't collide with existing ones
existing_csv_path = os.path.join(PROCESSED_DIR, "candidate_training_points.csv")
with open(existing_csv_path) as f:
    reader = csv.DictReader(f)
    existing_rows = list(reader)
    fieldnames = reader.fieldnames
max_id = max(int(r["point_id"]) for r in existing_rows)

new_rows = []
with rasterio.open(feature_stack_path) as src:
    band_names = src.descriptions
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    for i, feat in enumerate(features):
        lon, lat = feat["geometry"]["coordinates"]
        worldcover_raw = feat["properties"]["Map"]

        x, y = transformer.transform(lon, lat)
        try:
            row, col = src.index(x, y)
        except Exception:
            continue
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            continue

        window = rasterio.windows.Window(col, row, 1, 1)
        pixel_values = src.read(window=window)[:, 0, 0]

        record = {
            "point_id": max_id + 1 + i,
            "lon": lon,
            "lat": lat,
            "worldcover_raw_code": worldcover_raw,
            "worldcover_suggested_label": "vegetation_forest",
            "verified_label": "",
            "google_maps_check_link": f"https://www.google.com/maps/@{lat},{lon},17z/data=!3m1!1e3",
        }
        for name, val in zip(band_names, pixel_values):
            record[name] = float(val) if not np.isnan(val) else ""

        new_rows.append(record)

print(f"Extracted feature values for {len(new_rows)} new points")

# ---------------------------------------------------------------------------
# 4. Append to the existing CSV
# ---------------------------------------------------------------------------
with open(existing_csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(new_rows)

print(f"\nAppended {len(new_rows)} rows to: {existing_csv_path}")
print(f"Total candidate points now: {len(existing_rows) + len(new_rows)}")