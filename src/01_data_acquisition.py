"""
01_data_acquisition.py

Purpose
-------
Define the study area (Makurdi + Guma LGAs, Benue State, Nigeria) using
official administrative boundaries, then pull and cloud-mask five
Sentinel-2 seasonal composites spanning one agricultural year.

Why 5 dates, not 1
-------------------
A single snapshot cannot separate cropland from natural vegetation, because
at any one moment both can look "green." Cropland has a distinctive
trajectory: bare/prepped soil -> rapid green-up -> peak canopy ->
senescence/harvest -> bare again. Sampling across the year lets later
scripts build NDVI time-series features that capture this pattern, which
is the core signal that makes cropland separable from forest/shrubland.

Dates chosen based on Nigeria's rain-fed cropping calendar for the Guinea
Savanna belt (which Benue sits in):
    - Mar-Apr : land preparation / early planting (mostly bare/dry)
    - May-Jun : early rains, planting underway (green-up starting)
    - Jul-Aug : peak vegetative growth (peak greenness)
    - Sep-Oct : maturity / early harvest (still green but senescing)
    - Dec-Jan : dry season, post-harvest (bare/fallow, lowest cloud cover)
These windows are wide (2 months) specifically to guarantee enough
cloud-free Sentinel-2 scenes for a clean median composite, since Benue's
wet season (Apr-Oct) has heavy cloud cover.

Why Earth Engine
-----------------
Avoids downloading raw Sentinel-2 scenes (multiple GB) to a normal laptop.
All cloud masking and compositing happens server-side; only the small
final composite is exported.

Study area sizing
-----------------
The full Makurdi + Guma LGA polygons proved too large for a laptop-friendly
workflow (5+ GB across 5 composites). Instead we use a 30 km x 30 km window
centered on the boundary between the two LGAs, which still captures the
Benue River, Makurdi's urban core, and surrounding floodplain farmland —
the same agricultural-urban gradient the project is built around — at a
data volume (roughly 1-2 GB total) that's actually workable. This is a
deliberate scoping decision, disclosed here and in the README, not a
shortcut: it trades rural extent we weren't using for manageability.

Output
------
- data/raw/study_area.geojson         : AOI boundary (the 30x30km window)
- 5x GeoTIFF composites exported to your Google Drive (folder
  "benue_crop_classification"), one per season window, each containing:
    Blue, Green, Red, Red Edge (B5), NIR, SWIR1, SWIR2, NDVI, NDWI
  at 10 m resolution, clipped to the AOI.

Note on exports: Earth Engine exports to Drive (not directly to disk),
because large raster exports need Earth Engine's own export pipeline.
After the export tasks finish (check https://code.earthengine.google.com/tasks),
download the GeoTIFFs from Drive into data/raw/ before running script 02.
"""

import ee
import geemap
import json
import os

# ---------------------------------------------------------------------------
# 0. Authenticate & initialize
# ---------------------------------------------------------------------------
EE_PROJECT = "benue-crop-project"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Study area: Makurdi + Guma LGAs, Benue State, Nigeria
#    Using FAO GAUL Level 2 administrative boundaries — an authoritative,
#    citable source — rather than a hand-drawn/hand-typed bounding box.
#    NOTE: this dataset spells Makurdi as "Markurdi" — confirmed by
#    listing all Benue LGA names directly from the data.
# ---------------------------------------------------------------------------
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")

benue_lgas = gaul.filter(
    ee.Filter.And(
        ee.Filter.eq("ADM0_NAME", "Nigeria"),
        ee.Filter.eq("ADM1_NAME", "Benue"),
        ee.Filter.inList("ADM2_NAME", ["Markurdi", "Guma"]),
    )
)

n_features = benue_lgas.size().getInfo()
print(f"LGA boundaries found: {n_features} (expected 2: Markurdi, Guma)")
if n_features != 2:
    names = benue_lgas.aggregate_array("ADM2_NAME").getInfo()
    print(f"WARNING: check ADM2_NAME spelling in GAUL. Found instead: {names}")

full_lga_area = benue_lgas.union().first().geometry()

# Narrow to a 30km x 30km window centered on the Markurdi/Guma boundary —
# see "Study area sizing" note at the top of this file for why.
HALF_WIDTH_M = 15000  # 15 km each side -> 30 km x 30 km window
center_point = full_lga_area.centroid(maxError=100)
study_area = center_point.buffer(HALF_WIDTH_M).bounds()

# Save AOI locally for reuse in later scripts / QGIS inspection
aoi_geojson = geemap.ee_to_geojson(ee.FeatureCollection(ee.Feature(study_area)))
with open(os.path.join(OUTPUT_DIR, "study_area.geojson"), "w") as f:
    json.dump(aoi_geojson, f)
print("Saved study_area.geojson")

# ---------------------------------------------------------------------------
# 2. Cloud masking function (Sentinel-2 Surface Reflectance Harmonized + s2cloudless)
# ---------------------------------------------------------------------------
CLOUD_FILTER = 60           # max scene-level cloud % before it's even considered
CLD_PRB_THRESH = 40         # per-pixel cloud probability threshold


def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
    )
    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2_sr,
            secondary=s2_cloudless,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
    )


def mask_clouds(img):
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")
    return img.updateMask(is_cloud.Not()).divide(10000).copyProperties(img, ["system:time_start"])


# ---------------------------------------------------------------------------
# 3. Build one cloud-masked median composite per season window
# ---------------------------------------------------------------------------
SEASON_WINDOWS = {
    "s1_land_prep_mar_apr": ("2024-03-01", "2024-04-30"),
    "s2_early_growth_may_jun": ("2024-05-01", "2024-06-30"),
    "s3_peak_growth_jun_sep": ("2024-06-15", "2024-09-15"),
    "s4_maturity_sep_oct": ("2024-09-01", "2024-10-31"),
    "s5_postharvest_dec_jan": ("2024-12-01", "2025-01-31"),
}

BANDS = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]  # Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2
BAND_NAMES = ["blue", "green", "red", "rededge", "nir", "swir1", "swir2"]


def build_composite(aoi, start_date, end_date):
    col = get_s2_sr_cld_col(aoi, start_date, end_date).map(mask_clouds)
    n = col.size().getInfo()
    print(f"    scenes available after cloud filter: {n}")
    composite = col.select(BANDS, BAND_NAMES).median().clip(aoi)

    ndvi = composite.normalizedDifference(["nir", "red"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["green", "nir"]).rename("NDWI")

    return composite.addBands([ndvi, ndwi])


tasks = []
for label, (start, end) in SEASON_WINDOWS.items():
    print(f"Building composite: {label} ({start} to {end})")
    comp = build_composite(study_area, start, end)

    task = ee.batch.Export.image.toDrive(
        image=comp,
        description=f"benue_{label}",
        folder="benue_crop_classification",
        fileNamePrefix=f"benue_{label}",
        region=study_area,
        scale=10,
        crs="EPSG:32632",  # UTM zone 32N, appropriate for this part of Nigeria
        maxPixels=1e10,
    )
    task.start()
    tasks.append(task)
    print(f"    export task started: benue_{label}")

print("\nAll 5 export tasks submitted.")
print("Monitor progress at https://code.earthengine.google.com/tasks")
print("Once complete, download the GeoTIFFs from Google Drive into data/raw/")