# Satellite-Based Agricultural Land-Cover Classification in Benue State, Nigeria

A geospatial AI project applying Sentinel-2 satellite imagery and machine learning to map agricultural land cover in Nigeria's "Food Basket" state.

## Research Question

Can freely available Sentinel-2 imagery and machine learning classify agricultural land-cover classes (cropland, vegetation/forest, built-up, bare land, water) in a mixed urban-agricultural landscape in Benue State, Nigeria?

## Why Benue?

Benue State is widely known as Nigeria's "Food Basket" — one of the country's largest producers of yam, rice, soybean, and cassava. Understanding how much land is actually under cultivation, and how it's distributed relative to urban growth, has real relevance for food-security monitoring. This project treats that as a genuine applied problem, not an arbitrary choice of study area.

## Problem Statement

Nigeria lacks fine-grained, locally-verified agricultural land-cover data. Global products like ESA WorldCover exist, but — as this project demonstrates directly — they can systematically underrepresent smallholder cropland in favor of broader "vegetation" categories. This project builds a locally-trained classifier and, in the process, surfaces and corrects some of that underrepresentation using the region's own satellite data.

## Study Area

A 30 km × 30 km window centered on the boundary between **Makurdi** and **Guma** Local Government Areas, chosen because it captures:
- The Benue River and its floodplain (water, seasonally-flooded agriculture)
- Makurdi's urban core (built-up land)
- Surrounding farmland and savanna vegetation (the agricultural-urban gradient the project is built around)

The full Makurdi + Guma administrative boundaries were considered first, but produced Sentinel-2 exports over 5 GB — impractical for a laptop-based workflow — so the area was deliberately narrowed to this focused window. This is disclosed as a scoping decision, not a hidden shortcut.

## Data Sources

- **Sentinel-2 Level-2A (Surface Reflectance), Harmonized**, via Google Earth Engine — 5 seasonal cloud-masked composites across one agricultural year (2024–2025):
  - Mar–Apr (land preparation)
  - May–Jun (early growth)
  - Jun–Sep (peak growth — widened from the original Jul–Aug window after an initial run found only 1 usable cloud-free scene there)
  - Sep–Oct (maturity)
  - Dec–Jan (post-harvest, dry season)
- **ESA WorldCover v200** (10 m global land cover) — used only as a starting reference for where each class is likely to be found, never as final ground truth (see Training Data below)
- **FAO GAUL Level 2** administrative boundaries — used to define the study area precisely (note: this dataset spells Makurdi as "Markurdi," confirmed by listing all Benue LGA names directly from the data)

## Methodology

1. **Data acquisition**: Cloud-masked Sentinel-2 composites built server-side in Google Earth Engine (avoiding multi-GB local downloads), for 5 dates spanning the cropping calendar.
2. **Feature engineering**: 13 purposeful features — 7 dry-season spectral bands (most stable for water/built-up/bare-land discrimination) + NDVI at each of the 5 dates (the phenology signal) + NDVI amplitude (max minus min NDVI across the year — the single strongest predictor of cultivated vs. natural land, confirmed by feature importance below).
3. **Training data**: 125 candidate points sampled using ESA WorldCover as a starting reference, then corrected using a quantitative rule (see below) rather than treated as ground truth.
4. **Modeling**: Logistic Regression (baseline) and Random Forest, compared directly.
5. **Evaluation**: Spatially-stratified train/test split (within each class, split by longitude) to avoid spatial-autocorrelation leakage, with macro-averaged precision/recall/F1 to fairly weight the smaller classes.
6. **Prediction**: Full-image classification, with pixels lacking valid data in any feature explicitly marked "no data" rather than guessed.

## Training Data: Why Not Just Trust WorldCover

ESA WorldCover is itself a machine-learning product, and — as this project found directly — it can underrepresent small, mixed smallholder farmland in favor of broader natural-vegetation classes. Rather than manually eyeballing satellite image patches (which turned out to be unreliable: dry-season imagery makes harvested cropland and senesced natural vegetation look nearly identical, and even after switching to peak-season imagery, 30×30-pixel patches are hard to judge confidently by eye), this project used each point's own **NDVI time series** as a quantitative check:

- Points suggested as "vegetation_forest" but showing a high NDVI amplitude (>0.45 — a strong green-up/die-back swing inconsistent with stable natural cover) were reclassified as cropland.
- Points suggested as "cropland" but showing very low amplitude (<0.20) were reclassified as vegetation_forest.

Of 125 candidate points, **16 were reclassified this way — 15 of them from vegetation_forest to cropland.** This is a real, notable finding: it suggests WorldCover meaningfully undercounts cropland in this landscape, consistent with known limitations of global land-cover products in areas with small-plot smallholder farming.

**Important caveat**: this rule is a disclosed simplification. Guinea savanna grassland also dries out dramatically each dry season and can show a similarly high NDVI amplitude without being cultivated. Some points corrected to "cropland" may actually be seasonal grassland rather than true farmland. This could not be fully resolved without field verification data, which was outside the scope of a laptop-based, freely-available-data project.

Final training set: **112 points** (cropland 30, vegetation_forest 25, built_up 19, bare_land 19, water 19) after dropping points with missing feature values.

## Machine Learning Models & Evaluation

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.735 | 0.801 | 0.727 | **0.733** |
| Random Forest | 0.588 | 0.664 | 0.586 | 0.549 |

**Logistic Regression was selected for the final map**, based on its higher macro F1.

A genuinely interesting, disclosed finding: the simpler model outperformed the more flexible one. On a small dataset like this (112 points across 5 classes), Random Forest has more capacity to overfit incidental patterns, while Logistic Regression's simplicity acts as a form of regularization. This is a real bias-variance trade-off, not a fluke to be hidden.

### Feature Importance (Random Forest)

The 6 most important features, in order, were **all NDVI time-series values** (`ndvi_amplitude` ranked highest, followed by the individual seasonal NDVI readings). The 7 static dry-season spectral bands ranked lowest. This directly validates the project's core methodological premise: **phenology — how land changes across a year — is more informative for this task than any single spectral snapshot**, which is the reason this project used 5 dates instead of 1.

### Per-Class Performance (Logistic Regression, final model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bare_land | 1.00 | 0.50 | 0.67 |
| built_up | 1.00 | 0.67 | 0.80 |
| cropland | 0.78 | 0.78 | 0.78 |
| vegetation_forest | 0.60 | 0.86 | 0.71 |
| water | 0.62 | 0.83 | 0.71 |

## Results: Final Classification Map

See `outputs/maps/classification_map.png` (visual) and `classification_map.tif` (GeoTIFF, for GIS use).

Estimated land cover across the ~900 km² study area:

| Class | Area (km²) | % of study area |
|---|---|---|
| Cropland | 410.8 | 46.0% |
| Vegetation/forest | 284.0 | 31.8% |
| Built-up | 35.4 | 4.0% |
| Bare land | 14.1 | 1.6% |
| Water | 30.2 | 3.4% |
| No data (cloud gaps) | 118.1 | 13.2% |

## Agricultural Interpretation

Cropland dominates the study area at an estimated 46%, consistent with Benue's status as a major agricultural state. The corrections applied during training-data preparation (see above) suggest the true cropland proportion may be even higher than global land-cover products like WorldCover would indicate on their own — a meaningful finding in itself, separate from the classification map. Built-up area is concentrated as expected around Makurdi's urban core, with cropland and natural vegetation forming a mosaic around it, consistent with a peri-urban agricultural gradient.

## Limitations

- **Small training set** (112 points across 5 classes, as few as ~19 per class). Results should be read as a baseline demonstration, not a precise, generalizable accuracy figure.
- **No field-verified ground truth.** Training labels rely on ESA WorldCover plus a quantitative NDVI-based correction rule, not on-the-ground survey data. The correction rule cannot fully distinguish cultivated cropland from naturally high-amplitude seasonal grassland.
- **Cloud gaps**: up to ~10% of pixels in the cloudiest season (Sep–Oct) had no valid Sentinel-2 observation; these are honestly marked "no data" in the final map rather than guessed.
- **Reduced study area**: results describe a 30×30 km window, not the full Makurdi/Guma LGAs, for computational manageability.
- **Crop-species-level classification was not attempted.** Distinguishing yam, maize, rice, etc. would require reliable crop-specific training labels, which were not available through free data sources — attempting it would have meant fabricating labels, which this project deliberately avoided.

## Reproducibility

1. Install Python 3.10+ and the packages in `requirements.txt`.
2. Create a free Google Earth Engine account and Cloud project; enable the Earth Engine API and register the project for noncommercial use.
3. Run the scripts in `src/` in order:
   - `01_data_acquisition.py`
   - `02_preprocessing.py`
   - `03_generate_candidate_points.py`
   - `03b_add_vegetation_points.py`
   - `04_finalize_training_labels.py`
   - `05_train_models.py`
   - `06_generate_map.py`
4. Outputs land in `data/processed/` (intermediate data, trained models) and `outputs/` (figures, metrics, final maps).

## Technologies

Python, Google Earth Engine (`earthengine-api`, `geemap`), rasterio, GeoPandas, NumPy, pandas, scikit-learn, matplotlib, Shapely, pyproj.

## Future Improvements

- Field-verified ground truth (even a small number of GPS-tagged farm visits) would allow genuine crop-species classification and validate the NDVI-amplitude correction rule against reality.
- Extending the study area to a second, contrasting part of Benue (e.g. the more hilly Vandeikya/Gboko yam belt) would test whether the model generalizes beyond the Makurdi/Guma floodplain.
- A denser time series (monthly composites instead of 5 seasonal windows) could sharpen the phenology signal further, at the cost of more Earth Engine processing.