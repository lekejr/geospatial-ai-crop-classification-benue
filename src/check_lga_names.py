import ee

EE_PROJECT = "benue-crop-project"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")

benue_all = gaul.filter(
    ee.Filter.And(
        ee.Filter.eq("ADM0_NAME", "Nigeria"),
        ee.Filter.eq("ADM1_NAME", "Benue"),
    )
)

names = benue_all.aggregate_array("ADM2_NAME").getInfo()
print("All Benue LGA names in GAUL dataset:")
for n in sorted(names):
    print(f"  - {n}")