# Data Directory

## GEE Collection IDs

| Dataset | GEE Collection ID | Band | Resolution | Period |
|---------|------------------|------|------------|--------|
| MODIS Burned Area | `MODIS/061/MCD64A1` | `BurnDate` | 500 m | 2018–2024 |
| MODIS Land Cover | `MODIS/061/MCD12Q1` | `LC_Type1` | 500 m | 2018–2024 |
| MODIS Active Fire | `MODIS/061/MOD14A1` | `FireMask` | 1 km | 2018–2024 |
| MODIS NDVI | `MODIS/061/MOD13A3` | `1_km_monthly_NDVI` | 1 km | 2018–2024 |
| WorldPop Population | `WorldPop/GP/100m/pop` | `population` | 100 m | 2018–2020 |
| VIIRS Nighttime Lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | `avg_rad` | ~500 m | 2018–2024 |
| ERA5-Land Precipitation | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | `total_precipitation_sum` | 9 km | 2018–2024 |
| Sentinel-5P AAI | `COPERNICUS/S5P/OFFL/L3_AER_AI` | `absorbing_aerosol_index` | 7 km | 2018–2024 |
| Sentinel-5P CO | `COPERNICUS/S5P/OFFL/L3_CO` | `CO_column_number_density` | 7 km | 2018–2024 |
| Country Boundaries | `USDOS/LSIB_SIMPLE/2017` | — | Vector | Reference |

## Processed Data Files

| File | Description | Rows |
|------|-------------|------|
| `burned_fraction_grid_2018_2024.csv` | Annual burned fraction per 1°×1° cell | 2,100 |
| `socio_grid_2018_2024.csv` | Population density + nighttime lights per cell × year | 2,100 |
| `landcover_grid_2018_2024.csv` | Cropland / forest / savanna % per cell × year | 2,100 |
| `environment_grid_2018_2024.csv` | Dry-season NDVI + annual rainfall per cell × year | 2,100 |
| `city_atmospheric_2018_2024.csv` | Monthly S5P AAI, CO, fire count per city | 1,260 |
| `city_population_2020.csv` | Urban population per city (25 km buffer) | 15 |
| `shap_importance.csv` | SHAP feature importance from Random Forest | 8 |
| `cv_results.csv` | 5-fold spatial block CV results per block | 5 |

## Reproduction Notes

GEE extraction runtime: ~40–50 minutes total
- Section 3.1 (burned fraction): ~10 min
- Section 3.2 (socioeconomic + land cover + environment): ~15 min
- Section 3.3–3.5 (city atmospheric + population): ~15 min

Requires a valid GEE project ID. Replace `'ee-ujjwalkumarswainiirs1'` in Section 1 with your own project ID.
