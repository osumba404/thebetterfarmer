# The Better Farmer

**The Better Farmer: A Data-Driven Crop Suitability and Fertilizer Recommendation System Using Soil and Climate Data**

The Better Farmer is a machine-learning decision-support prototype for selected agricultural locations in Kenya. A user supplies a farm location. The system reads mapped soil properties, historical climate, crop-suitability layers, and crop–nutrient response functions, then ranks crops and estimates fertilizer nutrients and, where composition data exist, product quantities.

The planning horizon is 12–24 months. That horizon is handled with historical seasonal patterns, seasonal outlooks, and stated climate risk. The system does not claim to predict exact weather one or two years ahead.

Fertilizer recommendation systems already exist, including Kenya-specific tools from OFRA and newer KALRO–CGIAR work. This project does not claim that combining soil, weather, and crop requirements is new. The research question is narrower:

> How effectively can machine learning models rank suitable crops and estimate fertilizer types and application rates from soil nutrient levels, crop characteristics, and weather conditions in selected agricultural regions of Kenya?

The intended contribution is a reproducible prototype that compares candidate models with an agronomic baseline, predicts nutrient rates as well as product type where the data allow it, and shows the factors and uncertainty behind each result.

This repository currently records the project design, including the concept-paper draft in [Chapter One](#14-chapter-one-introduction). The architecture is technically plausible. The dataset-level compatibility test is still pending: the core sources have not yet been downloaded, joined on Kenyan coordinates, and checked for overlap, missing values, and usable training labels.

Machine learning is the core of the system. It is applied where each dataset can support a supervised or time-series task. Soil estimates from iSDAsoil are used as features. Crop suitability and nutrient response are separate models. Algorithm choice waits until the training tables exist and the candidate models have been compared.

The first prototype should cover one or two Kenyan zones and a short crop list, not the whole country. Candidate zones discussed so far are Nakuru, Uasin Gishu, and Trans Nzoia. Candidate crops are maize, beans, and sorghum, kept only where both GAEZ layers and OFRA records exist.

## Contents

1. [What the system is asked to produce](#1-what-the-system-is-asked-to-produce)
2. [Design principles](#2-design-principles)
3. [Data sources](#3-data-sources)
4. [Dataset compatibility](#4-dataset-compatibility)
5. [Integration architecture](#5-integration-architecture)
6. [Recommended data model](#6-recommended-data-model)
7. [What can actually be predicted](#7-what-can-actually-be-predicted)
8. [Machine learning architecture](#8-machine-learning-architecture)
9. [Methodology statement](#9-methodology-statement)
10. [Validation limits](#10-validation-limits)
11. [Work remaining before implementation](#11-work-remaining-before-implementation)
12. [Prototype settings to lock](#12-prototype-settings-to-lock)
13. [Related systems](#13-related-systems)
14. [Chapter One: Introduction](#14-chapter-one-introduction)
15. [References and access links](#15-references-and-access-links)

## 1. What the system is asked to produce

| Desired output | Feasibility | Condition |
| --- | --- | --- |
| Soil characteristics at a location | High | iSDAsoil mapped estimates are available for the point |
| Suitable crops for a location | High | Matching FAO GAEZ v5 crop layers are available |
| Seasonal climate characteristics | High | NASA POWER historical records are available |
| Exact weather 12–24 months ahead | Not reliable | Replace with climate-risk scenarios built from historical seasons |
| Optimal nutrient application rate | Conditional | An applicable OFRA response function exists, and economic assumptions are stated |
| Commercial fertilizer type and quantity | Conditional | Nutrient requirements can be converted with a fertilizer composition table and application constraints |
| Actual farm yield | Not yet established | Needs independent yield observations for validation |

Compatibility ratings in this document describe whether a source fits the proposed architecture. They are not a claim of verified predictive accuracy.

## 2. Design principles

1. **Coordinates are the join key.** Latitude and longitude connect soil, climate, suitability, and the farm record. OFRA is joined by crop and response region, not by dropping every response function onto every soil pixel.
2. **Keep each source’s native resolution.** A 30 m soil estimate and a NASA POWER grid cell describe different areas. Both are stored at the farm coordinate, with the source resolution recorded. Matching coordinates does not make the coarser layers as precise as the soil layer.
3. **Keep each source’s time period.** Soil maps, historical climate, GAEZ climate baselines, and OFRA field trials are not simultaneous observations. Every extracted value keeps its reference period.
4. **Separate the prediction targets.** GAEZ estimates crop suitability. OFRA estimates how yield responds to nutrients. Those are two models, then one recommendation engine.
5. **Use machine learning where the labels support it.** iSDAsoil is already a modelled soil product, so it enters as features. Crop ranking and nutrient response are the supervised tasks. Seasonal climate is a time-series task. Fertilizer product quantities are a conversion and optimization step after the nutrient estimate.
6. **Choose algorithms after the data check.** Random Forest and XGBoost are the candidate models. The winner is the model that performs better on geographically separated test data. SARIMA is the candidate for seasonal climate, compared with XGBoost using lag features.

The main scientific risk is false precision. Retrieving soil at 30 m does not mean the climate summary or the fertilizer rate has 30 m accuracy.

## 3. Data sources

Four sources are the minimum stack. They are used together. They are not downloaded and stacked into one CSV for a single model. Seasonal outlooks, fertilizer composition, prices, and validation trials are listed in [section 3.6](#36-supporting-sources).

| Dataset | Role in The Better Farmer | Format | Coverage | Fit |
| --- | --- | --- | --- | --- |
| [iSDAsoil](https://www.isda-africa.com/isdasoil/) | Soil features at the farm point | API JSON; raster layers also published | Africa, mostly 30 m | High |
| [OFRA](https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1) | Nutrient-response functions | Excel (`.xlsx`) | Georeferenced sites and regions in tropical Africa | Conditional |
| [FAO GAEZ v5](https://data.apps.fao.org/gaez/) | Crop suitability reference | Geospatial raster layers | Global, varying resolution (suitability index about 1 km) | High |
| [NASA POWER](https://power.larc.nasa.gov/) | Historical climate | JSON, CSV, and other scientific formats | Global, gridded | High |

### 3.1 iSDAsoil — soil properties

**Use:** primary soil source. For a farm coordinate, retrieve estimated soil properties and store the depth, units, and uncertainty with each value.

iSDAsoil is a field-level soil map for Africa. Predictions were produced with machine learning from more than 130,000 soil samples, drawing on the African Soil Information Service (AfSIS) and other soil data. Most layers are 30 m. Coverage is agricultural land across Africa; deserts and water bodies are generally excluded. Access is free and requires an account and an access token.

| | |
| --- | --- |
| Product page | https://www.isda-africa.com/isdasoil/ |
| Developer portal | https://www.isda-africa.com/isdasoil/developer/ |
| Property catalogue and FAQ | https://www.isda-africa.com/isdasoil/faq/ |
| API registration | https://www.isda-africa.com/api/registration/ |
| API host | https://api.isda-africa.com |
| Notebook tutorials | https://github.com/iSDA-Africa/isdasoil-tutorial |
| Contact | info@isda-africa.com |

**Access pattern**

1. Create a username and password at the registration page. Read the terms of use on the developer portal before use. API access is free under the iSDA licence.
2. Obtain a bearer token from `POST https://api.isda-africa.com/login`. Tokens expire after 60 minutes.
3. Inspect variables, units, and depths with `GET https://api.isda-africa.com/isdasoil/v2/layers`.
4. Request a property at a coordinate. Latitude and longitude are required. `property` and `depth` are optional; setting them avoids downloading every layer.

Documented example for pH at 0–20 cm at a Kenyan location used in the iSDAsoil docs (latitude −0.7196, longitude 35.2400):

```text
GET https://api.isda-africa.com/isdasoil/v2/soilproperty?lat=-0.7196&lon=35.2400&property=ph&depth=0-20
Authorization: Bearer <access_token>
```

The interactive API documentation is linked from the [developer portal](https://www.isda-africa.com/isdasoil/developer/). Bulk raster downloads are described from the FAQ (iSDAsoil on AWS). The point API is the intended access method for farm coordinates.

**Properties relevant to the recommendation engine**

Depths below are the API depths published on the [iSDAsoil FAQ](https://www.isda-africa.com/isdasoil/faq/). The prototype should standardise on **0–20 cm** unless a crop-specific reason requires 20–50 cm. Uncertainty is returned for the numerical predictions at 50%, 68% (one standard deviation), and 90% intervals. Several FAQ accuracy figures are marked with an asterisk; read that note on the FAQ before quoting R² or RMSE.

| Property (API concept) | Unit | Depths (cm) | Resolution | Theme | FAQ R² |
| --- | --- | --- | --- | --- | --- |
| pH (1:1 soil–water) | — | 0–20, 20–50 | 30 m | Chemical | 0.818 |
| Organic carbon | g/kg | 0–20, 20–50 | 30 m | Chemical | 0.791* |
| Nitrogen, total | g/kg | 0–20, 20–50 | 30 m | Chemical | 0.732* |
| Phosphorus, extractable (Mehlich-3) | ppm | 0–20, 20–50 | 30 m | Nutrients | 0.486* |
| Potassium, extractable (Mehlich-3) | ppm | 0–20, 20–50 | 30 m | Nutrients | 0.773* |
| Effective CEC (Mehlich-3) | cmol(+)/kg | 0–20, 20–50 | 30 m | Chemical | 0.754* |

Phosphorus is the weakest of these layers on the published R². Fertilizer models that depend on soil P should carry that uncertainty rather than treat the 30 m P value as a precise soil test.

Other documented layers that may help later, and are not required for the first training table, include texture (sand, silt, clay), bulk density, stone content, slope, depth to bedrock, and extractable calcium, magnesium, sulphur, and zinc. Confirm the exact `property` identifiers from `/isdasoil/v2/layers` before coding the client. The FAQ lists third-party agronomy layers (land cover and cropland, 2015–2019) at 100 m and without uncertainty; those are separate from the 30 m soil predictions.

**Compatibility decision:** use iSDAsoil as features. Do not train a second soil model on top of it. Store depth, units, and uncertainty beside every measurement.

### 3.2 OFRA — nutrient response

**Use:** a separate fertilizer component that selects crop–nutrient response functions by crop and region, then supports a regression model and a constrained optimizer. OFRA records are not merged onto every iSDAsoil coordinate.

OFRA (Optimizing / Optimized Fertilizer Recommendations in Africa) compiled georeferenced crop–nutrient response functions from field research in sub-Saharan Africa. With support from the University of Nebraska–Lincoln, and implementation linked to the AGRA Soil Health Program and CABI in Nairobi, the project established response functions for important annual food crops across 67 agro-ecological zones in 13 countries. Representative functions were built from an “inference space” of trials grown under similar conditions. Those functions feed Excel Fertilizer Optimization Tools, which use linear optimization against crop choice, land allocation, fertilizer prices, grain values, and the farmer’s fertilizer budget.

The University of Nebraska reports a dataset of more than 6,000 crop–nutrient response functions with citations. IFDC describes the response-function dataset as more than 6,200 functions. The Dryad abstract describes selection from a database of about 5,500 functions in an earlier maize-focused comparison. Treat the row count in the 2023 workbook, after inspection, as the figure this project will use.

| | |
| --- | --- |
| Dataset (Dryad) | https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1 |
| DOI | https://doi.org/10.5061/dryad.tt6h5h1 |
| OFRA project page (UNL) | https://agronomy.unl.edu/ofra/ |
| OFRA tools and resources (IFDC) | https://ifdc.org/resources/ofra-analysis-tools/ |
| Inference tool referred to in the Dryad abstract | https://agronomy.unl.edu/ofra/ |

**Workbook versions on the Dryad record** (listing verified; sheet-level contents not yet audited):

| Version | File | Size |
| --- | --- | --- |
| 19 April 2018 | `GeorefCropNutrientResponseFunctions for Tropical Africa.xlsx` | 3.44 MB |
| 18 December 2020 (files dated 21 December 2020 on Dryad) | `GeorefCropNutrientResponseFunctions_for_Tropical_Africa_Dec_18_2020.xlsx` | 3.15 MB |
| 22 August 2023 | `OFRA_RespFunc_Dataset_Updated_Aug_2023.xlsx` | 5.04 MB |

Start with the 2023 workbook. It is the latest file on this Dryad record. Check the Dryad licence before reuse.

The Dryad abstract compares three ways of defining which trials may be applied to a site: the OFRA Inference Tool (soil properties plus climate), Harvest Choice agro-ecological zones (climate), and Global Yield Gap Atlas recommendation domains (climate). For maize N, P, and K, the inference tool gave more specific functions and higher R². The tool was generally insensitive to widening the query ranges by about 20–25%, which matters in regions with few trials. That result supports region- and crop-based selection of functions. It does not provide a ready-made table of soil-test values paired with an “optimal fertilizer” label.

**Related project resources on the UNL and IFDC pages**, for interpretation and for the later conversion from nutrients to products:

- Excel Fertilizer Optimization Tools by agro-ecological zone
- Paper fertilizer optimization tools
- Extension training materials and calibration tools
- Agricultural ecozone maps
- Wortmann, C. S., & Sones, K. (Eds.). (2017). *Fertilizer Use Optimization in Sub-Saharan Africa*. CAB International. ISBN 978-1-78639-204-6 (hardcopy), 978-1-78639-205-3 (e-book)

**Compatibility decision:** conditional. Build a fertilizer optimization component that selects applicable response functions by crop and region. A row-level audit still has to confirm column names, Kenyan coverage, crops, nutrients, and whether the coefficients can supervise a regression model.

A related Eastern and Southern Africa maize workbook exists at https://datadryad.org/dataset/doi:10.5061/dryad.fg15tg2 (DOI https://doi.org/10.5061/dryad.fg15tg2). It is background literature, not a fifth core source, unless the 2023 OFRA audit shows a maize gap that this file fills.

### 3.3 FAO GAEZ v5 — crop suitability

**Use:** reference targets for the crop-ranking model, or a direct agronomic baseline. For the prototype, lock one historical period, rainfed conditions, and one management level, and use that combination for every crop.

Global Agro-Ecological Zoning version 5 (FAO, with IIASA) assesses land suitability and production potential from climate, soils, terrain, crop requirements, and management. Suitability is not an observed yield. GAEZ already reduces agro-climatic potential yield by soil and terrain constraints, using the Harmonized World Soil Database v2.01. A GAEZ suitability score is therefore a modelled label. Training against it measures how well a model reproduces GAEZ, which is not the same as predicting yields measured on Kenyan farms.

| | |
| --- | --- |
| GAEZ v5 data portal | https://data.apps.fao.org/gaez/ |
| FAO GAEZ programme page | https://www.fao.org/gaez/en |
| Model documentation (wiki) | https://github.com/un-fao/gaezv5/wiki |
| Introduction, including the six themes | https://github.com/un-fao/gaezv5/wiki/01.-Introduction |
| Platform user guide | https://github.com/un-fao/gaezv5/wiki/21.-GAEZ-v5-Platform-User-Guide |
| Source repository | https://github.com/un-fao/gaezv5 |
| Suitability and attainable yield (Theme 4 master notes) | https://data.apps.fao.org/catalog/dataset/gaez-v5-master-config/resource/31abf121-1644-4473-8c1b-bfec83112749 |
| Crop suitability, continuous index (about 1 km) | https://data.apps.fao.org/catalog/iso/66bfe451-c6da-4edd-940e-cb4b0b83725a |
| Contact | GAEZ@fao.org |

The portal is organised in six themes:

| Theme | Contents | Modelling role |
| --- | --- | --- |
| 1. Land and water resources | Land cover, soil and terrain, agro-ecological zones, protected and high-biodiversity areas | Context |
| 2. Agro-climatic resources | Thermal and moisture regimes, growing period | Context |
| 3. Agro-climatic potential yield | Potential before soil and terrain reduction; more than 100 crops and varieties | Context |
| 4. Suitability and attainable yield | Suitability class, suitability index, attainable yield; about 70 crops; rainfed and irrigated; low and high input | **Primary labels** |
| 5. Actual yields and production | Downscaled harvested area, production, and yield for 26 crops or crop groups | Possible later comparison |
| 6. Yield and production gaps | Gap between Theme 5 and Theme 4 | Possible later comparison |

Theme 4 is the layer family for this project. Documented products include:

- Crop suitability classes
- A continuous suitability index on approximately a 1 km grid, scaled from 0 to 10,000
- Agroecologically attainable yields
- Rainfed and irrigated scenarios
- Low-input and high-input management
- Historical climate assessments and future climate projections

Historical climate in GAEZ v5 uses AgERA5 (Copernicus) for 1981–2000 and 2001–2020. Future layers use CMIP6 (IPCC AR6) under three Shared Socioeconomic Pathways for 2021–2040, 2041–2060, 2061–2080, and 2081–2100. Irrigated results are limited to grid cells marked as cropland in the GAEZ v5 2020 land-cover inventory. The platform user guide encodes time, climate model, scenario, crop, water regime, and input level in the file names. Theme 4 filenames follow a pattern of the form `GAEZ-V5..<TIME_PERIOD>.<CLIMATE_MODEL>...<WATER_MANAGEMENT_LEVEL>`, with historical period codes including `HP8100` (1981–2000) and `HP0120` (2001–2020). Confirm crop codes and the exact suitability-class versus suitability-index variable names in the user guide and the Theme 4 filename table before download.

**Compatibility decision:** high, as a reference target or baseline. Select the GAEZ crops that also appear in the OFRA workbook and that matter for the Kenyan study area. Hold water regime, management level, climate baseline, and crop set fixed across the training table.

### 3.4 NASA POWER — historical climate

**Use:** daily or monthly meteorology at the farm coordinate, aggregated to seasonal rainfall, temperature, and related crop indicators. Projections 12–24 months ahead are uncertainty-aware scenarios evaluated against historical holdout periods.

NASA POWER (Prediction Of Worldwide Energy Resource), from NASA Langley Research Center, provides analysis-ready solar and meteorological time series. This project uses the **Agroclimatology (AG)** community. Daily data are available from 1 January 1981 to near real time. The daily API returns parameters at the original source resolution. Meteorology is on a 0.5° × 0.625° grid; solar parameters are on a 1° × 1° grid. Requesting a denser grid repeats the same cell. Within about two to three months of real time, improved climate-quality products replace the first meteorological release, so recently downloaded series can change.

| | |
| --- | --- |
| POWER home | https://power.larc.nasa.gov/ |
| Documentation | https://power.larc.nasa.gov/docs/ |
| Agroclimatology community | https://power.larc.nasa.gov/docs/methodology/communities/ |
| Daily API | https://power.larc.nasa.gov/docs/services/api/temporal/daily/ |
| API explorer | https://power.larc.nasa.gov/api/pages/ |
| Daily OpenAPI document | https://power.larc.nasa.gov/api/temporal/daily/openapi.json |
| API tutorial | https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/ |
| Getting started | https://power.larc.nasa.gov/docs/tutorials/api-getting-started/ |
| Data Access Viewer | https://power.larc.nasa.gov/data-access-viewer/ |

No API key is required for the public temporal endpoints. Follow the acknowledgement guidance published by the POWER project. Avoid large bursts of synchronous requests; POWER may block clients that degrade the service. A single-point request allows up to 20 parameters. Daily requests default to local solar time; set `time-standard=LST` or `time-standard=UTC` explicitly. Formats include JSON, CSV, ASCII, NetCDF, and ICASA (ICASA is AG-only).

**Parameters selected for the first climate feature set**

| Parameter | API code | Intended use |
| --- | --- | --- |
| Temperature at 2 m | `T2M` | Crop temperature suitability |
| Maximum temperature at 2 m | `T2M_MAX` | Heat-stress assessment |
| Minimum temperature at 2 m | `T2M_MIN` | Cold-stress assessment |
| Precipitation corrected | `PRECTOTCORR` | Rainfall availability |
| Relative humidity at 2 m | `RH2M` | Growing conditions |
| All-sky surface shortwave downward irradiance | `ALLSKY_SFC_SW_DWN` | Crop growth conditions |

Confirm units in the POWER parameter dictionary before training. In the AG daily series, temperature is in °C and corrected precipitation is in mm/day. Solar radiation units depend on the community; record the unit returned in the response header rather than assuming it.

Worked example for the same Kenyan documentation coordinate, one year, Agroclimatology, CSV. This is an access example, not a chosen study farm:

```text
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN&community=AG&longitude=35.2400&latitude=-0.7196&start=20010101&end=20011231&format=CSV&time-standard=LST
```

**Aggregation:** sum `PRECTOTCORR` over the season of interest, and summarise temperature, humidity, and radiation as means or extremes over that same season. Define the season (for example the long rains and the short rains in Kenya) before computing features, and use the same definition for every location.

**Compatibility decision:** high for historical seasonal indicators. For a 12–24 month planning horizon, publish a range or scenario based on year-to-year historical spread. A forecasting model may be trained, and it must be scored on historical holdout years.

### 3.5 What each source is allowed to answer

| Question | Source | What the value represents |
| --- | --- | --- |
| What is the soil at this point? | iSDAsoil | A mapped prediction and an uncertainty interval |
| How has weather behaved here? | NASA POWER | Historical meteorology on a coarse grid |
| How suitable is this crop here under a stated management and water regime? | GAEZ v5 | A modelled suitability index or class that already includes soil, terrain, and climate |
| How has this crop responded to N, P, K, or another nutrient in similar conditions? | OFRA | A response function from field trials, selected by crop and region |

### 3.6 Supporting sources

The minimum stack that can start a prototype is iSDAsoil, OFRA, FAO GAEZ, and NASA POWER. The sources below fill gaps those four do not cover: a second soil estimate, measured soil samples, a Kenya catalogue, a seasonal outlook, trial yields, and the tables that turn a nutrient rate into a product and a cost. They are not merged into the first training table until each file has been inspected.

Laboratory sampling is not required to build the models. Mapped and archived soil data are enough for development. A laboratory soil test remains the right way to override a map for one field, and the right way to validate a fertilizer rate. pH already expresses acidity and alkalinity, so those three words are not three features. Exchangeable acidity is a separate laboratory or map layer where a source provides it.

| Priority | Dataset | Role | Access |
| --- | --- | --- | --- |
| Core | iSDAsoil | Soil features at the farm point | [Developer portal](https://www.isda-africa.com/isdasoil/developer/) |
| Core | OFRA | Nutrient-response functions and Kenya zone tools | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1), [IFDC tools](https://ifdc.org/resources/ofra-analysis-tools/) |
| Core | FAO GAEZ v5 | Crop suitability labels | [Portal](https://data.apps.fao.org/gaez/) |
| Core | NASA POWER | Historical rainfall, temperature, humidity, radiation | [Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) |
| Next | ICPAC | Seasonal rainfall and temperature outlooks for East Africa | https://www.icpac.net/ |
| Next | Fertilizer composition | Nutrient fraction of urea, DAP, CAN, NPK blends, and similar products | Product specifications and the Kenya OFRA chapter; not in the four core files |
| Next | Prices and budget | Fertilizer price, crop price, farm area, fertilizer budget | Local market figures supplied for the study season; OFRA tools already expect these |
| Support | KALRO Land, Soil and Crop Hub | Find Kenyan soil, crop, and climate layers | https://lschub.kalro.org/ |
| Support | AfSIS soil chemistry | Georeferenced samples with laboratory measurements, 19 countries, Phase I 2009–2013 | https://registry.opendata.aws/afsis/ |
| Support | SoilGrids | Global 250 m maps of pH, organic carbon, nitrogen, CEC, sand, silt, and clay | https://soilgrids.org/ and https://docs.isric.org/globaldata/soilgrids/ |
| Validation | TAMASA | Maize nutrient-omission trials with soil tests and yields in Ethiopia, Nigeria, and Tanzania | https://tamasa.cimmyt.org/tools-data/tamasa-data/ |
| Validation | AgWise / KALRO | Kenya trial harmonisation and site-specific N and P advice | Alliance reports linked in [section 13.2](#132-kalro-platforms-agwise-and-sleek) |

**How a farmer location becomes soil features.** The user selects a point, by GPS or on a map. The application calls the iSDAsoil API, stores depth, units, and uncertainty, and continues. The user does not have to type pH or NPK. An uploaded laboratory result replaces the mapped value for that field and is flagged as measured rather than modelled.

**Bulk extracts for training.** iSDAsoil also publishes raster layers, including Cloud-Optimized GeoTIFFs indexed from the Land, Soil and Crop Hub and described by Hengl et al. (2021), https://doi.org/10.1038/s41598-021-85639-y. A regional extract can be sampled to a table. The FAQ still points to the AWS bulk distribution: https://www.isda-africa.com/isdasoil/faq/

**What each soil source actually contains.** Availability below follows the published layer lists, not a download audit.

| Variable | iSDAsoil | SoilGrids 2.0 | Africa soil-property maps at 250 m (AfSIS-based, Hengl et al. 2015) |
| --- | --- | --- | --- |
| pH | Yes | Yes (`phh2o`) | Yes |
| Organic carbon | Yes, g/kg | Yes | Yes |
| Nitrogen | Yes, total, g/kg | Yes | Yes, total |
| Phosphorus | Yes, Mehlich-3 extractable | Not in the standard SoilGrids 2.0 property set | Not the same Mehlich-3 layer |
| Potassium | Yes, Mehlich-3 extractable | Not in the standard SoilGrids 2.0 property set | Exchangeable K is mapped |
| Texture (sand, silt, clay) | Yes | Yes | Yes |
| CEC | Yes, effective CEC | Yes | Yes |
| Exchangeable acidity | Not a dedicated FAQ layer; pH and extractable aluminium are partial indicators | Not a standard SoilGrids 2.0 layer | Mapped |
| Salinity | Not in the core FAQ property list; confirm with `/layers` before promising it | Not a standard layer | Confirm before use |

SoilGrids is a comparison layer at 250 m, with a 90% prediction interval on the published maps (CC BY 4.0). Its REST point API has been unavailable for long periods; use the documented map downloads or coverage services, and check current access before writing a client. Africa-wide 250 m maps that include exchangeable acidity and exchangeable bases are the ISRIC–AfSIS product described at https://isric.org/projects/soil-property-maps-africa-250-m-resolution (Hengl et al., 2015, *PLOS ONE*).

AfSIS on AWS is paired infrared spectra and wet-chemistry measurements, under the Open Database Licence with attribution. It is the closest open source of measured, rather than purely mapped, soil values. It is still not a sample from the farmer’s field.

**ICPAC** (IGAD Climate Prediction and Applications Centre, Nairobi) issues objective seasonal forecasts for Eastern Africa, including the Greater Horn of Africa Climate Outlook Forum (GHACOF). Use those outlooks for the coming season’s planting context. They are probabilistic (below normal, normal, above normal), not a 12–24 month daily forecast. https://www.icpac.net/

**Fertilizer products and money.** OFRA can imply a nutrient rate. It does not name the bag on the agro-dealer’s shelf until a composition table converts kilograms of N, P, and K into kilograms of urea, DAP, CAN, or a stated NPK blend. The Kenya chapter lists products actually used on maize, including CAN, urea, DAP, ammonium sulphate, and blends such as 23:23:0 and 17:17:17. Prices, farm area, and the fertilizer budget are what let the optimizer copy OFRA’s economic step. Collect them for the study season; do not invent them.

**Illustrative soil table only.** Values like the ones below are a shape for the processed file, not measurements.

| Location | pH | Organic carbon | N | P | K |
| --- | --- | --- | --- | --- | --- |
| Farm A | 5.8 | to be converted to the source unit | source unit | source unit | source unit |

Mapped soil columns are features. They are not the correct crop and they are not the correct fertilizer rate. Those targets come from GAEZ, OFRA, and, later, trial yields.

## 4. Dataset compatibility

**Verdict: feasible, with limitations.** All four sources can contribute. They cannot be concatenated into one table and treated as a single supervised dataset.

### 4.1 Why a single merged training table would be misleading

| Issue | Why it matters | How The Better Farmer handles it |
| --- | --- | --- |
| Different spatial resolution | iSDAsoil describes about 30 m. GAEZ suitability is about 1 km. NASA POWER meteorology is 0.5° × 0.625° (tens of kilometres). Solar POWER fields are 1°. | Extract each source at the farm coordinate and store its native resolution. Recommendations must not be presented as 30 m climate or 30 m fertilizer precision. |
| Different time periods | Soil maps, GAEZ baselines (1981–2000 or 2001–2020), POWER daily history (1981 to near real time), and OFRA trials (many years of compiled experiments) are not one growing season. | Store `climate_period`, soil depth, GAEZ time code, and the OFRA source reference on every row. |
| Different prediction targets | A suitability class is not a fertilizer rate, and a response coefficient is not a yield observation for that farm. | Crop model and fertilizer model are trained and evaluated separately, then combined in the recommendation engine. |
| GAEZ is not an independent outcome | GAEZ suitability already embeds soil and climate from HWSD and AgERA5, which are not the same products as iSDAsoil and POWER. | Use GAEZ as a reference target or baseline. State that agreement with GAEZ is reproduction of GAEZ. |
| OFRA is not a farm-level fertilizer label | Functions are regional and crop-specific. Column names, Kenyan row counts, and usable coefficients are not yet audited. | Select functions by crop and response region. Inspect the 2023 workbook before deciding that supervised regression is supported. |

### 4.2 Component verdicts

| Component | Verdict |
| --- | --- |
| Soil and climate integration on coordinates | Feasible |
| Crop suitability modelling against GAEZ | Feasible |
| Fertilizer optimization from OFRA | Needs validation against the actual workbook |
| 12–24 month climate planning | Scenario-based |
| One model that does soil, climate, crops, and fertilizer together | Out of scope |

Overlapping Kenyan coverage is not yet demonstrated. Documentation shows that each source can be queried or downloaded for Kenya or tropical Africa. It does not show that the same crops, seasons, and locations are dense enough to train and validate the full system.

## 5. Integration architecture

```text
Farm location
  latitude, longitude, selected crop, farm area
        |
        +-- iSDAsoil -------- soil estimates (depth, units, uncertainty)
        |
        +-- NASA POWER ------ seasonal climate from daily history
        |
        +-- FAO GAEZ v5 ----- suitability index and class
        |
        +-- OFRA ------------ nutrient-response functions by crop and region
        |
        v
Data integration
  coordinate extraction, unit conversion, missing values, crop-name mapping,
  reference period attached to every field
        |
        +-- Crop suitability model -------- rank suitable crops
        |
        +-- Fertilizer model -------------- nutrient response or rate
                 |
                 +-- composition table ---- kilograms of a named product
                 |
                 v
        Recommendation engine
          ranked crops, nutrient estimate, fertilizer quantity, stated uncertainty
```

Crop names must be mapped by hand across GAEZ crop codes, OFRA crop names, and the names shown to the user. That crosswalk is a project table, not something the four sources share.

## 6. Recommended data model

Keep three related tables. A single wide matrix that repeats OFRA coefficients on every soil pixel would invent precision the trials do not have.

These schemas are proposed. They are not confirmed column names from the source files. Names inside OFRA sheets and GAEZ rasters must replace the placeholders after the audit.

### Table A — Environmental data

One row per farm location (and, when climate is seasonal, per location and season).

| Field | Source |
| --- | --- |
| `location_id`, `latitude`, `longitude` | Application |
| `soil_ph`, `organic_carbon`, `nitrogen` | iSDAsoil |
| `phosphorus`, `potassium`, `cec` | iSDAsoil, where the layer returns a value |
| Soil uncertainty bounds, units, and depth | iSDAsoil |
| `rainfall_seasonal`, `temperature_mean`, plus max/min temperature, humidity, and solar radiation as aggregated | NASA POWER |
| `climate_period`, `soil_depth` | Dataset metadata |
| Native resolution of each extracted layer | Dataset metadata |

### Table B — Crop suitability

One row per location and crop, for the locked GAEZ scenario.

| Field | Source |
| --- | --- |
| `location_id` | Application |
| `crop_name` | GAEZ crop classification, mapped to the project crop list |
| `suitability_index` | FAO GAEZ v5 |
| `suitability_class` | FAO GAEZ v5 |
| `management_level`, `water_regime` | FAO GAEZ v5 |
| GAEZ time-period code and climate source | FAO GAEZ v5 |

### Table C — Nutrient response

One row per crop, response region, and nutrient function. Linked to farms through crop and region, not through a copied soil row.

| Field | Source |
| --- | --- |
| `crop_name` | OFRA |
| `response_region` | OFRA |
| `nutrient` | OFRA |
| `response_parameters` | OFRA |
| `source_reference` | OFRA |
| Coordinates and trial metadata, if present in the workbook | OFRA |

A fourth, small, non-ML reference table is required before any commercial product is recommended: nutrient composition and application constraints for products such as NPK grades, DAP, urea, and similar fertilizers available to the user. That table is not in the four source datasets.

## 7. What can actually be predicted

Soil at a point, a GAEZ-aligned crop ranking, and a historical seasonal climate summary are the outputs the current sources can support with high feasibility.

Nutrient rates are conditional on finding OFRA functions that apply to the crop and region, and on stating prices or other economic assumptions if the rate is an economic optimum rather than an agronomic response. Converting a nutrient rate into bags or kilograms of a branded product depends on the composition table.

Farm yield is out of scope until an independent yield dataset is added and used as a validation target. Agreement with GAEZ is reported as agreement with GAEZ.

## 8. Machine learning architecture

Machine learning is the core of The Better Farmer. It is not a single algorithm forced across every task. Soil values stay as features because iSDAsoil has already modelled them. Supervised models are trained where a target exists. Time-series methods are reserved for climate. Optimization sits after the nutrient model, where fertilizer quantities have to respect composition and constraints.

### 8.1 Components

| Component | Role of ML | Candidate algorithms |
| --- | --- | --- |
| 1. Soil | Feature extraction only | None added; iSDAsoil supplies the estimates |
| 2. Climate | Seasonal summaries, then time-series models of seasonal behaviour | SARIMA; XGBoost with lag features |
| 3. Crop recommendation | Classification or ranking | XGBoost primary; Random Forest comparison |
| 4. Fertilizer recommendation | Regression of nutrient response or application rate, if OFRA rows support supervision | XGBoost Regressor primary; Random Forest Regressor comparison |
| 5. Fertilizer quantity | Regression output plus constrained optimization | XGBoost plus a constrained optimizer |

Candidates are not selected winners. Selection happens after geographically separated evaluation.

### 8.2 Features, targets, and split

**Crop model**

- Inputs (X): soil pH, nitrogen, phosphorus, potassium, organic carbon, CEC, seasonal rainfall, seasonal temperature (and the other POWER aggregates that survive a missing-value check), location, and crop identity where the model is trained across crops.
- Target (y): GAEZ suitability class and/or suitability index for the locked scenario.
- Algorithms: XGBoost and Random Forest.
- Split: training and test locations must be geographically separated, so neighbouring points do not leak into both sides. A random row split is not acceptable for this spatial data.
- Reported meaning: skill at reproducing GAEZ under the locked scenario.

**Fertilizer model**

- OFRA must be audited first. The workbook may support regression on response coefficients or on rates derived from those functions. It may instead be better used as an explicit response function inside an optimizer, with ML only where the rows truly pair inputs and outcomes.
- If supervision is supported: XGBoost Regressor and Random Forest Regressor, with the same geographic discipline where trial coordinates exist.
- If supervision is not supported: keep the published response functions and use constrained optimization, and document that this component is agronomic optimization rather than a trained regressor. The project still uses ML for crop ranking and climate. This branch is a data outcome, not a retreat from machine learning.

**Climate model**

- Inputs: POWER daily history aggregated to the defined Kenyan seasons, with lag features for the XGBoost comparison.
- Candidates: SARIMA, and XGBoost with lags.
- Evaluation: historical holdout years, with prediction intervals or scenario ranges.
- The 12–24 month product shown to a user is a risk scenario, not a deterministic forecast.

**Recommendation engine**

The engine is deterministic given the model outputs:

1. Rank crops from the crop model (or from GAEZ directly when the model is being compared with that baseline).
2. Estimate nutrient response or nutrient rate from the fertilizer component.
3. Convert nutrients to product quantities with the composition table and application constraints.
4. Attach the source period, the spatial resolution, and the soil uncertainty to the result.

### 8.3 Recommended comparison plan

| Task | Primary candidate | Comparison model |
| --- | --- | --- |
| Crop suitability | XGBoost | Random Forest |
| Nutrient response | XGBoost Regressor | Random Forest Regressor |
| Seasonal climate | SARIMA | XGBoost with lag features |

Compare crop and nutrient models with the same features, the same geographic split, and metrics chosen for the target type (classification or ranking metrics for suitability; regression metrics for nutrient response). Keep the GAEZ baseline in the comparison: a model that cannot beat reading the GAEZ layer directly has not earned a place in the engine.

## 9. Methodology statement

The project will employ supervised machine learning to analyse soil characteristics, historical climatic conditions, crop suitability, and nutrient-response data. Random Forest and XGBoost will be developed and evaluated for crop suitability prediction and for fertilizer nutrient-response modelling. Time-series methods will be investigated for seasonal climate analysis. The resulting models will be integrated into a recommendation engine that ranks suitable crops and estimates fertilizer requirements for selected agricultural locations.

The immediate methodological step is to build the training tables: define X, y, and the geographic train–test strategy for each model, using the three-table structure in [section 6](#6-recommended-data-model). That build is the real compatibility test.

## 10. Validation limits

- Reproducing GAEZ suitability is not validation against harvested yield.
- iSDAsoil phosphorus is a weaker published layer than pH, organic carbon, or potassium. Nutrient advice that leans on soil P should show the iSDAsoil uncertainty.
- OFRA functions were estimated on research plots and then generalised by inference space. Applying a regional function to one farm is an extrapolation, even when the crop and zone match.
- POWER and GAEZ grid cells can hide within-cell differences that a 30 m soil map still sees. The user interface should name the climate and suitability resolution.
- Future GAEZ climate layers (CMIP6, 2021 onward) are a later scenario tool. They are not the training target for the first prototype, and they are not a weather forecast.

Independent yield observations, if a defensible Kenyan source is added later, would be a new validation target. They are not part of the four-source design.

## 11. Work remaining before implementation

The documentation review shows a workable architecture. It does not show that Kenyan records overlap enough to train the full system.

1. Download `OFRA_RespFunc_Dataset_Updated_Aug_2023.xlsx` from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1) and inspect sheets, crop coverage, Kenyan records, nutrients, and response coefficients.
2. Decide, from that inspection, whether the fertilizer component is supervised regression, response-function optimization, or both.
3. Choose the project crop list as the intersection of OFRA crops, GAEZ v5 crops, and the crops the system will offer.
4. Select GAEZ v5 Theme 4 layers for those crops and the study area, under one historical baseline, rainfed conditions, and one management level.
5. Retrieve sample iSDAsoil and NASA POWER records at matching coordinates, including the documented Kenyan example (−0.7196, 35.2400) and a small set of real study points.
6. Record missing values, units, geographic overlap, and the presence or absence of any independent yield data.
7. Write the crop-name crosswalk and the fertilizer composition table.
8. Only then fit XGBoost and Random Forest, plus the climate candidates, and keep the better model for each task.

Until step 1 and step 4 are done, no final algorithm is chosen.

## 12. Prototype settings to lock

These settings keep the first training table internally consistent. They are project choices still to be confirmed in code and filenames, not values copied out of a single source file.

| Setting | Prototype choice | Notes |
| --- | --- | --- |
| Study geography | One or two Kenyan zones, chosen after the OFRA audit | Candidates: Nakuru, Uasin Gishu, Trans Nzoia. OFRA already has Central, Rift Valley, Western, Eastern, and Coastal tools. |
| Soil depth | 0–20 cm | 20–50 cm retained only if a crop requires it |
| Soil properties | pH, organic carbon, total N, extractable P, extractable K, ECEC | Plus uncertainty, units, and the API property name |
| Climate community | NASA POWER Agroclimatology (`AG`) | Daily point API, then seasonal aggregates |
| Climate parameters | `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `ALLSKY_SFC_SW_DWN` | Units taken from the API response |
| GAEZ theme | Theme 4, suitability index and suitability class | Attainable yield can be stored beside the index |
| GAEZ water regime | Rainfed | Irrigated layers are a later comparison, and only on 2020 cropland cells |
| GAEZ management | One level for every crop | Low or high input; the choice is applied uniformly |
| GAEZ historical baseline | 2001–2020 (`HP0120`) as the recent historical period | 1981–2000 (`HP8100`) kept for sensitivity checks |
| GAEZ future layers | Not in the first training target | Available later as climate scenarios (CMIP6) |
| OFRA file | August 2023 workbook | Older 2018 and 2020 files kept for traceability |
| Join | Coordinates for environment and suitability; crop plus region for OFRA | Three tables, not one wide CSV |
| Crop model split | Geographic separation | Random row splits are not used |
| Crops | Maize, beans, and sorghum if both GAEZ and OFRA cover them | Drop any crop the audit cannot support |
| Planning horizon | 12–24 months as historical seasonal scenarios | ICPAC outlooks for the next season; exact weather in the 12–24 month window is not a deliverable |
| Laboratory override | Optional upload replaces the iSDAsoil value for that field | Mapped estimate remains the default |
| Fertilizer economics | Composition table required; prices and budget used when collected | Same inputs the Kenya OFRA tools expect |

## 13. Related systems

Fertilizer recommendation and crop suitability advice are an established part of precision agriculture and soil-nutrient management. A supervisor can fairly reject a claim that no such system exists. The Better Farmer is positioned as a test of whether public soil, climate, suitability, and response data can support an explainable machine-learning prototype for a defined Kenyan area, judged against agronomic tools that already exist.

Three kinds of evidence are easy to confuse and are kept separate:

| Evidence | Question it answers | Primary lead |
| --- | --- | --- |
| Crop suitability criteria | Which crops can grow under stated soil and climate conditions? | [FAO GAEZ v5](https://data.apps.fao.org/gaez/) |
| Fertilizer response trials | How does yield change when nutrients or rates change? | [OFRA on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1); [TAMASA](https://tamasa.cimmyt.org/tools-data/tamasa-data/) where access allows |
| Agronomic recommendations | What product, rate, and practice is advised for a crop and place? | [OFRA Kenya tools](https://ifdc.org/resources/ofra-analysis-tools/); KALRO–CGIAR AgWise reports |

### 13.1 OFRA country tools

OFRA built crop–nutrient response functions and Fertilizer Optimization Tools for agro-ecological zones in 13 sub-Saharan African countries. The tools combine crop responses, fertilizer cost, crop value, land allocation, and the farmer’s fertilizer budget. They use linear optimization. They are not a classifier that outputs a fertilizer name from a soil-test row.

Kenya is covered directly. The [IFDC OFRA tools page](https://ifdc.org/resources/ofra-analysis-tools/) lists Excel and paper tools for Central Kenya, Coastal Kenya, Eastern Kenya (upper and lower), the Rift Valley (upper and lower), and Western Kenya (upper and lower), with separate sheets for common products such as urea, DAP, and CAN. The Kenya chapter of the OFRA book is available as [Chapter 7 (PDF)](https://ifdc.org/wp-content/uploads/2026/01/OFRA-Chapter-7_Kenya_HR.pdf). That chapter is the agronomic benchmark for this project: economically optimum rates, recommended rates, and the 4Rs (right product, rate, time, and method).

The same programme produced zone tools for Ethiopia, Ghana, Nigeria (including Guinea Savanna and Sahel zones), Tanzania, and other countries. Those tools show that the method travels across countries. They also show that each tool is zone-specific. A model trained on a general fertilizer-name dataset is not a substitute for a Kenya zone function.

### 13.2 KALRO platforms, AgWise, and SLEEK

KALRO already operates digital advisory systems. Overlap with The Better Farmer is real. The contribution has to be measured: accuracy, rate error, interpretability, and behaviour on Kenyan data, compared with a rule-based or OFRA baseline.

| Initiative | What it does | Relation to this project |
| --- | --- | --- |
| [KALRO Selector](https://keep.kalro.org/), listed on the Kenya Digital Agriculture Platform | Advises which crops, pasture, and livestock suit a location | Suitability advisory. It is not, by itself, a fertilizer-rate optimizer. Delivery channel mentioned for later AgWise advice. |
| [Kenya Agricultural Observatory Platform (KAOP)](https://kaop.co.ke/) | Location-specific weather and agro-weather advisories, including mobile and USSD access | Related climate-advisory service. Not a core training dataset. |
| [Land, Soil and Crop Hub](https://lschub.kalro.org/) | Catalogue and map viewer for Kenyan and regional soil, crop, climate, and economic information. [User guide](https://lschub.kalro.org/docs/). | Kenya-specific discovery and a place to look for supporting layers. The hub points at datasets; it does not replace iSDAsoil or OFRA. |
| KALRO–CGIAR AgWise | Site-specific nutrient recommendations from harmonised agronomic trials, soil, climate, and geospatial covariates, with machine learning | Closest research neighbour. See below. |
| Kenya Soil Information System (KenSIS) | Planned national soil information system. Roadmap workshop with KALRO, the Ministry, ISRIC, and CABI, March 2025. [ISRIC note](https://isric.org/news/strengthening-the-soil-information-system-sis-in-kenya-a-collaborative-roadmap/); [roadmap record](https://zenodo.org/records/17696631). | Context for Kenyan soil data governance. Not a downloadable training table for this prototype. |

**AgWise.** The Alliance of Bioversity International and CIAT, IITA, and KALRO have been consolidating legacy fertilizer-response trials and joining them to soil, climate, and location data through the AgWise framework. Public descriptions refer to more than 150,000 data points from fertilizer response studies across Kenya. A 2025 report on long-rains maize in Western Kenya documents a pipeline that predicts maize yield and then searches feasible nitrogen and phosphorus rates, including rainfall-tercile scenarios. Reported held-out skill in that report is about R² 0.74 on training and 0.75 on testing. Those figures describe that maize pipeline, not this project.

| | |
| --- | --- |
| Partnership account | https://alliancebioversityciat.org/stories/tailored-fertilizer-advice-kenya-diverse-soils-crops |
| Data-harmonisation account | https://alliancebioversityciat.org/stories/scattered-agronomic-data-smarter-fertilizer-management-maize-productivity-kenya |
| Methods paper / report listing | https://alliancebioversityciat.org/publications-data/scattered-data-site-specific-nutrient-recommendation-systems-data-driven-approach |
| Western Kenya maize advisory report (2025) | https://alliancebioversityciat.org/publications-data/agwise-based-site-specific-and-climate-smart-fertilizer-advisory-long-rains-maize |
| Fertilizer module source | https://github.com/CGIAR-AgWise/agwise-fertilizer |

The public reports do not establish that the underlying trial-level database can be downloaded. Kenyan trial records should be requested from KALRO or the AgWise authors, with permission recorded before use. They are a validation and benchmarking route, not a prerequisite for starting on iSDAsoil, GAEZ, POWER, and OFRA.

**SLEEK is a different system.** No official KALRO fertilizer or crop-advisory platform named SLEEK was found. SLEEK is the System for Land-based Emissions Estimation in Kenya, a government land-sector measuring, reporting, and verification programme associated with the Ministry of Environment and Forestry, with KALRO among the institutions that contributed expertise. Its engine is the Full Lands Integration Tool (FLINT), used to combine land cover, biomass, climate, soil, and crop information for greenhouse-gas accounting and land-sector planning. It is not a farmer fertilizer recommender.

| | |
| --- | --- |
| SLEEK and FLINT in Kenya | https://moja.global/the-flint-in-kenya/ |
| FLINT description | https://moja.global/flint/ |
| Operational note (2019 runs) | https://moja.global/finally-sleek-flint-is-running-in-kenya/ |

If a screenshot or link of a product literally branded SLEEK turns up, it should be checked against these pages before it is described as a KALRO fertilizer tool.

### 13.3 Trial datasets outside the four core sources

[TAMASA](https://tamasa.cimmyt.org/tools-data/tamasa-data/) (Taking Maize Agronomy to Scale in Africa) collected soil tests, fertilizer treatments, and maize yields in Ethiopia, Nigeria, and Tanzania. Nutrient Omission Trials drop one nutrient at a time so the yield gap can be attributed. An example public Nigeria set is on the CIMMYT repository: https://data.cimmyt.org/dataset.xhtml?persistentId=hdl:11529/10548310. Some TAMASA files require an access request, and licences differ by file. These trials are not Kenyan labels. They are a pattern for what a validation table looks like: soil test, treatment, and measured yield in the same row. CIMMYT’s “space to place” targeting work uses this kind of trial to move from a blanket rate to a site rate. Use TAMASA to understand that method. Do not silently pool Tanzanian or Nigerian yields into a Kenyan training target.

IITA’s data repository (https://data.iita.org/) holds further field, plant, and plot sets from soil-management and fertilizer experiments, including AfSIS-related diagnostic work. Search it when the OFRA audit shows a missing crop or nutrient. Treat each dataset’s licence separately.

### 13.4 Machine-learning studies elsewhere

These papers are benchmarks for method, not evidence that the same accuracy will appear in Kenya.

Gimba and Mishra (manuscript published 31 December 2025, *JITSI*) compared ensemble classifiers for fertilizer **type** in Andhra Pradesh, India, using nitrogen, phosphorus, potassium, and moisture. The dataset they describe has 2,199 rows and 9 fertilizer varieties, from the Indian Institute of Soil Science. They report CatBoost accuracy of 94.78%, ahead of Extra Trees, gradient boosting, AdaBoost, and bagging. The paper recommends a fertilizer name. It does not estimate kilograms per hectare, and it does not use Kenyan soils or OFRA response functions. https://jurnal-itsi.org/index.php/jitsi/article/view/509

Dey and Ahmed (2024), *Heliyon*, compared SVM, XGBoost, Random Forest, k-nearest neighbours, and decision trees for **crop** recommendation using NPK, soil pH, temperature, rainfall, and humidity on a Kaggle dataset. XGBoost had the highest reported precision in that study (about 99% on the agricultural-crop subset they evaluated). That target is crop identity, not fertilizer rate. Very high scores on this kind of public crop table are not a Kenyan result. https://doi.org/10.1016/j.heliyon.2024.e25112

Useful contrast for the concept paper: the Andhra Pradesh study is a fertilizer-type classifier; the *Heliyon* study is a crop classifier; OFRA is an optimizer of nutrient rates. The Better Farmer has to say which of those three jobs each model is doing.

### 13.5 What to claim, and what to measure

Claim a comparison, not an empty field. The comparison worth writing down is:

1. Machine-learning crop ranking against GAEZ suitability for the same locations and the same locked scenario.
2. Nutrient-rate estimates against OFRA response functions and, where a Kenya tool exists, against that zone’s Fertilizer Optimization Tool.
3. Explanations of which soil and climate features drive each rank or rate, plus the resolution and period of the inputs.
4. Whether a model trained on general soil–fertilizer tables holds up when the features are Kenyan mapped soils and the products are fertilizers actually sold in the study zone. That last gap is still a literature claim until the local test is done.

An illustrative farmer report, not a model output, looks like this. Every number below is fictional.

```text
THE BETTER FARMER — Farm suitability report (illustrative only)
Location: Nakuru, Kenya
Soil pH: 5.8
Organic matter: 2.1%
Planning horizon: 24 months (historical seasonal scenario, not a weather forecast)

Crop suitability ranking (fictional)
  Maize     84%
  Beans     77%
  Sorghum   69%

Fertilizer (fictional, pending real soil and OFRA checks)
  Nutrients, product, kg/ha, and estimated cost would appear here
  only after composition, price, and response data are validated.
```

## 14. Chapter One: Introduction

This is the working concept-paper chapter. It matches the design in the sections above. Objectives, scope, and budget still need a supervisor’s confirmation, and the study area and crop list are not yet locked.

### 1.1 Introduction

Agricultural production depends on the suitability of a crop to local soil and climate conditions and on the effective use of nutrients. Soil acidity, organic matter, nutrient availability, temperature, and rainfall can vary considerably across locations. Consequently, the same crop or fertilizer application rate may not produce comparable results on different farms. Farmers need accessible information that connects these factors before they select crops and invest in fertilizers.

Recent developments in digital soil mapping, geospatial datasets, and machine learning make it possible to combine environmental information and generate location-informed recommendations. The Better Farmer is a proposed machine-learning-based decision-support prototype that will assess crop suitability and estimate fertilizer nutrient requirements using publicly accessible soil, climate, crop-suitability, and nutrient-response datasets. The system will provide explanations and indicate uncertainty, rather than present its estimates as laboratory-confirmed advice.

### 1.2 Background of the Study

#### 1.2.1 Background

Crop selection is often influenced by experience, prevailing market demand, and general extension advice. These are valuable sources of knowledge, but they may not fully capture differences in soil properties and seasonal climate. Fertilizer selection also requires an understanding of the crop’s nutrient requirements and its response to added nutrients. Applying an unsuitable product or quantity can waste resources, while inadequate nutrient supply may constrain yields.

Data-driven agricultural advisory systems have become increasingly practical. The iSDAsoil platform provides mapped estimates of soil characteristics across Africa; NASA POWER offers historical meteorological data; FAO’s Global Agro-Ecological Zones (GAEZ) supplies crop suitability and potential production information; and the Optimizing Fertilizer Recommendations in Africa (OFRA) initiative has compiled crop–nutrient response functions for fertilizer optimization. These resources offer complementary evidence, although they differ in geographic resolution, measurement methods, units, and time periods.

Soil pH is the measurement that expresses acidity or alkalinity. The project will not treat pH, acidity, and alkalinity as three independent inputs. Exchangeable acidity, and laboratory measures of alkalinity or salinity, will be used only when a source actually provides them.

#### 1.2.2 Overview of Existing Approaches

Traditional approaches include farmer experience, agricultural extension guidance, and laboratory soil testing. Laboratory testing can provide field-specific measurements but requires sampling, processing, and interpretation. Digital soil maps can provide more accessible estimates, although mapped values are not a substitute for a soil test. A farmer will be able to enter or upload a laboratory result that overrides the mapped estimate for that field.

Existing initiatives such as OFRA focus on the economic and agronomic response of crops to nutrients, while GAEZ assesses crop suitability under defined environmental and management assumptions. KALRO’s Selector and Land, Soil and Crop Hub, and the KALRO–CGIAR AgWise work on site-specific nutrient recommendations, show that crop and fertilizer decision support is already active in Kenya. SLEEK, the System for Land-based Emissions Estimation in Kenya, is a land-sector greenhouse-gas accounting system and is not a fertilizer recommendation platform. The proposed contribution is to develop and evaluate an integrated machine-learning prototype for a clearly defined Kenyan study area, against these agronomic baselines.

#### 1.2.3 Overview of the Proposed System

The Better Farmer will allow a user to select a farm location and view ranked crop options alongside estimated nutrient requirements for a selected crop. Soil features will be retrieved from iSDAsoil, climate summaries derived from NASA POWER, crop suitability reference targets obtained from FAO GAEZ, and crop–nutrient response information drawn from OFRA. Seasonal outlooks from ICPAC may inform near-term planting context. Separate machine learning models will be evaluated for crop suitability and nutrient response or rate estimation, subject to the availability of appropriate training labels. A simple interface will communicate the results, data sources, and limitations.

The proposed 12–24-month planning component will describe historical seasonal patterns and scenario-based climate risks. It will not claim to predict exact weather conditions two years in advance. GAEZ future-climate layers are long-term suitability scenarios, not a monthly forecast for the next two seasons. Commercial fertilizer quantities will be calculated only where nutrient composition and agronomic constraints are documented. Where fertilizer prices, farm size, and a budget are available, the fertilizer step will follow the OFRA practice of recommending what is economically practical as well as agronomically responsive.

### 1.3 Problem Statement

Crop and fertilizer decisions require information from several sources that are often difficult for individual farmers to combine. Soil maps describe soil properties, climate databases describe weather history, crop suitability models identify potential growing conditions, and nutrient-response datasets characterize the effects of fertilizer nutrients. These sources are not automatically compatible, and a general recommendation may not reflect the conditions of a particular location.

Digital agricultural advisory tools already exist in Kenya and elsewhere. The remaining practical problem is whether these heterogeneous public datasets can be integrated into a transparent, location-informed machine-learning prototype that ranks suitable crops and estimates fertilizer type and quantity for selected Kenyan agricultural areas, with performance and limits that can be evaluated against established agronomic guidance.

### 1.4 Objectives

#### 1.4.1 General Objective

To design, develop, and evaluate a machine-learning-based system that recommends suitable crops and estimates fertilizer nutrient requirements using soil and climate data for selected agricultural locations in Kenya.

#### 1.4.2 Specific Objectives

1. To acquire, clean, and integrate relevant soil, historical climate, crop suitability, and nutrient-response datasets from iSDAsoil, NASA POWER, FAO GAEZ, and OFRA.
2. To analyse relationships between soil characteristics, climate indicators, crop suitability, and crop nutrient response within the selected study area.
3. To develop and compare supervised machine learning models for predicting or ranking crop suitability using available reference labels.
4. To develop and evaluate a machine learning or optimization component for estimating crop nutrient response or fertilizer application rates where suitable labelled data are available, and to convert nutrients to product quantities where composition data exist.
5. To design a user interface that presents crop rankings, supported fertilizer recommendations, and clear explanations of data uncertainty and limitations.

### 1.5 Justification

#### 1.5.1 Agricultural and Social Justification

A location-informed advisory prototype could make technical soil and climate information easier to interpret and help farmers consider alternatives to habitual crop and fertilizer choices. The system is intended to support, not replace, agricultural extension officers or laboratory soil testing.

#### 1.5.2 Economic and Environmental Justification

Better matching of crop selection and nutrient application to local conditions has the potential to reduce unnecessary fertilizer expenditure and nutrient losses. Such benefits cannot be claimed as achieved until the recommendations have been validated through appropriate agronomic evidence or field evaluation.

#### 1.5.3 Technical and Academic Justification

The project offers a practical data science case involving geospatial data integration, exploratory analysis, feature engineering, supervised learning, model comparison, explainability, and interface development. Its academic contribution will be a reproducible evaluation of whether the selected data sources support defensible recommendations, rather than merely a working application.

### 1.6 Scope of the Study

The study will focus on a limited, to-be-confirmed agricultural area in Kenya and a manageable selection of crops for which suitable GAEZ and OFRA records exist. Starting candidates are one or two of Nakuru, Uasin Gishu, and Trans Nzoia, and maize, beans, and sorghum where the records support them. The system will use secondary datasets rather than conducting a large-scale soil sampling campaign. Key soil features may include pH, organic carbon, nitrogen, phosphorus, potassium, texture, and cation exchange capacity, subject to source availability. Climate features will include historical rainfall and temperature summaries. Crop suitability predictions will be compared with GAEZ reference information. Fertilizer modelling will depend on the actual structure and geographic applicability of OFRA response records.

The deliverable will be a prototype with data preprocessing, trained models, an evaluation report, and a basic farmer-facing interface. The project does not include real-time soil sensors, nationwide deployment, guaranteed yield prediction, commercial fertilizer purchasing, or a validated 24-month daily weather forecast.

### 1.7 Limitations of the Proposed System

- Digital soil maps provide estimates at mapped resolution and may not reflect recent changes in individual fields.
- The core datasets use different spatial scales, periods, and assumptions, which limits direct record-by-record joins.
- GAEZ suitability values are model-derived reference estimates, not independent observed harvest outcomes.
- OFRA response functions may not contain sufficient linked soil and management features for direct supervised fertilizer-rate prediction.
- Historical weather patterns and seasonal outlooks do not establish precise conditions for the next 12–24 months.
- Recommendations may require additional fertilizer composition information, local prices, and agronomic validation before practical use.
- A high classification accuracy reported on an Indian or Kaggle fertilizer or crop table does not transfer automatically to Kenyan fields.

### 1.8 Project Risks and Mitigation

| Risk | Potential effect | Mitigation |
| --- | --- | --- |
| Insufficient matching records | Weak training labels or thin regional coverage | Audit crop and Kenyan coverage before finalising the study area; narrow crops or scope. |
| Spatial or temporal mismatch | Misleading relationships | Track coordinates, resolution, soil depth, and reference period; avoid naïve joins. |
| Data leakage or overfitting | Inflated model accuracy | Use geographically separated train and test groups, baselines, and cross-validation. |
| Inaccurate nutrient estimates | Unsuitable recommendations | Compare against documented OFRA response functions and Kenya tools; label outputs as advisory. |
| API availability or access restrictions | Interrupted data collection | Cache permitted extracts and document download procedures. |
| Restricted trial data | AgWise or TAMASA records unavailable | Start on the four public sources; treat restricted trials as a later validation request. |
| Project time constraints | Incomplete evaluation | Prioritise reproducible datasets and the core models before interface enhancements. |

### 1.9 Proposed Project Schedule

Provisional 12-week schedule, subject to supervisor approval and dataset access.

| Activity | Weeks |
| --- | --- |
| Requirements and literature review | 1–2 |
| Dataset acquisition and compatibility audit | 2–4 |
| Data cleaning and exploratory analysis | 4–5 |
| Crop suitability model development | 6–7 |
| Nutrient-response model and evaluation | 7–9 |
| Prototype interface and integration | 9–10 |
| Testing, documentation, and presentation | 11–12 |

### 1.10 Budget and Resources

Planning allowances, not confirmed quotations. An existing laptop and open-source software are assumed.

| Resource | Estimated cost (KES) |
| --- | --- |
| Internet and data access | 2,000 |
| Local transport and consultation | 2,500 |
| Printing and documentation | 1,500 |
| Contingency | 2,000 |
| **Total estimated cash budget** | **8,000** |

Technical resources will include a computer, internet access, Python, pandas, scikit-learn, XGBoost if selected, raster processing tools, and a lightweight web application framework. The selected core data sources are iSDAsoil, OFRA (Dryad), FAO GAEZ, and NASA POWER. KALRO’s Land, Soil and Crop Hub, AfSIS, SoilGrids, ICPAC, TAMASA, and AgWise are supporting or validation sources. Access conditions and licensing will be checked during acquisition.

## 15. References and access links

### Datasets and platforms

- iSDA. *iSDAsoil*. https://www.isda-africa.com/isdasoil/ — developer portal: https://www.isda-africa.com/isdasoil/developer/ — FAQ and property table: https://www.isda-africa.com/isdasoil/faq/ — registration: https://www.isda-africa.com/api/registration/ — API: https://api.isda-africa.com — tutorials: https://github.com/iSDA-Africa/isdasoil-tutorial

- Wortmann, C., Milner, M., Kaizzi, K. C., Nouri, M., Cyamweshi, A. R., Dicko, M. K., Kibunja, C. N., Macharia, M., Maria, R., Nalivata, P. C., Demissie, N., Nkonde, D., Ouattara, K., Senkoro, C. J., Tarfa, B. D., & Tetteh, F. M. (2018, updated 22 August 2023). *Geo-referenced crop-nutrient response function dataset for Tropical Africa* [Dataset]. Dryad. https://doi.org/10.5061/dryad.tt6h5h1 — record: https://datadryad.org/dataset/doi:10.5061/dryad.tt6h5h1

- University of Nebraska–Lincoln, Department of Agronomy and Horticulture. *Optimized Fertilizer Recommendations in Africa (OFRA)*. https://agronomy.unl.edu/ofra/

- International Fertilizer Development Center. *Optimizing Fertilizer Recommendations in Africa (OFRA)*. https://ifdc.org/resources/ofra-analysis-tools/

- Wortmann, C. S., & Sones, K. (Eds.). (2017). *Fertilizer Use Optimization in Sub-Saharan Africa*. CAB International. ISBN 978-1-78639-204-6; e-book ISBN 978-1-78639-205-3.

- FAO and IIASA. *Global Agro-Ecological Zoning version 5 (GAEZ v5)*. Data portal: https://data.apps.fao.org/gaez/ — programme page: https://www.fao.org/gaez/en — model documentation: https://github.com/un-fao/gaezv5/wiki — platform user guide: https://github.com/un-fao/gaezv5/wiki/21.-GAEZ-v5-Platform-User-Guide — repository: https://github.com/un-fao/gaezv5 — suitability index metadata: https://data.apps.fao.org/catalog/iso/66bfe451-c6da-4edd-940e-cb4b0b83725a — Theme 4 master notes: https://data.apps.fao.org/catalog/dataset/gaez-v5-master-config/resource/31abf121-1644-4473-8c1b-bfec83112749

- NASA Langley Research Center. *POWER Project*. https://power.larc.nasa.gov/ — documentation: https://power.larc.nasa.gov/docs/ — daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/ — communities: https://power.larc.nasa.gov/docs/methodology/communities/ — API explorer: https://power.larc.nasa.gov/api/pages/ — Data Access Viewer: https://power.larc.nasa.gov/data-access-viewer/

When POWER data are used, acknowledge the NASA Langley Research Center POWER Project, funded through the NASA Earth Science Directorate Applied Science Program, following the wording in the POWER documentation.

### Background cited by the OFRA Dryad record

The Dryad abstract compares inference methods and names:

- OFRA Inference Tool, described via https://agronomy.unl.edu/ofra/
- Harvest Choice agro-ecological zones
- Global Yield Gap Atlas recommendation domains, https://www.yieldgap.org/

Those zone systems are context for how OFRA chose representative functions. They are not additional input datasets for the first prototype.

### Related maize response data

- Wortmann, C. S., Senkoro, C., Cyamweshi, A. R., Kibunja, C., Nkonde, D., Munthali, M., Nalivata, P., Nabahungu, L. N., & Kaizzi, K. (2018). *Maize-nutrient response functions for Eastern and Southern Africa* [Dataset]. Dryad. https://doi.org/10.5061/dryad.fg15tg2 — https://datadryad.org/dataset/doi:10.5061/dryad.fg15tg2

### Soil data behind GAEZ, for interpretation

GAEZ soil constraints use the Harmonized World Soil Database, not iSDAsoil. HWSD v2 is documented by FAO and IIASA (2023), as cited in the GAEZ v5 Theme 4 metadata: https://data.apps.fao.org/catalog/dataset/gaez-v5-master-config/resource/31abf121-1644-4473-8c1b-bfec83112749

Historical GAEZ climate uses Copernicus AgERA5 for 1981–2000 and 2001–2020, as described in the suitability-index metadata: https://data.apps.fao.org/catalog/iso/66bfe451-c6da-4edd-940e-cb4b0b83725a

### Kenya tools, KALRO, and neighbouring research

- Kenya OFRA chapter: Wortmann and co-authors, Chapter 7, *Optimizing Fertilizer Use within the Context of Integrated Soil Fertility Management in Kenya*, in Wortmann & Sones (Eds.). PDF: https://ifdc.org/wp-content/uploads/2026/01/OFRA-Chapter-7_Kenya_HR.pdf — zone tools: https://ifdc.org/resources/ofra-analysis-tools/

- KALRO. Kenya Digital Agriculture Platform, including KALRO Selector: https://keep.kalro.org/ — Kenya Agricultural Observatory Platform: https://kaop.co.ke/ — Land, Soil and Crop Hub: https://lschub.kalro.org/ — hub user guide: https://lschub.kalro.org/docs/

- Alliance of Bioversity International and CIAT, IITA, and KALRO. AgWise site-specific nutrient work in Kenya. Partnership account: https://alliancebioversityciat.org/stories/tailored-fertilizer-advice-kenya-diverse-soils-crops — data harmonisation account: https://alliancebioversityciat.org/stories/scattered-agronomic-data-smarter-fertilizer-management-maize-productivity-kenya — methods listing: https://alliancebioversityciat.org/publications-data/scattered-data-site-specific-nutrient-recommendation-systems-data-driven-approach — Degefie, T., Bendito, E. G., Assefa, F., Kinyua, M., Mkuhlani, S., Abera, W., Kihara, J., & Ghosh, A. (2025). *AgWise based site-specific and climate smart fertilizer advisory for long rains maize in Western Kenya*. https://alliancebioversityciat.org/publications-data/agwise-based-site-specific-and-climate-smart-fertilizer-advisory-long-rains-maize — module: https://github.com/CGIAR-AgWise/agwise-fertilizer

- ISRIC, KALRO, Ministry of Agriculture and Livestock Development, and CABI. Kenya Soil Information System roadmap. https://isric.org/news/strengthening-the-soil-information-system-sis-in-kenya-a-collaborative-roadmap/ — https://zenodo.org/records/17696631

- Government of Kenya SLEEK programme and the Full Lands Integration Tool. https://moja.global/the-flint-in-kenya/ — https://moja.global/flint/ — https://moja.global/finally-sleek-flint-is-running-in-kenya/

### Other soil, climate, and trial sources

- Hengl, T., Miller, M. A. E., Križan, J., et al. (2021). African soil properties and nutrients mapped at 30 m spatial resolution using two-scale ensemble machine learning. *Scientific Reports, 11*, 6130. https://doi.org/10.1038/s41598-021-85639-y

- ISRIC. *SoilGrids*. https://soilgrids.org/ — documentation: https://docs.isric.org/globaldata/soilgrids/

- Hengl, T., Heuvelink, G. B. M., Kempen, B., Leenaars, J. G. B., Walsh, M. G., Shepherd, K. D., Sila, A., MacMillan, R. A., Mendes de Jesus, J., Tamene, L., & Tondoh, J. E. (2015). Mapping soil properties of Africa at 250 m resolution: Random forests significantly improve current predictions. *PLOS ONE, 10*(6). Project page: https://isric.org/projects/soil-property-maps-africa-250-m-resolution

- Africa Soil Information Service. *AfSIS soil chemistry* (Phase I), Registry of Open Data on AWS. https://registry.opendata.aws/afsis/

- ICPAC. IGAD Climate Prediction and Applications Centre. https://www.icpac.net/

- CIMMYT. *TAMASA data*. https://tamasa.cimmyt.org/tools-data/tamasa-data/ — example Nigeria nutrient-omission set: https://data.cimmyt.org/dataset.xhtml?persistentId=hdl:11529/10548310

- IITA data repository: https://data.iita.org/

### Machine-learning comparison studies

- Gimba, A. M., & Mishra, P. K. (2025). Leveraging ensemble learning techniques for efficient fertilizer recommendation. *JITSI: Jurnal Ilmiah Teknologi Sistem Informasi*. Published 31 December 2025. https://jurnal-itsi.org/index.php/jitsi/article/view/509 — fertilizer-type classification for Andhra Pradesh using N, P, K, and moisture; reported CatBoost accuracy 94.78% on that dataset.

- Dey, B., & Ahmed, R. (2024). Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables. *Heliyon, 10*, e25112. https://doi.org/10.1016/j.heliyon.2024.e25112 — crop recommendation, not a fertilizer-rate model.
