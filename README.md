# final_size_WHO

A WHO influenza surveillance and forecasting project analyzing seasonal patterns and predicting US state hospitalizations from Southern Hemisphere countries data.

## Data Files

The `data/` folder contains raw surveillance and hospitalization data from multiple sources:

### WHO FluNet Data
- **`VIW_FNT.csv`**: WHO FluNet surveillance data containing weekly influenza laboratory confirmations by country, including specimen counts, virus subtypes (A/H1N1, A/H3N2, B/Victoria, B/Yamagata), and respiratory virus co-detections. Data spans multiple years and hemispheres.
- **`VIW_FLU_METADATA.csv`**: Metadata describing the FluNet dataset fields, data types, and variable definitions.

### Target Hospital Data (`target-data/` subfolder)
- **`target-hospital-admissions.csv`**: Primary hospital admission data from CDC's NHSN (National Healthcare Safety Network) containing weekly confirmed influenza hospitalizations by US state/territory
- **`target-hospital-admissions copy.csv`** and **`target-hospital-admissions copy 2.csv`**: Backup copies of hospital admission data
- **`get_target_data.R`**: R script that fetches the latest hospital admission data from CDC's data portal via RSocrata API and processes it into the target format
- **`README.md`**: Comprehensive documentation explaining the hospital admission data sources, processing methods, data quality considerations, and access methods

## Analysis Data Files

### Python Scripts

#### `analysis_data/format_hosp_data.py`
Processes raw hospital admission data and formats it for analysis:
- **Input**: `./data/target-data/target-hospital-admissions.csv`
- **Output**: `./analysis_data/us_hospital_data.csv`
- **Functionality**:
  - Converts dates to MMWR (Morbidity and Mortality Weekly Report) week format
  - Assigns flu seasons (e.g., 2021/2022) based on MMWR weeks (season starts week 40, ends week 30)
  - Filters out off-season data (weeks 31-39)
  - Adds sequential model week numbers within each season/location
  - Organizes data by location, season, and epidemiological week

### Data Files

- **`season_level_data.csv`**: Aggregated hospital data at the seasonal level
- **`us_hospital_data.csv`**: Formatted US hospital admission data with MMWR weeks and seasons
- **`week_country_level_data.csv`**: Weekly surveillance data by country
- **`week_level_data.csv`**: Weekly aggregated surveillance data

## Predicting State from Southern Hemisphere Countries

This analysis explores the relationship between Southern Hemisphere (SH) influenza patterns and US state hospitalizations, leveraging the fact that SH flu seasons precede Northern Hemisphere seasons by ~6 months.

### Python Scripts

#### `models/predicting_state_from_SH_countries/from_analysis_data_to_regression_datasets.py`
Creates normalized datasets for regression analysis:
- **Inputs**: 
  - `./analysis_data/us_hospital_data.csv`
  - `./analysis_data/week_country_level_data.csv`
- **Outputs**:
  - `normalized_US_hosp_and_SH_WHO_cases.csv`: Z-score normalized data
  - `un_normalized_US_hosp_and_SH_WHO_cases.csv`: Raw values
- **Functionality**:
  - Aggregates total hospitalizations by US state and season
  - Maps Northern Hemisphere seasons to corresponding Southern Hemisphere seasons
  - Normalizes both hospitalization and case proportion data using z-scores
  - Filters for complete cases and removes low-variability countries (e.g., Indonesia)
  - Merges US hospitalization data with SH country surveillance data

#### `models/predicting_state_from_SH_countries/pearsons_correlation.py`
Computes correlation analysis between SH countries and US states:
- **Input**: `normalized_US_hosp_and_SH_WHO_cases.csv`
- **Output**: `pearsons_correlation_between_SH_countries_and_US_state_hosps.csv`
- **Functionality**:
  - Separates US state data (numeric location codes) from SH country data
  - Calculates Pearson correlation coefficients between each US state and each SH country
  - Creates a comprehensive correlation matrix for identifying predictive relationships

### Data Files

- **`normalized_US_hosp_and_SH_WHO_cases.csv`**: Z-score normalized hospitalization and case data
- **`un_normalized_US_hosp_and_SH_WHO_cases.csv`**: Raw hospitalization and case data for reference
