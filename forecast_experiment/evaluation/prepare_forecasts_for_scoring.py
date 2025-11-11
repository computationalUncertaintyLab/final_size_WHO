#mcandrew

import pandas as pd
import numpy as np
import re

def format_location(x):
    if x=="US":
        return x
    return "{:02d}".format(int(x))

def extract_season_from_filename(filename):
    """Extract season from filename pattern: forecast_XX_YYYY_ZZZZ__WW.csv"""
    match = re.search(r'_(\d{4})_(\d{4})__', filename)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return None

if __name__ == "__main__":

    # Read control arm forecasts
    control_arm_forecasts = pd.read_csv("./forecast_experiment/control_arm/control_arm_forecasts.csv")
    control_arm_forecasts["location"] = [format_location(x) for x in control_arm_forecasts.location.values]
    
    # Extract season if not already present in the data
    # Assuming season information might be in a column or needs to be inferred
    # If season column doesn't exist, we can extract it from reference_date or other means
    if "season" not in control_arm_forecasts.columns:
        # Add season based on reference_date (MMWR weeks: season starts at week 40, ends at week 30)
        control_arm_forecasts["reference_date"] = pd.to_datetime(control_arm_forecasts["reference_date"])
        
        def get_season_from_date(date):
            year = date.year
            month = date.month
            # If month is Oct-Dec (MMWR week 40+), season is current_year/next_year
            if month >= 10:
                return f"{year}/{year+1}"
            # If month is Jan-Jul (MMWR week 1-30), season is prev_year/current_year
            elif month <= 7:
                return f"{year-1}/{year}"
            # Aug-Sep is off-season, but we'll assign to next season starting in Oct
            else:
                return f"{year}/{year+1}"
        
        control_arm_forecasts["season"] = control_arm_forecasts["reference_date"].apply(get_season_from_date)

    # Read true values
    true_values = pd.read_csv("./data/target-data/target-hospital-admissions.csv")
    true_values = true_values.rename(columns={"value":"obs"})
    true_values["location"] = [format_location(x) for x in true_values.location.values]

    # Merge forecasts with truth
    forecasts_and_truth = control_arm_forecasts.merge(
        true_values, 
        left_on=["location","target_end_date"], 
        right_on=["location","date"]
    )

    # Save original merged format
    forecasts_and_truth.to_csv("./forecast_experiment/evaluation/forecasts_and_truth.csv", index=False)
    
    print(f"Original merged data saved. Shape: {forecasts_and_truth.shape}")
    print(f"Columns: {list(forecasts_and_truth.columns)}")

    # ===== PREPARE DATA FOR SCORINGUTILS =====
    
    # Create a copy for scoringutils transformation
    scoringutils_data = forecasts_and_truth.copy()
    
    # Rename columns to match scoringutils format
    column_mapping = {
        'reference_date': 'forecast_date',
        'output_type_id': 'quantile_level',
        'value': 'predicted',
        'obs': 'observed'
    }
    
    scoringutils_data = scoringutils_data.rename(columns=column_mapping)
    
    # Convert quantile_level to numeric float
    scoringutils_data['quantile_level'] = scoringutils_data['quantile_level'].astype(float)
    
    # Add model column
    scoringutils_data['model'] = 'control_arm'
    
    # Filter out rows with missing observed values
    n_before = len(scoringutils_data)
    scoringutils_data = scoringutils_data.dropna(subset=['observed'])
    n_after = len(scoringutils_data)
    print(f"Filtered out {n_before - n_after} rows with missing observed values")
    
    # Select and order columns for scoringutils
    # Keep all temporal information and identifiers
    columns_to_keep = [
        'model',
        'location',
        'season',
        'forecast_date',
        'target_end_date',
        'horizon',
        'latest_MMWR',
        'target',
        'quantile_level',
        'predicted',
        'observed'
    ]
    
    # Only keep columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in scoringutils_data.columns]
    scoringutils_data = scoringutils_data[columns_to_keep]
    
    # Ensure dates are in proper format
    scoringutils_data['forecast_date'] = pd.to_datetime(scoringutils_data['forecast_date'])
    scoringutils_data['target_end_date'] = pd.to_datetime(scoringutils_data['target_end_date'])
    
    # Save scoringutils-ready format
    output_file = "./forecast_experiment/evaluation/forecasts_scoringutils_format.csv"
    scoringutils_data.to_csv(output_file, index=False)
    
    print(f"\nScoringutils-ready data saved to: {output_file}")
    print(f"Shape: {scoringutils_data.shape}")
    print(f"Columns: {list(scoringutils_data.columns)}")
    print(f"\nSample of data:")
    print(scoringutils_data.head(10))
    print(f"\nUnique seasons: {sorted(scoringutils_data['season'].unique())}")
    print(f"Unique locations: {sorted(scoringutils_data['location'].unique())}")
    print(f"Horizon range: {scoringutils_data['horizon'].min()} to {scoringutils_data['horizon'].max()}")
    print(f"Quantile levels: {sorted(scoringutils_data['quantile_level'].unique())}")

