#mcandrew

import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    print("="*60)
    print("MERGING SCORED FORECASTS WITH OBSERVED VALUES")
    print("="*60)
    
    # Read the scored forecasts from R scoringutils
    print("\nReading scored forecasts from scoringutils...")
    scores = pd.read_csv("./forecast_experiment/evaluation/forecast_scores.csv")
    print(f"Loaded {len(scores)} scored forecasts")
    print(f"Columns in scores: {list(scores.columns)}")
    
    # Read the original prepared data that has observed values
    print("\nReading original data with observed values...")
    forecasts_with_obs = pd.read_csv("./forecast_experiment/evaluation/forecasts_scoringutils_format.csv")
    print(f"Loaded {len(forecasts_with_obs)} forecast rows")
    
    # The original data has one row per quantile, but scores has one row per unique forecast
    # We need to get just the unique forecast identifier + observed value
    
    # Define the forecast identifiers (should match scoringutils forecast_unit)
    forecast_identifiers = ['location', 'forecast_date', 'target_end_date', 'horizon', 'model', 'season']
    
    # Get unique forecasts with their observed values
    # Each unique forecast has the same observed value across all quantiles
    unique_forecasts_with_obs = forecasts_with_obs[forecast_identifiers + ['observed']].drop_duplicates()
    
    print(f"\nUnique forecasts with observed values: {len(unique_forecasts_with_obs)}")
    
    # Check if scores already has observed column
    if 'observed' in scores.columns:
        print("\nNote: Scores already contains 'observed' column. Will overwrite with merge.")
        scores = scores.drop(columns=['observed'])
    
    # Merge scores with observed values
    print("\nMerging scores with observed values...")
    scores_with_obs = scores.merge(
        unique_forecasts_with_obs,
        on=forecast_identifiers,
        how='left'
    )
    
    print(f"Merged dataset has {len(scores_with_obs)} rows")
    
    # Check for any missing observed values after merge
    missing_obs = scores_with_obs['observed'].isna().sum()
    if missing_obs > 0:
        print(f"\nWARNING: {missing_obs} rows have missing observed values after merge")
    else:
        print("\nSuccess! All scored forecasts have observed values")
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY OF MERGED DATA")
    print("="*60)
    print(f"Total rows: {len(scores_with_obs)}")
    print(f"Total columns: {len(scores_with_obs.columns)}")
    print(f"\nColumns: {list(scores_with_obs.columns)}")
    
    # Save the merged dataset
    output_file = "./forecast_experiment/evaluation/forecast_scores_with_observations.csv"
    scores_with_obs.to_csv(output_file, index=False)
    
    print(f"\n✓ Merged scores saved to: {output_file}")
    