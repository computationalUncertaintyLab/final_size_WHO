#mcandrew

import pandas as pd
import numpy as np
from scipy import interpolate

def compute_pit_from_quantiles(quantile_levels, predicted_values, observed_value):
    """
    Compute the Probability Integral Transform (PIT) value from quantile forecasts.
    
    The PIT is the value of the predictive CDF at the observed value.
    For a perfectly calibrated forecast, PIT values should be uniformly distributed.
    
    Parameters:
    -----------
    quantile_levels : array-like
        Quantile levels (e.g., [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    predicted_values : array-like
        Predicted values at each quantile level
    observed_value : float
        The actual observed value
    
    Returns:
    --------
    float : PIT value between 0 and 1
    """
    
    # Sort by quantile level to ensure proper ordering
    sort_idx = np.argsort(quantile_levels)
    quantile_levels = np.array(quantile_levels)[sort_idx]
    predicted_values = np.array(predicted_values)[sort_idx]
    
    # Handle edge cases
    if observed_value <= predicted_values[0]:
        # Observed value is below the lowest prediction
        # PIT is proportional to how far below
        # Use linear extrapolation from first two quantiles
        if len(quantile_levels) > 1:
            slope = (quantile_levels[1] - quantile_levels[0]) / (predicted_values[1] - predicted_values[0])
            pit = max(0.0, quantile_levels[0] - slope * (predicted_values[0] - observed_value))
        else:
            pit = quantile_levels[0] / 2  # Rough estimate
        return pit
    
    if observed_value >= predicted_values[-1]:
        # Observed value is above the highest prediction
        # Use linear extrapolation from last two quantiles
        if len(quantile_levels) > 1:
            slope = (quantile_levels[-1] - quantile_levels[-2]) / (predicted_values[-1] - predicted_values[-2])
            pit = min(1.0, quantile_levels[-1] + slope * (observed_value - predicted_values[-1]))
        else:
            pit = (1.0 + quantile_levels[-1]) / 2  # Rough estimate
        return pit
    
    # Observed value is within the range of predictions
    # Interpolate to find the CDF value at the observed point
    pit = np.interp(observed_value, predicted_values, quantile_levels)
    
    return float(pit)


if __name__ == "__main__":
    
    print("="*60)
    print("COMPUTING PIT SCORES FROM QUANTILE FORECASTS")
    print("="*60)
    
    # Read the forecast data with quantiles, predictions, and observations
    print("\nReading forecast data...")
    forecasts = pd.read_csv("./forecast_experiment/evaluation/forecasts_scoringutils_format.csv")
    print(f"Loaded {len(forecasts)} forecast rows (including all quantiles)")
    
    # Define forecast identifiers
    forecast_identifiers = ['location', 'forecast_date', 'target_end_date', 'horizon', 'model', 'season']
    
    # Add any other metadata we want to preserve
    metadata_cols = ['target', 'latest_MMWR'] if 'latest_MMWR' in forecasts.columns else ['target']
    metadata_cols = [col for col in metadata_cols if col in forecasts.columns]
    
    print(f"\nForecast identifiers: {forecast_identifiers}")
    print(f"Additional metadata: {metadata_cols}")
    
    # Group by unique forecasts and compute PIT for each
    print("\nComputing PIT scores...")
    
    pit_scores = []
    n_forecasts = 0
    n_errors = 0
    
    grouped = forecasts.groupby(forecast_identifiers)
    total_groups = len(grouped)
    
    for i, (forecast_id, group) in enumerate(grouped):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total_groups} forecasts...")
        
        try:
            # Extract quantile levels and predictions
            quantile_levels = group['quantile_level'].values
            predicted_values = group['predicted'].values
            
            # Get observed value (should be same for all quantiles in this group)
            observed_value = group['observed'].iloc[0]
            
            # Check if observed value is NaN
            if pd.isna(observed_value):
                continue
            
            # Compute PIT
            pit_value = compute_pit_from_quantiles(quantile_levels, predicted_values, observed_value)
            
            # Create result dictionary
            result = dict(zip(forecast_identifiers, forecast_id))
            
            # Add metadata
            for col in metadata_cols:
                result[col] = group[col].iloc[0]
            
            # Add observed value and PIT
            result['observed'] = observed_value
            result['pit_value'] = pit_value
            
            pit_scores.append(result)
            n_forecasts += 1
            
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:  # Only print first few errors
                print(f"  Error computing PIT for forecast {forecast_id}: {e}")
    
    print(f"\n✓ Successfully computed PIT for {n_forecasts} forecasts")
    if n_errors > 0:
        print(f"  {n_errors} forecasts had errors and were skipped")
    
    # Convert to DataFrame
    pit_df = pd.DataFrame(pit_scores)
    
   
    # Check for values outside [0, 1]
    outside_range = ((pit_df['pit_value'] < 0) | (pit_df['pit_value'] > 1)).sum()
    if outside_range > 0:
        print(f"\n⚠ WARNING: {outside_range} PIT values are outside [0, 1] range")
    
    # Uniformity test hint
    
    # Display quantiles of PIT distribution
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\nPIT distribution quantiles:")
    for q in quantiles:
        print(f"  {q:.2f}: {pit_df['pit_value'].quantile(q):.3f}")
    
    # Save results
    output_file = "./forecast_experiment/evaluation/pit_scores.csv"
    pit_df.to_csv(output_file, index=False)
    
    print(f"\n✓ PIT scores saved to: {output_file}")
    
    # Display sample
    print("\nSample of PIT scores (first 10 rows):")
    print(pit_df.head(10))
    
