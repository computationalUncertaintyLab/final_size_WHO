#mcandrew 

import sys
sys.path.append("./models/tempo/")
from tempo_model import   tempo_model2#forecast_fit, preseason_fit

import pandas as pd
import numpy  as np
import pickle

from datetime import datetime
from epiweeks import Week

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="ili")
    args = parser.parse_args()

    target = args.target

    d   = pd.read_csv("./analysis_data/weekly_data.csv")
    d   = d.drop(columns = ["N"] )
    print(d.shape)
    
    ili = pd.read_csv("./analysis_data/influenza_like_illness.csv")
    print(ili.shape)

    ili = ili.drop(columns = ["season","season_week","semester"])
    d   = d.merge( ili, on = ["MMWR_YR","MMWR_WK"] )

    d["season"] = d.season.fillna("unk")

    d = d.loc[d.season!="unk"]

    nseasons = len(d.season.unique())
    nweeks   = int(len(d)/(nseasons))

    print(nseasons)
    print(nweeks)

    #--week by cases and depth is season
    if target == "ili":
        cases = np.clip( np.array(d["ILI"]).reshape(nseasons,nweeks), 0.1, np.inf)
    else:
        cases = np.clip( np.array(d["pos_cases"]).reshape(nseasons,nweeks), 0.1, np.inf)
    
    print(cases.shape)
    N     = (np.array(d["N"]).reshape(nseasons,nweeks)).astype(float)
    # Convert semester strings to integers
    semester_mapping = {
        'Fall': 0,
        'Winter_break': 1, 
        'Spring': 2,
        'Spring_break': 3,
        'Summer': 4
    }
    
    # Convert season strings to integers
    unique_seasons = sorted(d["season"].unique())
    season_mapping = {season: i for i, season in enumerate(unique_seasons)}
    S = np.array([season_mapping[season] for season in d["season"]])
    
    for (row,col) in np.argwhere(N==-1):
        cases[row,col] = np.nan
        N[row,col]     = np.nan

    #--Model the numberof visits to the HWC--------------------------------------
    from scipy.stats import gaussian_kde
    N_per_week = []
    for ns in N.T:
        ns = ns[~np.isnan(ns)]
        kde = gaussian_kde(np.log(ns+1))
        N_per_week.extend( np.exp(kde.resample(1500)))
    N_per_week = np.array(N_per_week).T

    #-------------------------------------------------------------------------------

    #--Fit the percent of those with lab-confirmed flu-----------------------------------------------------
    nobs = np.min(np.where(np.isnan(cases[-1,:]))[0]) if np.any(np.isnan(cases[-1,:])) else nweeks
   
    preseason_single_season_models = []
    for season in range(nseasons-1):
        tempo = tempo_model2(y=cases[season].reshape(1,nweeks), N=N[season].reshape(1,nweeks), nobs=nweeks)
        preseason_single_season_models.append(tempo.fit_past_seasons())
   
    prior_tensor = []
    for season in range(nseasons-1):
        model = preseason_single_season_models[season]
        prior_matrix = np.array([])
        for param in ["K","M_mu", "B_mu","B2_mu","nu_mu","Q_mu","transition_width"]:
            prior_vector = np.array(model[param])
            prior_vector = prior_vector.flatten()
            prior_matrix = np.vstack([prior_matrix, prior_vector]) if prior_matrix.size > 0 else prior_vector
        prior_tensor.append(prior_matrix)
    prior_tensor = np.array(prior_tensor)

    prior_mus  = []
    prior_covs = []
    for season in range(nseasons-1):
        prior_param_data = prior_tensor[season,...]
        mu  = prior_param_data.mean(1)

        cov = (prior_param_data - mu.reshape(-1,1))
        cov = (cov @ cov.T) / prior_param_data.shape[-1]

        prior_mus.append(mu)
        prior_covs.append(cov)
    prior_mus  = np.array(prior_mus)
    prior_covs = np.array(prior_covs)
    

    tempo          = tempo_model2(y= cases[-1].reshape(1,nweeks), N= N[-1].reshape(1,nweeks), nobs=nweeks)# np.nan*np.ones(nweeks).reshape(1,nweeks), nobs=0)
    forecast_model = tempo.fit_new_season(prior_mus=prior_mus, prior_covs = prior_covs, forecast=True, N_pred = N_per_week)
    
    forecast_samples        = forecast_model["inc_pred"]
    #forecast_samples_smooth = forecast_model["inc_smooth"]

    import jax
    import numpyro.distributions as dist
    forecast_cases_samples = dist.Binomial(total_count=N_per_week.astype(int), probs=forecast_samples).sample(jax.random.PRNGKey(42))

    #---create percentiles dataset------------------------------------------------
    #--Load academic calendar for 2025-2026
    CURRENT_SEASON = "2025/26"
    #calendar = pd.read_csv("./analysis_data/lehigh_academic_calendar_week_mappings.csv")
    calendar = pd.read_csv("./analysis_data/from_week_to_season_week.csv")
    calendar = calendar.loc[calendar.season == CURRENT_SEASON]

    #--read percentiles
    percentiles = pd.read_csv("./models/helper_files/percentiles_for_forecasts.csv")
    percentiles = percentiles["percentile"].values
    
    #--Compute percentiles for each week for both cases and incidence
    percentile_values_cases = np.percentile(forecast_cases_samples, [p * 100 for p in percentiles], axis=0)
    percentile_values_inc   = np.percentile(  forecast_samples    , [p * 100 for p in percentiles], axis=0)
    
    #--Create forecasted_percentiles dataset
    forecasted_percentiles = []

    for (idx, row), pcases, pincs in zip(calendar.iterrows(), percentile_values_cases.T, percentile_values_inc.T):
        week_info = row
        
        for pcase, pinc, percentile in zip(pcases, pincs, percentiles):
            forecasted_percentiles.append({
                'season'                : week_info['season'],
                'semester'              : week_info['semester'],
                'MMWR_YR'               : week_info['MMWR_YR'],
                'MMWR_WK'               : week_info['MMWR_WK'],
                'percentile'            : percentile,
                'percentile_value_cases': pcase,
                'percentile_value_inc'  : pinc
            })
    forecasted_percentiles_df = pd.DataFrame(forecasted_percentiles)
    
    # Create filename with today's date
    this_week   = Week.thisweek()
    end_of_week = this_week.enddate().strftime("%Y-%m-%d")
    
    # Save to forecasts folder
    if target == "ili":
        forecasted_percentiles_df.to_csv(f"./forecasts/2025_26/tempo/ili/{end_of_week}_tempo_forecast_ili.csv", index=False)
        pickle.dump(forecast_cases_samples, open(f"./forecasts/2025_26/tempo/ili/{end_of_week}_tempo_ili_cases_samples.pkl", "wb"))
        pickle.dump(forecast_samples, open(f"./forecasts/2025_26/tempo/ili/{end_of_week}_tempo_ili_inc_samples.pkl", "wb"))

    else:
        forecasted_percentiles_df.to_csv(f"./forecasts/2025_26/tempo/flu/{end_of_week}_tempo_forecast_flu.csv", index=False)
        pickle.dump(forecast_cases_samples, open(f"./forecasts/2025_26/tempo/flu/{end_of_week}_tempo_flu_cases_samples.pkl", "wb"))
        pickle.dump(forecast_samples, open(f"./forecasts/2025_26/tempo/flu/{end_of_week}_tempo_flu_inc_samples.pkl", "wb"))



        
