#mcandrew

import sys
sys.path.append('./forecast_experiment/model/tempo/')
from tempo_model import tempo_model4

import numpy as np
import pandas as pd

from pathlib import Path

from joblib import Parallel, delayed

def format_counts(d,interp=False):
    weeks = list(np.arange(40,53+1)) + list(np.arange(1,20+1)) 

    for week in weeks:
        if week not in d.columns:
            d[ week ] = np.nan
    d = d[weeks]

    if interp:
        d_counts = []
        for row in d.to_numpy():
            d_counts.append(interpolate_nans(row))
        d_counts = np.array(d_counts)
    else:
        d_counts = d.to_numpy()
    return d_counts



def add_time_data(row):
    from epiweeks import Week
    from datetime import datetime

    epiweek = Week.fromdate( datetime.strptime(row.Week,"%Y-%m-%d"))
    row["MMWRYR"] = epiweek.year
    row["MMWRWK"] = epiweek.week
    row["season"] = "{:d}/{:d}".format(epiweek.year,epiweek.year+1) if epiweek.week >=40 else "{:d}/{:d}".format(epiweek.year-1,epiweek.year)

    return row

def interpolate_nans(array):
    """
    Linearly interpolates NaN values in a 1D NumPy array.
    For leading/trailing NaNs, it performs forward/backward filling.
    """
    nans = np.isnan(array)
    # Create an array of indices for the original array
    x = np.arange(len(array))
    
    # Use np.interp to fill NaNs
    # x=x[nans]: Indices where NaNs are present (where we want to interpolate)
    # xp=x[~nans]: Indices where non-NaN values are present (known points)
    # fp=array[~nans]: Values at the non-NaN indices (known values)
    array[nans] = np.interp(x=x[nans], xp=x[~nans], fp=array[~nans])
    
    return array



def add_epitime(row):
    
    def from_time_to_season(x, yrstop):
        yr,week = x.year, x.week

        if yr==yrstop and week>=35:
            season = "-1"
            return season

        if week>20 and week<35:
            season="-1"
        else:
            if week>=40:
                season = "{:d}/{:d}".format( yr, yr+1)
            else:
                season = "{:d}/{:d}".format( yr-1, yr)
        return season

    from datetime import datetime
    from epiweeks import Week

    week = Week.fromdate(datetime.strptime(row.date,"%Y-%m-%d"))

    season = from_time_to_season(week,2025)

    row["season"]   = season
    row["CDCDATE"]  = week.cdcformat()
    row["MMWRYR"]   = week.year
    row["MMWRWK"]   = week.week

    return row



if __name__ == "__main__":

    THIS_SEASON = "2025/2026"
    
    #--data set of populations (contains all FIPS)
    pops                = pd.read_csv("./data/locations.csv")
    
    #--incident hospitalizations dataset
    inc_hosps           = pd.read_csv("./data/target-data/target-hospital-admissions.csv")

    #--subset by only information after 09-01
    inc_hosps           = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]
    inc_hosps = inc_hosps.apply(add_epitime,1)
    
    seasons = ["2021/2022","2022/2023","2023/2024","2024/2025", "2025/2026"]
    seasons             = ["2025/2026"]

    #--ILI data
    ili_data            = pd.read_csv("./analysis_data/ili_data_all_states_2021_present__formatted.csv")
    ili_data["week"]    = [ int(str(x)[-2:]) for x in ili_data.epiweek]

    ili_data             = ili_data.rename(columns = {"state_fips":"location","week":"MMWRWK","year":"MMWRYR"})
    ili_data["location"] = [ np.nan if np.isnan(x) else "{:02d}".format(int(x)) for x in ili_data.location.values]
    
    #------
    from_season_to_number = { season:n for n,season in enumerate(sorted(inc_hosps.season.unique())) }

    all_params = pd.read_csv("./forecast_experiment/historical_model_run_for_tempo/all_past_param_estimates__tempo4.csv")

    seasons = ["2021/2022","2022/2023","2023/2024","2024/2025"]
   
    #------
    from_season_to_number = { season:n for n,season in enumerate(sorted(inc_hosps.season.unique())) }
   
    def build_parameter_data( location,  subset ):
       
        import os 
        fstring = "./forecast_experiment/historical_model_run_for_tempo/arxiv__tempo4/params_{:s}.csv".format(location)
        if os.path.exists(fstring):
            return 
        print(location)
        
        param_data = {"location":[],"season":[],"param_type":[], "param1":[],"param2":[],"value":[]}

        state_ili = ili_data.loc[(ili_data.location==location) ].drop_duplicates()
        state_ili = state_ili.loc[(state_ili.MMWRWK>=40) | (state_ili.MMWRWK <=20)]

        state_ili = state_ili.loc[ state_ili.season.isin(subset.season) ]
        
        def format_counts(d,interp=False):
            d.columns = [y for x,y in d.columns]
            
            weeks = list(np.arange(40,53+1)) + list(np.arange(1,20+1)) 

            for week in weeks:
                if week not in d.columns:
                    d[ week ] = np.nan
            d = d[weeks]

            if interp:
                d_counts = []
                for row in d.to_numpy():
                    d_counts.append(interpolate_nans(row))
                d_counts = np.array(d_counts)
            else:
                d_counts = d.to_numpy()
            return d_counts
        N                = pd.pivot_table(index=["season"],columns = ["MMWRWK"],values=["num_patients"], data = state_ili)
        ttl_flu_         = pd.pivot_table(index=["season"],columns = ["MMWRWK"],values=["value"], data = subset)

        N       = format_counts(N, interp=True)
        ttl_flu = format_counts(ttl_flu_)
        
        import jax 
        base_key    = jax.random.PRNGKey(20200320)
        worker_key  = jax.random.fold_in(base_key, 1)
 
        model = tempo_model4( y = (ttl_flu+1.), X=None, N = N, key = worker_key ).fit_past_seasons()

        #--load up parameter data--
        for season_number,season in enumerate(ttl_flu_.index):
            prior_matrix = np.array([])
            for param in ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season"] :
                prior_vector = np.array(model[param][:,season_number])
                prior_vector = prior_vector.flatten()
                prior_matrix = np.vstack([prior_matrix, prior_vector]) if prior_matrix.size > 0 else prior_vector

            # #--treat F as a set of ncov variables F1,F2, etc
            # for n,prior_vector in enumerate(model["F_season"][:,season_number].T):
            #     prior_vector = prior_vector.flatten()
            #     prior_matrix = np.vstack([prior_matrix, prior_vector]) if prior_matrix.size > 0 else prior_vector

            mu  = prior_matrix.mean(1)
            cov = (prior_matrix - mu.reshape(-1,1))
            cov = (cov @ cov.T) / prior_matrix.shape[-1]

            #--unroll data into dict
            param_names = ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season"]#,"F1_season","F2_season","F3_season","F4_season"]
            for param, param_mu in zip(param_names,mu):
                param_data["location"].append(location)
                param_data["season"].append(season)
                param_data["param_type"].append("mu")
                param_data["param1"].append(param)
                param_data["param2"].append(param)
                param_data["value"].append(param_mu)

            for row,param1 in zip(cov,param_names):
                for param_cov,param2 in zip(row,param_names):
                    param_data["location"].append(location)
                    param_data["season"].append(season)
                    param_data["param_type"].append("cov")
                    param_data["param1"].append(param1)
                    param_data["param2"].append(param2)
                    param_data["value"].append(param_cov)

        param_data = pd.DataFrame(param_data)
        param_data.to_csv(fstring)

    inc_hosps = inc_hosps.loc[inc_hosps.season!="-1"]


    def tryit(location,subset):
        try:
            build_parameter_data(location,subset)
        except:
           print("Fail")
           print(location)
    Parallel(n_jobs=20)( delayed(tryit)(location,subset) for location, subset in inc_hosps.groupby("location") )
