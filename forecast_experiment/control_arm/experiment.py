#mcandrew

import sys
sys.path.append('./forecast_experiment/model/tempo/')
from tempo_model import tempo_model4

import numpy as np
import pandas as pd

from pathlib import Path

from epiweeks import Week

from joblib import Parallel, delayed

import os

def add_time_data(row):
    from epiweeks import Week
    from datetime import datetime

    epiweek = Week.fromdate( datetime.strptime(row.Week,"%Y-%m-%d"))
    row["MMWRYR"] = epiweek.year
    row["MMWRWK"] = epiweek.week
    row["season"] = "{:d}/{:d}".format(epiweek.year,epiweek.year+1) if epiweek.week >=35 else "{:d}/{:d}".format(epiweek.year-1,epiweek.year)

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
            if week>=35:
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

    def format(x):
        if x=="US":
            return x
        return "{:02d}".format(int(x))
    all_params["location"] = [format(x) for x in all_params.location.values]

    def forecast( location, season, subset ):
        print(location)

        
        
        #param_data = {"location":[],"season":[],"param_type":[], "param1":[],"param2":[],"value":[]}

        #--subset all data to specific state
        #season = "2025/2026"
    
        #thisweek = Week.thisweek().enddate().strftime("%Y-%m-%d")
        
        latest_MMWR = subset.MMWRWK.max()
        if location=="US":
           forecast_file  = "./forecast_experiment/control_arm/forecasts/forecast_US_{:02d}.csv".format(latest_MMWR)
        else:
            forecast_file = "./forecast_experiment/control_arm/forecasts/forecast_{:02d}_{:02d}.csv".format( int(location),latest_MMWR)
        
        if os.path.exists(forecast_file):
            return
        
        past_inc_hosps_state       = inc_hosps.loc[ (inc_hosps.location==location) & (inc_hosps.season!=season) ]
        past_inc_hosps_MMWR20      = past_inc_hosps_state.loc[past_inc_hosps_state.MMWRWK==20]
        mean_past_inc_hosps_MMWR20 = np.nanmean(past_inc_hosps_MMWR20.value.values)
        sd_past_inc_hosps_MMWR20   = np.nanstd(past_inc_hosps_MMWR20.value.values)


        state_ili = ili_data.loc[(ili_data.location==location) & (ili_data.season==season)].drop_duplicates()
        state_ili = state_ili.loc[(state_ili.MMWRWK>=40) | (state_ili.MMWRWK <=20)]
 
        weeks_needed = list(np.arange(40,52+1))  + list(np.arange(1,20+1))
        for week in weeks_needed:
            if week not in state_ili.MMWRWK.values:
                if week >=40:
                    MMWRYR = min(subset.MMWRYR)
                else:
                    MMWRYR = max(subset.MMWRYR)
                
                d_ = pd.DataFrame({"week":[week],"MMWRWK":[week], "MMWRYR":[MMWRYR] })
                state_ili = pd.concat([state_ili, d_])

        weeks_needed = pd.DataFrame({"MMWRWK":weeks_needed})
        state_ili    = weeks_needed.merge(state_ili, on = ["MMWRWK"])
        subset       = weeks_needed.merge(subset   , on = ["MMWRWK"], how = "left")
                
        #--need to add a "53rd"week if one does not exist and fill it with NA
        if 53 not in subset.MMWRWK.values:
            part_one = subset.loc[ (subset.MMWRWK>=35) & (subset.MMWRWK<=52), "value" ].values
            part_two = subset.loc[ (subset.MMWRWK>=1) & (subset.MMWRWK<=20) , "value" ].values
            ttl_flu  = np.append( np.append( part_one, np.array([np.nan]) ), part_two)

            part_one = state_ili.loc[ (state_ili.MMWRWK>=35) & (state_ili.MMWRWK<=52), "num_patients" ].values
            part_two = state_ili.loc[ (state_ili.MMWRWK>=1) & (state_ili.MMWRWK<=20) , "num_patients" ].values
            N         = np.append( np.append( part_one, np.array([np.nan]) ), part_two)
        else:
            ttl_flu   = subset["value"].values
            N         = state_ili.num_patients.values

        N       = interpolate_nans(N)
        ttl_flu = np.array(list(ttl_flu) + [np.nan]*( len(N) - len(ttl_flu) ))

        #--collect prior data
        #--load in prior param densities

        #--choose the params that are not included in this season
        season_excluded_params = all_params.loc[all_params.season!=season]
        
        historical_params = season_excluded_params.loc[ (season_excluded_params.location==location) ]
        mu_params         = historical_params.loc[historical_params.param_type=="mu"]
        cov_params        = historical_params.loc[historical_params.param_type=="cov"] 

        param_names = ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season"]
        
        prior_mus = []
        for season, mus in mu_params.groupby(["season"]):
            mu_vals = [ float(mus[mus.param1==x]["value"])  for x in param_names ]
            prior_mus.append(mu_vals)
        prior_mus = np.array( prior_mus )

        prior_covs = []
        for season, covs in cov_params.groupby(["season"]):
            covs         = pd.pivot_table(index=["param1"], columns = ["param2"], values=["value"], data = covs)
            covs.columns = [y for x,y in covs.columns]
            covs         = covs.loc[param_names][param_names].to_numpy()

            prior_covs.append(covs)
        prior_covs = np.array(prior_covs)

        #--compute condition number for covs
        conds = []
        for cov in prior_covs:
            conds.append(np.linalg.cond(cov))
        conds = np.array(conds)

        kmax = 10
        
        new_prior_covs = [] 
        for cov,cond in zip(prior_covs,conds):
            if cond>kmax:
                lambdas, vectors = np.linalg.eigh(cov)
                delta            = (lambdas[-1]-lambdas[0]*kmax)/(kmax-1)
                new_prior_covs.append(cov + delta*np.eye(len(cov)))
            else:
                new_prior_covs.append(cov)
                
        prior_covs = np.array(new_prior_covs)

        import jax 
        base_key    = jax.random.PRNGKey(20200320)
        worker_key  = jax.random.fold_in(base_key, 1)

        model = tempo_model4(   y  = (ttl_flu+1./10).reshape(1,-1)
                              , X  = None
                              , N  = N.reshape(1,-1)
                              ,key = worker_key ).fit_new_season(   prior_mus     = prior_mus
                                                                  , prior_covs    = prior_covs
                                                                  , forecast      = True
                                                                  , N_pred        = N 
                                                                  , constraint_mu = mean_past_inc_hosps_MMWR20
                                                                  , constraint_sd = sd_past_inc_hosps_MMWR20
                                                                 )
        yhats = model["cases_predicted"].squeeze()

        #--STORE DATA-----------------------------------------------------
        #---extract quantiles
        quantiles          = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.05)), [0.975,0.99])
        
        #--WEEKLY INCIDENCE DATA------------------------------------------------------------------------------
        weekly_times            = np.percentile(yhats, quantiles*100, axis=0) #--the -1 is the most recent season
        
        def generate_epiweek_end_dates(start_year, start_week, end_year, end_week):
            current_week = Week(start_year, start_week)
            end_week_obj = Week(end_year, end_week)
            end_dates    = []

            while current_week <= end_week_obj:
                # Calculate the Sunday (end of the week)
                end_dates.append(current_week.enddate())
                # Move to the next week
                current_week = current_week + 1

            return end_dates

        # Define the start and end epiweeks for the 2024/2025 season
        start_year, start_week = int(min(subset.MMWRYR.values)), 40  
        end_year, end_week     = int(max(subset.MMWRYR.values)), 22

        #reference_date         = Week(start_year,start_week).enddate()
        reference_date         = datetime.strptime( subset.loc[ subset.MMWRWK==latest_MMWR, "date"].values[0], "%Y-%m-%d").date() #Week.thisweek().enddate() 
        
        # Generate and print all epiweek end dates for the 2024/2025 influenza season
        timepoints = generate_epiweek_end_dates(start_year, start_week, end_year, end_week)
        
        #--add data to dictionary
        forecast_data = {"reference_date"  :[]
                         ,"horizon"        :[]
                         ,"target_end_date":[]
                         ,"output_type_id" :[]
                         ,"value"          :[]}
        for forecast_time,d in zip(timepoints, weekly_times.T):
            fmt = "%Y-%m-%d"
            
            forecast_data["reference_date"].extend( [reference_date.strftime(fmt)]*23 )

            week_from_reference = int((forecast_time - reference_date).days/7)
            
            forecast_data["horizon"].extend( [week_from_reference]*23 )

            ted = Week.fromdate(forecast_time).enddate().strftime(fmt)
            forecast_data["target_end_date"].extend([ted]*23)

            forecast_data["output_type_id"].extend( ["{:0.3f}".format(x) for x in quantiles] )
            forecast_data["value"].extend( [ int(x) for x in np.floor(d)] )
            
        weekly_forecast_data = pd.DataFrame(forecast_data)
        weekly_forecast_data["location"]    = location
        weekly_forecast_data["output_type"] = "quantile"
        weekly_forecast_data["target"]      = "wk inc flu hosp"
        weekly_forecast_data["latest_MMWR"] = latest_MMWR

        columns              = ["reference_date","target","horizon","target_end_date","latest_MMWR","location","output_type","output_type_id","value"]
        weekly_forecast_data = weekly_forecast_data[columns]

        if location=="US":
            weekly_forecast_data.to_csv("./forecast_experiment/control_arm/forecasts/forecast_US_{:s}.csv".format(latest_MMWR))
        else:
            weekly_forecast_data.to_csv("./forecast_experiment/control_arm/forecasts/forecast_{:02d}_{:s}.csv".format( int(location),latest_MMWR))

    inc_hosps = inc_hosps.loc[inc_hosps.location.isin(all_params.location.unique())]
    inc_hosps = inc_hosps.loc[inc_hosps.season!='-1']
    
    def tryit(location,season,subset,weather_data, pct_hosps_reporting,ili_augmented):
        try:
            forecast(location,season,subset,weather_data,pct_hosps_reporting,ili_augmented)
        except:
          print("Fail")
          print(location)

    for (location,season),subset in inc_hosps.groupby(["location","season"]):
        subset = subset.sort_values(["date"])
        subset["season_week"] = np.arange(len(subset))
        
        for week in subset.season_week:
            subset_by_week = subset.loc[ (subset.season_week<=week) & (subset.MMWRWK>=40) ] #<--lets keep everything fair by starting at MMWR40
            forecast(location,season,subset_by_week)

            break
        break
    
        
    Parallel(n_jobs=20)( delayed(tryit)(location,season,subset,weather_data, pct_hosps_reporting,ili_augmented) for (location,season), subset in inc_hosps.groupby(["location","season"]) )
   






























    

