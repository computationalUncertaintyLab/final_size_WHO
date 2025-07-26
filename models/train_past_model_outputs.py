#mcandrew

import sys
sys.path.append("./models/")

import numpy as np
import pandas as pd

from pathlib import Path

from model import model

import argparse

if __name__ == "__main__":

    prior_parameters_dataset = pd.read_csv("./models/prior_params.csv")
    
    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    hosp_data = hosp_data.loc[hosp_data.season!="2020/2021"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--SEASON', type=str)
    parser.add_argument('--LOCATION', type=str)

    args = parser.parse_args()
    LOCATION = args.LOCATION
    SEASON   = args.SEASON.replace("_","/") 

    subset_data = hosp_data.loc[(hosp_data.season==args.SEASON) &(hosp_data.location==args.LOCATION),: ]

    print("Length")
    print(len(subset_data))
    
    for model_week in subset_data.model_week.unique():
        quantile_data = {"location":[],"season":[],"model_week":[],"forecast_week":[],"mmwr_yr":[],"mmwr_wk":[],"mmwr_enddate":[],"quantile":[],"quantile_value":[]}

        data = subset_data.loc[subset_data.model_week <= model_week]
        
        model_instance = model()

        y = data['value'].values
        y = np.append(y, np.nan*np.ones( len(subset_data)-len(y) ) )
        
        forecasted_inc = model_instance.train(y = y
                                     , prior_parameters_dataset = prior_parameters_dataset
                                     , location = args.LOCATION
                                     , season   = args.SEASON)

        percentiles     = np.arange(0,100+1,1)
        quantiles       = percentiles/100 
        quantile_values = np.percentile( forecasted_inc,percentiles,axis=0 )

        for week, quantile_value in enumerate(quantile_values.T):
            last_row = data.iloc[-1]
            L        = len(quantiles) 
            quantile_data["location"].extend([args.LOCATION]*L)
            quantile_data["season"].extend([args.SEASON]*L)

            quantile_data["model_week"].extend(  [last_row["model_week"]  ]*L)
            quantile_data["forecast_week"].extend(  [week]*L)
            
            quantile_data["mmwr_yr"].extend(     [last_row["mmwr_yr"]     ]*L)
            quantile_data["mmwr_wk"].extend(     [last_row["mmwr_wk"]     ]*L)
            quantile_data["mmwr_enddate"].extend([last_row["mmwr_enddate"]]*L)

            quantile_data["quantile"].extend( quantiles )
            quantile_data["quantile_value"].extend(quantile_value)

        quantile_data = pd.DataFrame(quantile_data)

        path = "./models/arxiv_cdfs.csv"
        fout = Path(path)
        if fout.is_file():
            quantile_data.to_csv(path, index=False, mode="a", header=False)
        else:
            quantile_data.to_csv(path, index=False, mode="w", header=True)


    
