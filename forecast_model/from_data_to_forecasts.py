#mcandrew

import sys
sys.path.append("./forecast_model/tempo/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from epiweeks import Week

from tempo_model import tempo_model2

if __name__ == "__main__":

    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    hosp_data = hosp_data.loc[hosp_data.season!="2020/2021"]

    pop_data =  pd.read_csv("./data/locations.csv")

    hosp_data = hosp_data.merge(pop_data, on = ["location"])

    for location, subset in hosp_data.groupby(["location"]):
        N = subset.iloc[0]["population"]
        
        y_by_season = pd.pivot_table(index="season",columns="model_week",values="value",data=subset)
        y_by_season = y_by_season.to_numpy()

        nseasons, ntimes= y_by_season.shape

        #--exclude this season
        new_cases   = y_by_season[-1,:]
        new_cases[10:] = np.nan
        
        y_by_season = y_by_season[:-1,:]

        past_season_forecasts = []
        for season, season_data in enumerate(y_by_season):
            season_data = season_data.reshape(1,-1)
            model = tempo_model2( y = season_data, N = np.ones( season_data.shape )*N  )
            past_season_forecasts.append(model.fit_past_seasons())

        prior_tensor = []
        for season in range(nseasons-1):
            model = past_season_forecasts[season]
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

        #--only need to run this through time 
        tempo          = tempo_model2(y= new_cases.reshape(1,ntimes), N = np.ones( (1,ntimes) )*N , nobs=ntimes)# np.nan*np.ones(nweeks).reshape(1,nweeks), nobs=0)
        forecast_model = tempo.fit_new_season(prior_mus=prior_mus, prior_covs = prior_covs, forecast=True, N_pred =  np.ones( (1,ntimes) )*N)
        forecast_samples        = forecast_model["inc_pred"]
 
