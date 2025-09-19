#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    hosp_data = hosp_data.loc[hosp_data.season!="2020/2021"]

    def sumup(x):
        return pd.Series({"ttl_hosp": int(x.value.sum())})
    hosp_data = hosp_data.groupby(["season","location"]).apply(sumup).reset_index()

    #--make wide
    hosp_data = pd.pivot_table(index=["season"], columns = ["location"], values = ["ttl_hosp"], data  = hosp_data)

    #--map season in NH to season in SH
    hosp_data.index = [ int(x.split("/")[0])  for x in hosp_data.index]
    hosp_data.columns = [y for x,y in hosp_data.columns]

    original_hosp_data = hosp_data.copy()

    #--normalize
    hosp_means = hosp_data.mean(0)
    hosp_stds  = hosp_data.std(0)
    hosp_min   = hosp_data.min(0)
    hosp_max   = hosp_data.max(0)
 
    hosp_data_norm  = (hosp_data - hosp_means) / hosp_stds 
 
    locations = hosp_data_norm.columns
    
    d = pd.read_csv("./analysis_data/week_country_level_data.csv")
    d = d.loc[ (d.SEASON != -1)  ,: ]
    d = d.loc[d.SEASON >=2015]

    #--IDN has no variability
    d = d.loc[d.COUNTRY_CODE!="IDN"]

    d["prop"] = (d.POS+1) / (d.POS + d.NEG + 1)

    d = d.loc[ (d.HEMISPHERE == "SH") | (d.COUNTRY_CODE == "USA") ]
    d = d.loc[d.SEASON!=2020]
    
    d = pd.pivot_table(index=["SEASON"],columns = ["COUNTRY_CODE"],values = ["prop"], data = d)
    d.columns = [y for (x,y) in d.columns]

    #--look at complete cases only for now 
    d  = d.dropna(axis=1)

    dmean = d.mean(0)
    dstd  = d.std(0) 
    dmin  = d.min(0)
    dmax  = d.max(0)

    d_ = (d - dmean) / dstd  

    d  = d.loc[d.index >=2021]

    country_names = d_.columns

    #-----MERGE IN
    hosp_data_norm = hosp_data_norm.merge(d_, left_index=True, right_index = True)
    hosp_data_norm.to_csv("./models/predicting_state_from_SH_countries/normalized_US_hosp_and_SH_WHO_cases.csv", index=False)

    un_norm_hosp = hosp_data.merge(d, left_index=True, right_index = True)
    un_norm_hosp.to_csv("./models/predicting_state_from_SH_countries/un_normalized_US_hosp_and_SH_WHO_cases.csv", index=False)
    
