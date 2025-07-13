#mcandrew

import sys
import numpy as np
import pandas as pd

from epiweeks import Week

if __name__ == "__main__":

    d = pd.read_csv("./data/VIW_FNT.csv")

    cols = ['WHOREGION','FLUSEASON','HEMISPHERE'
            ,'COUNTRY_CODE','COUNTRY_AREA_TERRITORY'
            ,'ISO_WEEKSTARTDATE','MMWRYW'
            ,'SPEC_PROCESSED_NB'
            ,'INF_ALL','INF_NEGATIVE']
    
    d = d[cols]

    def assign_season(x):
        hem    = str(x.HEMISPHERE)
        MMWRYW = str(x.MMWRYW)
        yr,wk  = int(MMWRYW[:4]), int(MMWRYW[-2:])

        if hem == "NH":
            if wk>=40 and wk<=53:
                season=yr
            elif wk>=1 and wk<=22:
                season=yr-1
            else:
                season=-1
                
        elif hem == "SH":
            if wk>=1 and wk<40:
                season=yr
            else:
                season=-1
        return season
    
    d["SEASON"] = d.apply(assign_season, 1)
    
    #--H1N1 and after 
    d = d.loc[ (d.SEASON >=2009) | (d.SEASON==-1) ]
    
    def add_up_cases(x):
        return pd.Series({"POS": np.nansum(x.INF_ALL), "NEG": np.nansum(x.INF_NEGATIVE), "TTL": np.nansum(x.SPEC_PROCESSED_NB)  })

    season_level_data      = d.loc[(d.SEASON >=2009),:].groupby(["SEASON","HEMISPHERE"]).apply(add_up_cases).reset_index()
    season_level_data      = season_level_data.loc[season_level_data.SEASON!=-1]
    season_level_data["P"] = season_level_data.POS/season_level_data.TTL

    season_level_data.to_csv("./analysis_data/season_level_data.csv",index=False)

    week_level_data = d.groupby(["HEMISPHERE","MMWRYW","SEASON"]).apply(add_up_cases).reset_index()

    def extract_wee_and_year(x):
        MMWRYW      = str(x.MMWRYW)
        yr,wk       = int(MMWRYW[:4]), int(MMWRYW[-2:])
        x["MMWRYR"] = yr
        x["MMWRWK"] = wk
        return x
    week_level_data      = week_level_data.apply(extract_wee_and_year,1)
    week_level_data["P"] = week_level_data.POS/week_level_data.TTL

    model_week = {"MMWRYR":[],"MMWRWK":[],"MODELWEEK":[]}

    n=0
    week = Week(2009,1)
    while week != Week(2025,52):
        model_week["MMWRYR"].append( week.year )
        model_week["MMWRWK"].append( week.week )
        model_week["MODELWEEK"].append( n )

        week = week + 1
        n+=1
    model_week = pd.DataFrame(model_week)

    week_level_data = week_level_data.merge(model_week, on = ["MMWRYR","MMWRWK"], how = "left")

    week_level_data.to_csv("./analysis_data/week_level_data.csv",index=False)

    #--week and country level data
    week_country_level_data = d.groupby(["WHOREGION","COUNTRY_CODE","COUNTRY_AREA_TERRITORY","HEMISPHERE","SEASON"]).apply(add_up_cases).reset_index()
    week_country_level_data.to_csv("./analysis_data/week_country_level_data.csv",index=False)
