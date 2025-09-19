#mcandrew

import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from epiweeks import Week

if __name__ == "__main__":

    hosp_data = pd.read_csv("./data/target-data/target-hospital-admissions.csv")

    #--add MMWR data
    def row_to_mmwr(row):
        date = datetime.strptime(row.date, "%Y-%m-%d")
        mmwr_data = Week.fromdate(date)

        return pd.Series({"mmwr_yr"      : mmwr_data.year
                          ,"mmwr_wk"     : mmwr_data.week
                          ,"mmwr_enddate": mmwr_data.enddate()})
        
    mmwr_data = hosp_data.apply( row_to_mmwr,1 )
    hosp_data = pd.concat([hosp_data,mmwr_data] ,axis=1)

    #--add season data
    def row_to_season(x):
        if x.mmwr_wk>=40:
            return "{:d}/{:d}".format(x.mmwr_yr,x.mmwr_yr+1)
        elif x.mmwr_wk>=1 and x.mmwr_wk<=30:
            return "{:d}/{:d}".format(x.mmwr_yr-1,x.mmwr_yr)
        else:
            return "offseason"
        
    hosp_data["season"] = hosp_data.apply(row_to_season,1)
    hosp_data = hosp_data.loc[hosp_data.season!="offseason"]

    #--add modelweek
    def add_mdoel_week(x):
        x = x.sort_values("mmwr_enddate")
        x["model_week"] = np.arange(len(x))
        return x
    hosp_data = hosp_data.groupby(["season","location","location_name"]).apply(add_mdoel_week).reset_index(drop=True)

    columns = ["location","location_name","season","mmwr_yr","mmwr_wk","mmwr_enddate","model_week","value"]
    hosp_data = hosp_data[ columns ]
 
    hosp_data.to_csv("./analysis_data/us_hospital_data.csv", index=False)
