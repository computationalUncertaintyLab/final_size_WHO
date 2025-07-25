#mcandrew

import sys
sys.path.append("./models/")

import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    hosp_data = hosp_data.loc[hosp_data.season!="2020/2021"]
    hosp_data.loc[: , ["location","season"]].drop_duplicates()

    f = open("./models/iteration_list.csv","w")
    for _,row in hosp_data.iterrows():
        f.write("--export=ALL,LOCATION={:s},SEASON={:s}\n".format(row.location,row.season))
    f.close()
