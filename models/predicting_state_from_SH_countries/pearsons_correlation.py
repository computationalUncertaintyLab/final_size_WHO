#mcandrew

import sys
import numpy as np
import pandas as pd
import re

if __name__ == "__main__":

    hosp_data_norm = pd.read_csv("./models/predicting_state_from_SH_countries/normalized_US_hosp_and_SH_WHO_cases.csv")

    locations = [x for x in hosp_data_norm.columns if re.match("[0-9]",x) ]
    country_names = [x for x in hosp_data_norm.columns if not re.match("[0-9]",x) ]
    
    states       = hosp_data_norm.loc[:,locations]
    sh_countries = hosp_data_norm.loc[:,country_names]
    
    #--correlations
    corrs = {"location":[],"country_names":[],"corr":[]}
    for location in locations:
        correlations = np.corrcoef(states[location].T, sh_countries.T)[1:,0]
        for name,corr in zip(country_names, correlations.reshape(-1,)):
            corrs["location"].append(location)
            corrs["country_names"].append(name)
            corrs["corr"].append(corr)
    corrs = pd.DataFrame(corrs)

    corrs.to_csv("./pearsons_correlation_between_SH_countries_and_US_state_hosps.csv", index=False)
