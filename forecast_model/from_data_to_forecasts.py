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

    
