#mcandrew

import sys
import numpy as np
import pandas as pd

from glob import glob

if __name__ == "__main__":

    files = glob("./historical_model_run_for_tempo/arxiv__tempo4/*.csv")
    d = []
    for fin in files:
        d.append( pd.read_csv(fin) )
    all_params = pd.concat(d)

    all_params = all_params[ ['location','season','param_type','param1','param2','value'] ]
    all_params.to_csv("./historical_model_run_for_tempo/all_past_param_estimates__tempo4.csv", index=False)
