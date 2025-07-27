#mcandrew

import sys
sys.path.append("./models/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from epiweeks import Week
from model import model

if __name__ == "__main__":

    import jax.numpy as jnp
   
    prior_parameters_dataset = pd.read_csv("./models/prior_params.csv")

    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    hosp_data = hosp_data.loc[hosp_data.season!="2020/2021"]

    T        = 13
    location = "22"
    
    reference_date = Week(2024,40)
    for _ in range(T):
        reference_date = reference_date+1
        

    y_full = hosp_data.loc[(hosp_data.location==str(location)) & (hosp_data.season=="2024/2025"), "value"].values
    y      = y_full.copy()
    y[T:]  = np.nan

    model = model()
    
    forecasted_inc = model.train(y = y
                                 , prior_parameters_dataset = prior_parameters_dataset
                                 , location = location
                                 , season = "2024/2025")


    model.build_calibration_function( "./models/calibration_data.csv" )

    PI_      = model.build_PI_dataset(reference_date = reference_date, recal=False)
    PI_recal = model.build_PI_dataset(reference_date = reference_date, recal=True)
    

    PI_      = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_)
    PI_.columns = ["{:.1f}".format(float(y)) for x,y in PI_.columns] 
    
    PI_recal = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_recal)
    PI_recal.columns = ["{:.1f}".format(float(y)) for x,y in PI_recal.columns] 

    times = np.arange(34)
    # plt.plot(times,  jnp.append( y[~np.isnan(y)], med))
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low1),jnp.append( y[~np.isnan(y)], high1),alpha=0.20,color="blue")
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low2),jnp.append( y[~np.isnan(y)], high2),alpha=0.20,color="blue")

    past_data = hosp_data.loc[(hosp_data.location=="42") ]
    past_data = pd.pivot_table(index= ["model_week"], columns = ["season"], values = ["value"], data = past_data)

    
    fig, ax = plt.subplots()
    
    plt.plot(times,PI_["50"], color="red")
    plt.fill_between(times, PI_['2.5'], PI_["97.5"] ,alpha=0.20,color="red")
    plt.fill_between(times, PI_['10.0'], PI_["90.0"],alpha=0.20,color="red")


    plt.plot(times,PI_recal["50"], color="blue")
    plt.fill_between(times, PI_recal['2.5'], PI_recal["97.5"] ,alpha=0.20,color="blue")
    plt.fill_between(times, PI_recal['10.0'], PI_recal["90.0"],alpha=0.20,color="blue")


    ax.plot( np.arange(34), y)
    ax.scatter( np.arange(34)[:T], y_full[:T] , s=50, edgecolors='white', linewidths=2, zorder=3)
    ax.scatter( np.arange(34)[T:], y_full[T:] , s=50, color="blue")

    plt.plot(past_data.values,color="0.40")
    
    ax.set_xlabel("MMWR Week")
    ax.set_ylabel("Inc. Hosps. for PA")

    ax.set_xticks([0,9,22,32])
    ax.set_xticklabels(["40","50","10","20"])

   
    plt.show()
 
