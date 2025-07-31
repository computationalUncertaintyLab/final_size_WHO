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
    location = "42"
    
    reference_date = Week(2024,40)
    for _ in range(T):
        reference_date = reference_date+1
        

    y_full = hosp_data.loc[(hosp_data.location==str(location)) & (hosp_data.season=="2024/2025"), "value"].values
    y      = y_full.copy()
    y[T:]  = np.nan

    model_one = model()
    
    forecasted_inc = model_one.train(y = y
                                 , prior_parameters_dataset = prior_parameters_dataset
                                 , location                 = location
                                 , season                   = "2024/2025")


    model_one.build_calibration_function( "./models/calibration_data.csv" )

    PI_      = model_one.build_PI_dataset(reference_date = reference_date, recal=False)
    PI_recal = model_one.build_PI_dataset(reference_date = reference_date, recal=True)
    

    PI_      = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_)
    PI_.columns = ["{:.1f}".format(float(y)) for x,y in PI_.columns] 
    
    PI_recal = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_recal)
    PI_recal.columns = ["{:.1f}".format(float(y)) for x,y in PI_recal.columns]


    #--constrained system
    model = model()
    
    forecasted_inc = model.train(y = y
                                 , prior_parameters_dataset = prior_parameters_dataset
                                 , location                 = location
                                 , season                   = "2024/2025"
                                 , total_constraint         = (np.sum(y_full), 0.10*np.sum(y_full) ) )
    model.build_calibration_function( "./models/calibration_data.csv" )

    PI_2      = model.build_PI_dataset(reference_date = reference_date, recal=False)
    PI_2recal = model.build_PI_dataset(reference_date = reference_date, recal=True)
    

    PI_2         = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_2)
    PI_2.columns = ["{:.1f}".format(float(y)) for x,y in PI_2.columns] 
    
    PI_2recal = pd.pivot_table(index= ["target_end_date"], columns = ["output_type_id"], values = ["value"], data = PI_2recal)
    PI_2recal.columns = ["{:.1f}".format(float(y)) for x,y in PI_2recal.columns]
    

    times = np.arange(34)
    # plt.plot(times,  jnp.append( y[~np.isnan(y)], med))
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low1),jnp.append( y[~np.isnan(y)], high1),alpha=0.20,color="blue")
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low2),jnp.append( y[~np.isnan(y)], high2),alpha=0.20,color="blue")

    past_data = hosp_data.loc[(hosp_data.location==location) ]
    past_data = pd.pivot_table(index= ["model_week"], columns = ["season"], values = ["value"], data = past_data)

    
    fig, axs = plt.subplots(1,2)

    ax = axs[0]
    
    ax.plot(times        , PI_recal["50.0"], color="red")
    ax.fill_between(times, PI_recal['25.0'], PI_recal["75.0"] ,alpha=0.10,color="red")
    ax.fill_between(times, PI_recal['10.0'], PI_recal["90.0"] ,alpha=0.10,color="red")
    ax.fill_between(times, PI_recal['2.5'] , PI_recal["97.5"] ,alpha=0.10,color="red")

    ax.plot( np.arange(34), y)
    ax.scatter( np.arange(34)[:T], y_full[:T] , s=50, edgecolors='white', linewidths=2, zorder=3)
    ax.scatter( np.arange(34)[T:], y_full[T:] , s=50, color="blue")

    ax.set_ylim(0,6000)

    ax = axs[1]
    
    ax.plot(times        , PI_2recal["50.0"], color="blue")
    ax.fill_between(times, PI_2recal['25.0'], PI_2recal["75.0"] ,alpha=0.10,color="blue")
    ax.fill_between(times, PI_2recal['10.0'], PI_2recal["90.0"] ,alpha=0.10,color="blue")
    ax.fill_between(times, PI_2recal['2.5'] , PI_2recal["97.5"] ,alpha=0.10,color="blue")


    ax.plot( np.arange(34), y)
    ax.scatter( np.arange(34)[:T], y_full[:T] , s=50, edgecolors='white', linewidths=2, zorder=3)
    ax.scatter( np.arange(34)[T:], y_full[T:] , s=50, color="blue")

    plt.plot(past_data.values,color="0.40")
    
    ax.set_xlabel("MMWR Week")
    ax.set_ylabel("Inc. Hosps. for PA")

    ax.set_xticks([0,9,22,32])
    ax.set_xticklabels(["40","50","10","20"])

    ax.set_ylim(0,6000)
   
    plt.show()
 
