#mcandrew

import pandas as pd
import numpy as np

def format_location(x):
    if x=="US":
        return x
    return "{:02d}".format(int(x))


if __name__ == "__main__":

    control_arm_forecasts = pd.read_csv("./forecast_experiment/control_arm/control_arm_forecasts.csv")
    control_arm_forecasts["location"] = [ format_location(x) for x in control_arm_forecasts.location.values]

    true_values = pd.read_csv("./data/target-data/target-hospital-admissions.csv")
    true_values = true_values.rename(columns={"value":"obs"})
    true_values["location"] = [ format_location(x) for x in true_values.location.values]

    forecasts_and_truth = control_arm_forecasts.merge(true_values, left_on=["location","target_end_date"], right_on=["location","date"])

    forecasts_and_truth.to_csv("./forecast_experiment/evaluation/forecasts_and_truth.csv", index=False)