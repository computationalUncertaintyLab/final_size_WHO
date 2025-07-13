#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LassoCV
import shap
import xgboost as xgb

if __name__ == "__main__":


    d = pd.read_csv("./analysis_data/week_country_level_data.csv")
    d = d.loc[ (d.SEASON != -1) & (d.SEASON != 2025) ,: ]
    d["prop"] = (1+d.POS)/(1+d.TTL)

    d = d.loc[ (d.HEMISPHERE == "SH") | (d.COUNTRY_CODE == "USA") ]
    d = d.loc[d.SEASON!=2020]
    
    d = pd.pivot_table(index=["SEASON"],columns = ["COUNTRY_CODE"],values = ["prop"], data = d)
    d.columns = [y for (x,y) in d.columns]

    #--look at complete cases only for now 
    d  = d.dropna(axis=1)
    d_ = (d - d.mean(0)) / d.std(0)
    
    US_cases  = d_["USA"].to_numpy().reshape(-1,1)

    SH            = d_.drop(columns = ["USA"])
    country_names = SH.columns

    #--correlations
    correlations = np.corrcoef(d_.T)[np.argwhere(d_.columns=="USA")  ]
    corrs = {"country_names":[],"corr":[]}
    for name,corr in zip(d_.columns, correlations.reshape(-1,)):
        corrs["country_names"].append(name)
        corrs["corr"].append(corr)
    corrs = pd.DataFrame(corrs)
    corrs = corrs.loc[corrs.country_names!="USA"]

    #--LASSO
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.01)
    clf.fit(SH, US_cases)

    params = {"country_names":[], "params":[]}
    for name,param in zip(country_names, clf.coef_):
        params["country_names"].append(name)
        params["params"].append(param)
    params = pd.DataFrame(params)

    X = SH.to_numpy()
    y = US_cases.reshape(-1,)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=94)

    clf = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=10
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])


    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)
    shap.plots.bar(shap_values.abs.mean(0))



