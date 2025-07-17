#mcandrew

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import numpyro
numpyro.enable_x64(True)
import numpyro.distributions as dist
from   numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_sample,init_to_uniform

import jax
import jax.numpy as jnp

#--logit function
def genLogModel(data         = None
                , tobs       = None
                , priors     = {}
                , forecast   = False):
    
    def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
        exp_term = jnp.exp(-B * (t - M))
        denom = (1 + Q * exp_term)
        return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

    times = np.arange(1,data.shape[-1]+1)
    nseasons = data.shape[0]

    y      = data.reshape(-1,)
    x      = jnp.arange(33)

    params = {"A":1}
    for v in ["B","M","Q","nu"]:
        if v in priors:
            m,s       = priors.get(v)
            log_v     = numpyro.sample( v, dist.Normal(m,s) )
            params[v] = jnp.exp(log_v)
        else:
            params[v] = numpyro.sample(v  , dist.LogNormal(-1,1) )

    if "K" in priors:
        conc,mean = priors["K"]
        K  = numpyro.sample("K"  , dist.Beta(conc*mean, conc*(1-mean) ) )
    else:
        params["K"] = numpyro.sample("K"  , dist.Beta(1,1)       )

    dS = d_generalized_logistic( times
                                 , params["A"]
                                 , params["K"]
                                 , params["B"]
                                 , params["M"]
                                 , params["Q"]
                                 , params["nu"]
                                )
    numpyro.deterministic("dS", dS)

    if "N" in priors:
        m,s = priors["N"]
        log_N  = numpyro.sample( "N", dist.Normal(m,1))
        N      = jnp.exp(log_N)
    else:
        N      = numpyro.sample( "N", dist.LogNormal(jnp.log(10*10**3),1) )

    inc_p      = numpyro.deterministic("inc_p"     , -dS )
    inc        = numpyro.deterministic("inc"       , jnp.clip( N*inc_p ,10**-6,jnp.inf ) )
    peak_time  = numpyro.deterministic( "peak_time", jnp.argmax(inc_p) )

    std        = numpyro.sample("s", dist.HalfCauchy(1))
    nany       = ~jnp.isnan(data.reshape(-1,))
    
    inc_process = N*inc_p.reshape(-1,)
    with numpyro.handlers.mask(mask=nany):
        numpyro.sample("ll", dist.Normal(inc_process,std), obs = y)

    if forecast:
        numpyro.sample("forecast", dist.Normal(inc_process, std))



if __name__ == "__main__":

    all_params = {"location_name":[]
                  ,"location"    :[]
                  ,"season"      :[]
                  ,"K"           :[]
                  ,"B"           :[]
                  ,"M"           :[]
                  ,"Q"           :[]
                  ,"nu"          :[]
                  ,"N"           :[]
                  ,"peak_time"   :[]
                  }

    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")
    for (state_name,fips,season),data in hosp_data.groupby(["location_name","location","season"]):
        if season == "2020/2021":
            continue

        print("{:s} - {:s}".format(state_name,season))
        data["mmwr_enddate"] = [ datetime.strptime(x,"%Y-%m-%d") for x in data.mmwr_enddate.values]
    
        y = data["value"].values.reshape(-1,)

        mcmc = MCMC(NUTS(genLogModel), num_warmup=10*10**3, num_samples=10*10**3)
        mcmc.run(jax.random.PRNGKey(1)
                 ,data            = y
                 ,tobs            = None
                 ,priors          = {}
                 )

        mcmc.print_summary()
        training_samples = mcmc.get_samples()

        predictive  = Predictive(genLogModel
                                 , posterior_samples = training_samples
                                 , return_sites = ["K","B","M","Q","nu","N","forecast","peak_time"])

        predictions = predictive(jax.random.PRNGKey(10)
                                 ,data     = y
                                 ,tobs     = None
                                 ,priors   = {}
                                 ,forecast = True
                                 )

        K         = predictions["K"].mean(0)  
        B         = predictions["B"].mean(0)  
        M         = predictions["M"].mean(0)  
        Q         = predictions["Q"].mean(0)
        nu        = predictions["nu"].mean(0)
        N         = predictions["N"].mean(0)  
        peak_time = predictions["peak_time"].mean(0)

        all_params["location_name"].append(state_name)
        all_params["location"].append(fips)
        all_params["season"].append(season)
        all_params["K"].append(K)
        all_params["B"].append(B)
        all_params["M"].append(M)
        all_params["Q"].append(Q)
        all_params["nu"].append(nu)
        all_params["N"].append(N)
        all_params["peak_time"].append(peak_time)

    all_params = pd.DataFrame(all_params)
    all_params["A"] = 1.

    all_params.to_csv("./models/prior_params.csv",index=False)
