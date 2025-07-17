#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    import jax.numpy as jnp
    def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
        exp_term = jnp.exp(-B * (t - M))
        denom = (1 + Q * exp_term)
        return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))
    
    d = pd.read_csv("./models/prior_params.csv")

    curves = {"t":[],"value":[],"sim":[],"location":[]}
    times = np.arange(43)
    for sim,(idx, row) in enumerate(d.iterrows()):
        inc = -d_generalized_logistic(times
                               ,row.A
                               ,row.K
                               ,row.B
                               ,row.M
                               ,row.Q
                               ,row.nu)
    
        curves["value"].extend([ float(x) for x in inc])
        curves["t"].extend(times)
        curves["sim"].extend(43*[sim])
        curves["location"].extend(43*[row.location])
        
    curves = pd.DataFrame(curves)

    import numpyro
    numpyro.enable_x64(True)
    import numpyro.distributions as dist
    from   numpyro.infer import Predictive

    from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_sample,init_to_uniform

    import jax
    import jax.numpy as jnp

    def model(y,X,m,k):
        numpyro.sample("f", dist.MultivariateNormal(m[:10],k[:10,:10]), obs = y.reshape(-1,))

    C = pd.pivot_table(index=['sim'],columns = ['t'],values=['value'], data=curves.loc[curves.location=="42"] ).to_numpy()
    
    m = C.mean(0)
    k = np.cov(C.T)
    s = C.std(0)

    vaccine_effect = pd.read_csv("./models/vaccine_effectiveness.py")
    vaccine_effect = vaccine_effect.loc[vaccine_effect.season.isin(["2021/2022","2022/2023","2023/2024"])]
 
    
    
    def genLogModel(data         = None
                    , tobs       = None
                    , priors     = {}
                    , mn = None
                    , sc = None
                    , forecast   = False):

        def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
            exp_term = jnp.exp(-B * (t - M))
            denom = (1 + Q * exp_term)
            return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))


        def d2_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
            """
            Second derivative of the generalized logistic function (incidence acceleration).

            Parameters
            ----------
            t : array-like
                Time points.
            A, K, B, M, Q, nu : float
                Parameters of the generalized logistic function.

            Returns
            -------
            array-like
                Second derivative of the generalized logistic function at time t.
            """
            exp_term = jnp.exp(-B * (t - M))           # e^{-B(t - M)}
            denom = 1 + Q * exp_term                   # 1 + Q e^{-B(t - M)}
            r = 1 / nu + 1

            base = ((K - A) * B * Q * exp_term) / (nu * denom**r)
            correction = -B + ((r * Q * B**2 * exp_term) / denom)

            return base * correction


        times = np.arange(1,data.shape[-1]+1)
        nseasons = data.shape[0]

        y      = data.reshape(-1,)
        x      = jnp.arange(len(y))

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
            params["K"]  = numpyro.sample("K"  , dist.Beta(conc*mean, conc*(1-mean) ) )
        else:
            params["K"] = numpyro.sample("K"  , dist.Beta(1,1)       )

        sigma_f      = numpyro.sample("sigma_f"     , dist.LogNormal(-1,1))
        random_walk  = numpyro.sample("random_walk" , dist.GaussianRandomWalk( sigma_f, len(y) ) )
        params["B"]  = params["B"] + random_walk 
        dS = jax.vmap(d_generalized_logistic, in_axes = (0,)+ (None,)*2 + (0,) + (None,)*3  )( times
                                                                                               , params["A"]
                                                                                               , params["K"]
                                                                                               , params["B"]
                                                                                               , params["M"]
                                                                                               , params["Q"]
                                                                                               , params["nu"]
                                                                                              )
        

        
        if "N" in priors:
            m,s = priors["N"]
            log_N  = numpyro.sample( "N", dist.Normal(m,2))
            N      = jnp.exp(log_N)
        else:
            N      = numpyro.sample( "N", dist.LogNormal(jnp.log(10*10**3),1) )

        inc_p      = numpyro.deterministic("inc_p"     , -dS )
        inc        = numpyro.deterministic("inc"       , jnp.clip( N*inc_p ,10**-6,jnp.inf ) )
        peak_time  = numpyro.deterministic( "peak_time", jnp.argmax(inc_p) )

        nany       = ~jnp.isnan(data.reshape(-1,))
        numpyro.sample("ll2", dist.Normal(mn[tobs:],sc[tobs:]), obs = inc_p.reshape(-1,)[tobs:] )

        second_deriv =  jax.vmap(d2_generalized_logistic, in_axes = (0,)+ (None,)*2 + (0,) + (None,)*3  )( times
                                                                                               , params["A"]
                                                                                               , params["K"]
                                                                                               , params["B"]
                                                                                               , params["M"]
                                                                                               , params["Q"]
                                                                                               , params["nu"]
                                                                                              )
        #numpyro.sample("penalty", dist.Normal(0, 1), obs = second_deriv.sum())
        inc_process = jnp.clip( N*inc_p.reshape(-1,) - (1./3) + (1./50), 10**-6,jnp.inf)
        with numpyro.handlers.mask(mask=nany):
            numpyro.sample("ll" , dist.Poisson(inc_process), obs = y)

        if forecast:
            numpyro.sample("forecast", dist.Poisson(inc_process))


    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")

    y_full = hosp_data.loc[(hosp_data.location=="42") & (hosp_data.season=="2024/2025"), "value"].values
    y = y_full.copy()
    y[13:] = np.nan

    priors_for_state = d.loc[d.location == "42"]
    priors_for_state = priors_for_state.loc[ priors_for_state.season!="2024/2025" ]

    prior_params = {}
    for col_name in priors_for_state[ ["B","M","Q","nu","N"] ]:
        v = np.log(priors_for_state[col_name])
        prior_params[col_name] = ( v.mean(), v.std())
        
    
    mcmc = MCMC(NUTS(genLogModel), num_warmup=10*10**3, num_samples=10*10**3)
    mcmc.run(jax.random.PRNGKey(1)
             ,data            = y
             ,tobs            = np.min(np.argwhere(np.isnan(y))[0])
             ,mn               = m[:34]
             ,sc               = s[:34]
             ,priors          = prior_params
             )

    mcmc.print_summary()

    training_samples = mcmc.get_samples()

    predictive  = Predictive(genLogModel
                             , posterior_samples = training_samples
                             , return_sites = ["K","B","M","Q","nu","N","forecast","peak_time"])

    predictions = predictive(jax.random.PRNGKey(10)
                             ,data     = y
                             ,tobs     = np.min(np.argwhere(np.isnan(y))[0])
                             ,priors   = prior_params
                             ,mn        = m[:34]
                             ,sc        = s[:34]
                             ,forecast = True
                             )
    low1,low2,low3,med,high3,high2,high1 = np.nanpercentile( predictions["forecast"], [2.5,10,25,50,75,90,97.5], axis=0 )

    times = np.arange(34)
    # plt.plot(times,  jnp.append( y[~np.isnan(y)], med))
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low1),jnp.append( y[~np.isnan(y)], high1),alpha=0.20,color="blue")
    # plt.fill_between(times,jnp.append( y[~np.isnan(y)], low2),jnp.append( y[~np.isnan(y)], high2),alpha=0.20,color="blue")

    past_data = hosp_data.loc[(hosp_data.location=="42") ]
    past_data = pd.pivot_table(index= ["model_week"], columns = ["season"], values = ["value"], data = past_data)
 
    
    fig, ax = plt.subplots()
    
    plt.plot(times,med, color="red")
    plt.fill_between(times,low1,high1,alpha=0.20,color="red")
    plt.fill_between(times,low2,high2,alpha=0.20,color="red")
    plt.fill_between(times,low3,high3,alpha=0.20,color="red")

    ax.plot( np.arange(34), y)
    ax.scatter( np.arange(34)[:10], y_full[:10] , s=50, edgecolors='white', linewidths=2, zorder=3)
    ax.scatter( np.arange(34)[10:], y_full[10:] , s=50, color="blue")

    plt.plot(past_data.values,color="0.40")
    
    ax.set_xlabel("MMWR Week")
    ax.set_ylabel("Inc. Hosps. for PA")

    ax.set_xticks([0,9,22,32])
    ax.set_xticklabels(["40","50","10","20"])

   
    plt.show()
 
