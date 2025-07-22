#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class model(object):
    def __init__(self,y, prior_parameters_dataset, season, location):
        self.y                        = y
        self.T                        = len(y)
        self.season                   = season
        self.location                 = location
        self.prior_parameters_dataset = prior_parameters_dataset

    def d_generalized_logistic(self,t, A, K, B, M, Q=1.0, nu=1.0):
        import jax
        import jax.numpy as jnp
        
        exp_term = jnp.exp(-B * (t - M))
        denom = (1 + Q * exp_term)
        return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

    def d2_generalized_logistic(self,t, A, K, B, M, Q=1.0, nu=1.0):
        import jax.numpy as jnp

        exp_term = jnp.exp(-B * (t - M))           # e^{-B(t - M)}
        denom = 1 + Q * exp_term                   # 1 + Q e^{-B(t - M)}
        r = 1 / nu + 1

        base = ((K - A) * B * Q * exp_term) / (nu * denom**r)
        correction = -B + ((r * Q * B**2 * exp_term) / denom)

        return base * correction

    def build_set_of_curves(self):
        import numpy as np
        import pandas as pd
        
        d      = self.prior_parameters_dataset

        def create_curve(row):
            d_generalized_logistic = self.d_generalized_logistic
            times  = np.arange(len(self.y))
            inc = -d_generalized_logistic(times
                                          ,float(row.A.iloc[0])
                                          ,float(row.K.iloc[0])
                                          ,float(row.B.iloc[0])
                                          ,float(row.M.iloc[0])
                                          ,float(row.Q.iloc[0])
                                          ,float(row.nu.iloc[0]))
            d = pd.DataFrame({"value": inc, "t":times})
            return d

        curves = d.groupby(["location","location_name","season"]).apply(create_curve).reset_index()
        curves = curves[ ["location","location_name","season","t","value"] ]

        self.set_of_curves = curves
        return curves

    def select_curves_to_use(self):
        import pandas as pd

        curves   = self.set_of_curves
        location = self.location
        selected_curves = pd.pivot_table(  index     = ['location',"season"] #<--this identifies each (season, location) pair
                                           , columns = ['t']
                                           , values  = ['value']
                                           , data    = curves.loc[curves.location==location] ).to_numpy()
        self.selected_curves = selected_curves
        return selected_curves


    def build_prior_for_params(self):
        import numpy as np
        import pandas as pd
        
        d        = self.prior_parameters_dataset
        location = self.location
        
        priors_for_state = d.loc[(d.location == location) & (d.season!=self.season)]
        prior_params     = {}
        for col_name in priors_for_state[ ["B","M","Q","nu","N"] ]:
            v = np.log(priors_for_state[col_name])
            prior_params[col_name] = ( v.mean(), v.std())
        self.prior_parameters_for_training = prior_params
        return prior_params

    def train(self):
        import numpyro
        numpyro.enable_x64(True)
        import numpyro.distributions as dist
        from   numpyro.infer import Predictive
        import jax
        import jax.numpy as jnp

        from numpyro.infer import MCMC, NUTS#, init_to_value, init_to_median, init_to_sample,init_to_uniform

        def genLogModel(  y                = None
                        , tobs             = None
                        , priors           = {}
                        , mean_prior_curve = None
                        , std_prior_curve  = None
                        , forecast         = False):

            #--prelim data and functions needed
            d_generalized_logistic = self.d_generalized_logistic

            times    = jnp.arange(1,len(y)+1)
            y        = y.reshape(-1,)
            
            #--setup priors for gLogistic regression parameters
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

            if "N" in priors:
                m,s = priors["N"]
                log_N  = numpyro.sample( "N", dist.Normal(m,2))
                N      = jnp.exp(log_N)
            else:
                N      = numpyro.sample( "N", dist.LogNormal(jnp.log(10*10**3),1) )

            #--The B parameter is time-varying and controls how fast the epidemic moves over time 
            sigma_f      = numpyro.sample("sigma_f"     , dist.LogNormal(-1,1))
            random_walk  = numpyro.sample("random_walk" , dist.GaussianRandomWalk( sigma_f, len(y) ) )
            params["B"]  = params["B"] + random_walk


            #--dS is the proportion of incident casesper week
            dS = jax.vmap(d_generalized_logistic, in_axes = (0,)+ (None,)*2 + (0,) + (None,)*3  )( times
                                                                                                   , params["A"]
                                                                                                   , params["K"]
                                                                                                   , params["B"]
                                                                                                   , params["M"]
                                                                                                   , params["Q"]
                                                                                                   , params["nu"]
                                                                                                  )

            #--create proportion of incident cases, number of incident cases, and extract attributes from this curve
            inc_p      = numpyro.deterministic("inc_p"     , -dS.reshape(-1,) )
            inc        = numpyro.deterministic("inc"       , jnp.clip( N*inc_p ,10**-6,jnp.inf ) )
            peak_time  = numpyro.deterministic("peak_time" , jnp.argmax(inc_p) )

            #--This is a proposed correction for the Poisson
            inc_corrected = jnp.clip( N*inc_p.reshape(-1,) - (1./3) + (1./50), 10**-6,jnp.inf)

            #--Data likelihood
            nany       = ~jnp.isnan(y.reshape(-1,))
            with numpyro.handlers.mask(mask=nany):
                numpyro.sample("ll__from_data" , dist.Poisson(inc), obs = y)
                
            #--Prior influenza from the shape of the curve where our current data is not yet observed
            numpyro.sample("ll__from_prior", dist.Normal(mean_prior_curve[tobs:],std_prior_curve[tobs:]), obs = inc_p[tobs:] )

            #--Forecast
            if forecast:
                numpyro.sample("forecast", dist.Poisson(inc))

        y                = self.y
        tobs             = np.min(np.argwhere(np.isnan(y))[0])

        try:
            priors           = self.prior_parameters_for_training
        except AttributeError:
            self.build_prior_for_params()
            priors           = self.prior_parameters_for_training

        try:
            selected_curves = self.select_curves_to_use()
        except AttributeError:
            self.build_set_of_curves()
            selected_curves = self.select_curves_to_use()

        mean_prior_curve = selected_curves.mean(0)
        std_prior_curve  = selected_curves.std(0)

        #--MCMC training
        mcmc = MCMC(NUTS(genLogModel), num_warmup=10*10**3, num_samples=10*10**3)
        mcmc.run(jax.random.PRNGKey(1)
                 , y                = y
                 , tobs             = tobs
                 , priors           = priors
                 , mean_prior_curve = mean_prior_curve
                 , std_prior_curve  = std_prior_curve)

        mcmc.print_summary()

        training_samples = mcmc.get_samples()

        #--Forecast
        predictive  = Predictive(genLogModel
                                 , posterior_samples = training_samples
                                 , return_sites = ["K","B","M","Q","nu","N","forecast","peak_time"])

        predictions = predictive(jax.random.PRNGKey(10)
                                 , forecast         = True
                                 , y                = y
                                 , tobs             = tobs
                                 , priors           = priors
                                 , mean_prior_curve = mean_prior_curve
                                 , std_prior_curve  = std_prior_curve)
        #--Save important objects
        self.mcmc        = mcmc
        self.predictions = predictions

        return predictions["forecast"]


if __name__ == "__main__":

    import jax.numpy as jnp
   
    prior_parameters_dataset = pd.read_csv("./models/prior_params.csv")

    hosp_data = pd.read_csv("./analysis_data/us_hospital_data.csv")

    T = 13
    
    y_full = hosp_data.loc[(hosp_data.location=="42") & (hosp_data.season=="2024/2025"), "value"].values
    y      = y_full.copy()
    y[T:]  = np.nan

    model = model(y = y
                  , prior_parameters_dataset = prior_parameters_dataset
                  , location = "42"
                  , season = "2024/2025")
    forecasted_inc = model.train()
    
   
    low1,low2,low3,med,high3,high2,high1 = np.nanpercentile( forecasted_inc, [2.5,10,25,50,75,90,97.5], axis=0 )

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
    ax.scatter( np.arange(34)[:10], y_full[:T] , s=50, edgecolors='white', linewidths=2, zorder=3)
    ax.scatter( np.arange(34)[10:], y_full[T:] , s=50, color="blue")

    plt.plot(past_data.values,color="0.40")
    
    ax.set_xlabel("MMWR Week")
    ax.set_ylabel("Inc. Hosps. for PA")

    ax.set_xticks([0,9,22,32])
    ax.set_xticklabels(["40","50","10","20"])

   
    plt.show()
 
