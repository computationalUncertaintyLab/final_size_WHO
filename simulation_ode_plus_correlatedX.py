# mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

import scienceplots

class compartment_forecast_with_GP(object):
    # Initialize the forecasting framework
    def __init__(self
                 , N=None                 # Total population
                 , y=None                 # Observed incident cases (with missing values)
                 , X=None                 # Covariate matrix for GP kernel (e.g. time or other predictors)
                 , times=None             # Array of time points
                 , start=None, end=None   # Start and end time (used if times is None)
                 , infectious_period=None):  # Fixed infectious period (used to derive gamma)
        
        self.N = N
        self.times = times
        self.infectious_period = infectious_period

        # Set time boundaries
        if times is not None:
            self.start = min(times)
            self.end   = max(times)
        else:
            self.start, self.end = start, end

        self.y = y

        # Find first missing value in y to determine training length
        if y is not None:
            self.nobs = np.min(np.argwhere(np.isnan(y)))
        else:
            self.nobs = None

        self.X = X

    # Simulate epidemic trajectories with a stochastic SIR model
    def simulation(self, I0=None, repo=None, dt=1./7):
        import numpy as np

        N = self.N
        infectious_period = self.infectious_period
        start, end = self.start, self.end
        gamma = 1. / infectious_period

        S0, I0, R0, i0 = N - I0, I0, 0, I0
        y = [(S0, I0, R0, i0)]

        # Time grid for simulation
        times = np.linspace(start, end, (end - start) * int(1. / dt))

        for t in times:
            S, I, R, i = y[-1]
            beta = repo * gamma

            # Simulate infections and recoveries using Poisson noise
            infection = np.random.poisson(dt * (beta * S * I / N))
            recover = np.random.poisson(dt * (gamma * I))

            # Update compartments (clipped to [0, N])
            S = np.clip(S - infection, 0, N)
            I = np.clip(I + infection - recover, 0, N)
            R = np.clip(R + recover, 0, N)
            i += infection

            y.append((S, I, R, i))

        S, I, R, i = zip(*y)
        i = np.diff(i)  # Daily incident cases

        return times, i, np.random.poisson(y)

    # Fit model to control scenario using NumPyro and GP residuals
    def control_fit(self, dt=1./7):
        import jax
        import jax.numpy as jnp

        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from diffrax import diffeqsolve, ODETerm, Heun, SaveAt

        def model(y=None, X = None, times=None, N=None, forecast = False):
            # Define SIR ODE system
            def f(t, y, args):
                S, I, R, i = y
                repo, gamma, N = args
                beta = repo * gamma
                dS = -beta * S * I / N
                dI = beta * S * I / N - gamma * I
                dR = gamma * I
                di = beta * S * I / N
                return jnp.array([dS, dI, dR, di])

            # Sample initial infectious proportion and scale to population
            I0 = numpyro.sample("I0", dist.Beta(1, 1))
            I0 = N * I0

            infectious_period = self.infectious_period
            gamma = 1. / infectious_period

            # Sample reproduction rate (scaled by 5)
            repo = numpyro.sample("repo", dist.Beta(1, 1))
            repo = 5 * repo

            # Solve ODE forward in time
            saves    = SaveAt(ts=jnp.arange(-1., self.end + 1, 1))

            term     = ODETerm(f)
            solver   = Heun()
            
            y0       = jnp.array([N - I0, I0, 0, I0])
            solution = diffeqsolve(term, solver, t0=-1, t1=self.end + 1, dt0=dt, y0=y0, saveat=saves, args=(repo, gamma, N))

            times = solution.ts[1:]
            cinc  = solution.ys[:, -1]  # cumulative incidence

            inc   = jnp.diff(cinc)      # incidence
            inc  = numpyro.deterministic("inc", inc)

            #--setup residual vector
            nobs = self.nobs

            #resid_center   = jnp.nanmean(y)
            resid          = (y.reshape(-1,) - inc )[:nobs]

            # Define RBF kernel (optional for multiple covariates)
            def rbf_kernel_ard(X1, X2, amplitude, lengthscales):
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                dists = jnp.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :])**2, axis=-1)
                return amplitude**2 * jnp.exp(-0.5 * dists)

            def random_walk_kernel(X, X2=None, variance=1.0):
                if X2 is None:
                    X2 = X
                return variance * jnp.minimum(X, X2.T)

            noise  = numpyro.sample("noise", dist.HalfCauchy(1.))
            ncols  = X.shape[-1]
            
            rw_var = numpyro.sample("rw_var", dist.HalfCauchy(1.))
            K1     = random_walk_kernel(X[:, 0].reshape(-1, 1), variance=rw_var)

            # Optionally add RBF kernel if extra features exist
            if ncols > 1:
                amp  = numpyro.sample("amp", dist.Beta(1., 1.))
                leng = numpyro.sample("leng", dist.HalfCauchy(1.))
                K2   = rbf_kernel_ard(X[:, 1:], X[:, 1:], amp, leng)
                K    = K1 + K2
            else:
                K = K1

            # Compute submatrices for GP residual conditioning
            KOO = numpyro.deterministic("KOO", K[:nobs, :nobs] + noise * jnp.eye(nobs))
            KTT = numpyro.deterministic("KTT", K[nobs:, nobs:]                        )
            KOT = numpyro.deterministic("KOT", K[:nobs, nobs:]                        )

        
            # Poisson observation model on residual-corrected prediction
            training_resid = numpyro.sample("training_resid", dist.MultivariateNormal(0, covariance_matrix = KOO))
            numpyro.sample("likelihood",
                           dist.Poisson(inc[:nobs] + training_resid),
                           obs=y[:nobs])
            
            if forecast:
                L     = jnp.linalg.cholesky(KOO + 1e-5 * jnp.eye(nobs))
            
                alpha = jax.scipy.linalg.solve_triangular(L  , resid, lower=True)
                alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)

                mean = KOT.T @ alpha

                v    = jax.scipy.linalg.solve_triangular(L, KOT, lower=True)
                cov  = KTT - v.T @ v

                fitted_resid = numpyro.sample("fitted_resid", dist.MultivariateNormal(mean, covariance_matrix=cov))
                final_resid  = jnp.concatenate([resid[:nobs], fitted_resid]) 

                yhat_mean = numpyro.deterministic("yhat_mean", inc + final_resid)
                yhat_obs  = numpyro.sample("yhat_obs", dist.Poisson(jnp.clip( inc + final_resid,10**-5,jnp.inf) ) )
                yhat      = numpyro.deterministic("yhat", jnp.concatenate( [yhat_mean[:nobs], yhat_obs[nobs:]])) 

                print(len(yhat))
                
            #yhat      = numpyro.sample("yhat", dist.Normal( inc + final_resid, jnp.clip(inc + final_resid,10**-5,jnp.inf) ) )
            

        # Run MCMC with NUTS sampler
        mcmc = MCMC(NUTS(model, max_tree_depth=3), num_warmup=5000, num_samples=5000)
        mcmc.run(jax.random.PRNGKey(1)
                 , y=jnp.array(self.y)
                 ,X        = jnp.array(self.X)
                 , times=jnp.array(self.times)
                 , N=self.N)

        mcmc.print_summary()
        samples = mcmc.get_samples()
        #incs    = samples["yhat"]

        # Generate posterior predictive samples using previously drawn MCMC samples
        from numpyro.infer import Predictive

        # Define model as used in control_fit (reusing trace)
        predictive = Predictive(model
                                ,posterior_samples=samples
                                ,return_sites=["yhat"])

        preds = predictive(jax.random.PRNGKey(2)
                           ,y        = jnp.array(self.y)
                           ,X        = jnp.array(self.X)
                           ,times    = jnp.array(self.times)
                           ,N        = self.N
                           ,forecast = True)

        yhats = preds["yhat"]
        
        self.samples = samples
        return times, yhats, samples

def create_correlated_column(y,rho):
    # Standardize y
    y_std = (y - np.mean(y)) / np.std(y)

    # Generate independent noise
    z = np.random.normal(0, 1, size=len(y))

    # Create x with desired correlation
    x = rho * y_std + np.sqrt(1 - rho**2) * z

    # Optionally rescale x
    correlated_column = (x * np.std(y) + np.mean(y)).reshape(-1,1)

    return correlated_column
    
    
if __name__ == "__main__":

    np.random.seed(1010)

    #--this simulation is at a daily temporal scale
    framework = compartment_forecast_with_GP(N                   = 1000
                                             , start             = 0
                                             , end               = 32 
                                             , infectious_period = 2)
    times, infections, all_states = framework.simulation(I0=5,repo=2)

    #--aggregating up to week temporal scale
    weeks              = np.arange(0,32)
    weekly_infections  = infections.reshape(32,-1).sum(-1)

    #--lets further assume we know only the first 10 time units
    full_weekly_infections = weekly_infections
    
    weekly_infections = np.array([ float(x) for x in weekly_infections])
    

    #--time paraemters
    start,end = min(weeks), max(weeks)+1

    #--add the correlated 0.90 column here
    X     = np.arange( 1,end+1 ).reshape(-1,1)
    correlated_column = create_correlated_column(weekly_infections, 0.90)
    Xaugmented90 = np.hstack([X,correlated_column])

    #--add the correlated 0.10 column here
    X     = np.arange( 1,end+1 ).reshape(-1,1)
    correlated_column = create_correlated_column(weekly_infections, 0.10)
    Xaugmented10 = np.hstack([X,correlated_column])

    #--block a bunch of known values
    weekly_infections[10:] = np.nan

    colors = sns.color_palette("tab10",3)
    plt.style.use("science")
    
    fig, ax = plt.subplots()
    ax.scatter(weeks, full_weekly_infections,s=8, color="black")
    ax.set_xlabel("MMWR week", fontsize=8)
    ax.set_ylabel("Incident cases", fontsize=8)

    ax.axvline(9,color="black",ls="--")

    #--wit the correlated column
    framework1 = compartment_forecast_with_GP(N       = 1000
                                             , times = weeks
                                             , y     = weekly_infections
                                             , X     = Xaugmented90
                                             , infectious_period = 2)
 
    times,infections,samples = framework1.control_fit()
    
    lower1,lower2,lower3,middle,upper3,upper2,upper1 = np.percentile(infections,[2.5, 10, 25,50,75,90,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[0])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[0])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[0])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[0], label="Signal 90%")


    #--wit the correlated column
    framework2 = compartment_forecast_with_GP(N       = 1000
                                             , times = weeks
                                             , y     = weekly_infections
                                             , X     = Xaugmented10
                                             , infectious_period = 2)
 
    times,infections,samples = framework2.control_fit()
    
    lower1,lower2,lower3,middle,upper3,upper2,upper1 = np.percentile(infections,[2.5, 10, 25,50,75,90,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[1])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[1])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[1])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[1], label="Signal 10%")

    
    #--without the correlated column
    framework3 = compartment_forecast_with_GP(N       = 1000
                                             , times = weeks
                                             , y     = weekly_infections
                                             , X     = X
                                             , infectious_period = 2)
 
    times,infections,samples = framework3.control_fit()
    
    lower1,lower2,lower3,middle,upper3,upper2,upper1 = np.percentile(infections,[2.5, 10, 25,50,75,90,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[2])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[2])
    #ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[2])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[2], label="No extra signal")
 

    plt.legend(frameon=False)
    plt.show()
        

    

    

