#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

import scienceplots


class compartment_forecast_with_GP(object):
    #--
    def __init__(self
                 , N                 = None
                 , y                 = None
                 , X                 = None 
                 , times             = None
                 , start             = None
                 , end               = None
                 , infectious_period = None):
        
        self.N                 = N
        self.times             = times
        self.infectious_period = infectious_period

        if times is not None:
            self.start = min(times)
            self.end   = max(times)
        else:
            self.start, self.end   = start,end

        self.y = y

        if y is not None:
            self.nobs = np.min(np.argwhere(np.isnan(y)))
        else:
            self.nobs = None

        self.X = X
            
    #--     
    def simulation(self
                   , I0               = None
                   , repo             = None
                   , dt               = 1./7):

        import numpy as np
        
        #--simulation
        N                 = self.N
        infectious_period = self.infectious_period
        start, end        = self.start, self.end
        
        gamma   = 1./infectious_period
        eps     = np.finfo(float).eps

        S0, I0, R0, i0 = N-I0, I0, 0, I0
        y              = [ (S0,I0,R0,i0) ]

        times     = np.linspace(start,end,(end-start)*int(1./dt))
        T         = len(times)

        for t in times:
            S,I,R,i = y[-1]
            beta = repo*gamma

            infection = np.random.poisson( dt*(beta*S*I/N)  )
            recover   = np.random.poisson( dt*(gamma*I) )

            S = np.clip(S - infection         ,0,N)
            I = np.clip(I + infection-recover ,0,N)
            R = np.clip(R + recover           ,0,N)
            i+=infection

            y.append((S,I,R,i))

        S,I,R,i = zip(*y)
        i       = np.diff(i)

        return times,i, y

    def control_fit(self,dt = 1./7):
        import jax
        import jax.numpy as jnp
        
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        from diffrax import diffeqsolve, ODETerm, Dopri5, Heun, SaveAt
        
        def model(y=None,times=None,N=None):
            import jax.numpy as jnp
            
            def f(t, y, args):
                S,I,R,i = y
                repo, gamma, N = args

                beta    = repo*gamma

                dS = -beta*S*I/N
                dI =  beta*S*I/N - gamma*I
                dR =  gamma*I
                di =  beta*S*I/N 
                return jnp.array([dS,dI,dR,di])

            I0 = numpyro.sample( "I0", dist.Beta(1,1) )
            I0 = N*I0

            infectious_period = self.infectious_period
            gamma             = 1./infectious_period

            repo              = numpyro.sample("repo", dist.Beta(1,1))
            repo              = 5*repo 

            saves             = SaveAt(ts = jnp.arange(-1.,self.end+1,1) )

            term     = ODETerm(f)
            solver   = Heun()
            y0       = jnp.array([N-I0,I0,0,I0])
            solution = diffeqsolve(term
                                   , solver
                                   , t0     = -1
                                   , t1     = self.end+1
                                   , dt0    = dt
                                   , y0     = y0
                                   , saveat = saves
                                   , args   = (repo,gamma,N)
                                   )

            times    = solution.ts[1:]
            cinc     = solution.ys[:,-1]
            inc      = jnp.diff(cinc)

            nobs  = self.nobs
            inc   = numpyro.deterministic("inc", inc)

            resid = (y.reshape(-1,)-inc)[:nobs]

            resid_center   = jnp.nanmean(resid)
            centered_resid = resid-resid_center

            #--noise
            #--Kernel
            def rbf_kernel_ard(X1, X2, amplitude, lengthscales):
                """
                X1, X2: (n1, D), (n2, D)
                lengthscales: (D,) array for each input dimension
                """
                X1_scaled = X1 / lengthscales
                X2_scaled = X2 / lengthscales
                dists     = jnp.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :])**2, axis=-1)
                return amplitude**2 * jnp.exp(-0.5 * dists)

            import jax.numpy as jnp

            def random_walk_kernel(X, X2=None, variance=1.0):
                """
                Random Walk (Brownian motion) kernel.

                Args:
                    X: [n,1] array of input locations.
                    X2: [m,1] array of second input locations (or None for auto-kernel).
                    variance: scalar variance σ².

                Returns:
                    [n,m] kernel matrix
                """
                if X2 is None:
                    X2 = X
                return variance * jnp.minimum(X, X2.T)

            #--The implication is that the first column of X is an integer that denotes
            #--time step. For Example (time 0,1,2,3,4,...,end).
            #-- A Random walk kernel is applied to the first column and RBF to remaining columns

            noise = numpyro.sample("noise", dist.Beta(1.,1.) )
            
            ncols = X.shape[-1]
            rw_var = numpyro.sample("rw_var", dist.HalfCauchy(1.) )
            K1     = random_walk_kernel(X[:,0].reshape(-1,1), X[:,0].reshape(-1,1), rw_var )

            if ncols>1:
                amp   = numpyro.sample("amp"  , dist.Beta(1.,1.) )
                leng  = numpyro.sample("leng" , dist.HalfCauchy(1.) )
                K2    = rbf_kernel_ard(X[:,1:].reshape(-1,ncols-1), X[:,1:].reshape(-1,ncols-1), amp, leng )

                K = K1+K2
            else:
                K=K1
            #---------------------------------------------------------------------------------------
                
            KOO   = K[:nobs ,:nobs] + (noise) * jnp.eye(nobs) 
            KTT   = K[nobs: ,nobs:]
            KOT   = K[:nobs ,nobs:]

            training_resid = numpyro.sample( "training_resid", dist.MultivariateNormal(0, KOO ) )

            #--likelihood
            numpyro.sample( "likelihood"
                            , dist.Poisson( inc[:nobs].reshape(-1,) + (resid_center +  training_resid).reshape(-1,))
                            , obs = y[:nobs].reshape(-1,)  )

            L     = jnp.linalg.cholesky(KOO + 1e-5 * jnp.eye(nobs))
            alpha = jax.scipy.linalg.solve_triangular(L, centered_resid, lower=True)
            alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)

            mean = KOT.T @ alpha

            v   = jax.scipy.linalg.solve_triangular(L, KOT, lower=True)
            cov = KTT - v.T @ v

            fitted_resid = numpyro.sample("fitted_resid", dist.MultivariateNormal(  mean  ,covariance_matrix=cov) )
            final_resid  = jnp.concatenate([resid[:nobs],fitted_resid]) + resid_center 
            
            yhat         =  numpyro.deterministic( "yhat", inc.reshape(-1,) + final_resid.reshape(-1,) )

        mcmc = MCMC(NUTS(model, max_tree_depth=3), num_warmup=4*10**3, num_samples=5*10**3)
        mcmc.run(jax.random.PRNGKey(1)
                 , y     = jnp.array(self.y)
                 , times = jnp.array(self.times)
                 , N     = self.N
        )
        
        mcmc.print_summary()
        samples = mcmc.get_samples()

        incs = samples["yhat"]
        
        self.samples = samples
        return times,incs,samples

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
    weekly_infections[10:] = np.nan

    #--time paraemters
    start,end = min(weeks), max(weeks)

    #--Control model only uses X = time in the kernel
    X     = np.arange( end+1 ).reshape(-1,1)

    #--model fit for control
    framework = compartment_forecast_with_GP(N       = 1000
                                             , times = weeks
                                             , y     = weekly_infections
                                             , X     = X
                                             , infectious_period = 2)
 
    times,infections,samples = framework.control_fit()

    

    colors = sns.color_palette("tab10",2)
    plt.style.use("science")
    
    fig, ax = plt.subplots()
    ax.scatter(weeks, full_weekly_infections,s=8, color="black")
    ax.set_xlabel("MMWR week", fontsize=8)
    ax.set_ylabel("Incident cases", fontsize=8)

    ax.axvline(10)
    
    lower1,lower2,middle,upper2,upper1 = np.percentile(infections,[2.5,25,50,75,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[0])
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[0])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[0])
    
    plt.show()
        

    

    

