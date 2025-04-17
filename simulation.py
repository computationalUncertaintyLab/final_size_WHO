#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

import scienceplots

def simulation(N=1000,I0=1,repo=2,infectious_period=2,start=0,end=32):
    #--simulation
    gamma             = 1./infectious_period
    eps = np.finfo(float).eps

    S0, I0, R0, i0 = N-I0, I0, 0, I0
    y     = [ (S0,I0,R0,i0) ]

    dt        = 1./7
    times     = np.linspace(start,end,(end-start)*7)
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


def control_fit(y,times,N):

    def model(y,times,N):
        from diffrax import diffeqsolve, ODETerm, Dopri5, Heun, SaveAt
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

        infectious_period = 2
        gamma             = 1./infectious_period

        repo              = numpyro.sample("repo", dist.Beta(1,1))
        repo              = 5*repo 

        end               = 32
        print(end)
        
        dt                = 1./7
        saves             = SaveAt(ts = jnp.arange(-1.,end,1) )
        
        term     = ODETerm(f)
        solver   = Heun()#Dopri5()
        y0       = jnp.array([N-I0,I0,0,I0])
        solution = diffeqsolve(term
                               , solver
                               , t0     = -1
                               , t1     = end+1
                               , dt0    = dt
                               , y0     = y0
                               , saveat = saves
                               , args   = (repo,gamma,N)
                               )

        times = solution.ts[1:]
        cinc  = solution.ys[:,-1]
        inc   = jnp.diff(cinc)
        
        nobs = len(y)
        inc   = numpyro.deterministic("inc", inc)
        inc_O = inc[:nobs].reshape(-1,)
        
        resid = y.reshape(-1,)-inc_O

        #--noise
        #--Kernel
        def rbf_kernel_ard(X1, X2, amplitude, lengthscales):
            """
            X1, X2: (n1, D), (n2, D)
            lengthscales: (D,) array for each input dimension
            """
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            dists = jnp.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :])**2, axis=-1)
            return amplitude**2 * jnp.exp(-0.5 * dists)

        T     = 32
        X     = jnp.arange( 32 ).reshape(-1,1)

        X = jnp.concatenate([ jnp.arange(10), jnp.linspace(0.1,32-0.1,50) ]).reshape(-1,1)
        
        print("Number of obs {:d}".format(nobs))

        noise = numpyro.sample("noise", dist.HalfNormal(1.) )
        amp   = numpyro.sample("amp"  , dist.HalfCauchy(1.) )
        leng  = numpyro.sample("leng" , dist.HalfNormal(1.) )

        jitter = 1e-5
        
        K     = rbf_kernel_ard(X, X, amp,leng ) 
        KOO   = K[:nobs,:nobs] + (noise) * jnp.eye(nobs) + jitter * jnp.eye(KOO.shape[0])
        KTT   = K[nobs:,nobs:] #+ jitter * jnp.eye(KTT.shape[0])
        KOT   = K[:nobs,nobs:]

        training_resid = numpyro.sample( "training_resid", dist.MultivariateNormal(0, KOO ) )
        
        numpyro.sample( "likelihood", dist.Poisson( inc[:nobs].reshape(-1,) + training_resid.reshape(-1,) ), obs = y.reshape(-1,)  )  

        #--prediction step
        #KOO_CH = jnp.linalg.cholesky(KOO)
        #alpha = jax.scipy.linalg.cho_solve((L, True), b)
        #L = jax.scipy.linalg.cho_solve((KOO, True), KOT)
        
        L            = KOT.T @ jnp.linalg.inv(KOO)
        cov          = KTT - L @ KOT 
        fitted_resid = numpyro.sample("fitted_resid", dist.MultivariateNormal( L @ resid  ,covariance_matrix=cov) )

        final_resid  = jnp.concatenate([resid,fitted_resid]) 
        
        yhat =  numpyro.deterministic( "yhat", inc.reshape(-1,) + final_resid.reshape(-1,) )
        

    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    mcmc = MCMC(NUTS(model, max_tree_depth=4), num_warmup=4*10**3, num_samples=5*10**3)
    mcmc.run(jax.random.PRNGKey(1)
             , y     = jnp.array(y)
             , times = jnp.array(times)
             , N     = N
             )
    mcmc.print_summary()
    samples = mcmc.get_samples()

    #--now run tau-leap
    I0s   = N*samples["I0"]
    repos = 5*samples["repo"]

    def gillespie(x,y):
        times,i,_ = simulation(N  = N
                               ,I0=x,repo=y
                               ,infectious_period=2
                               ,start=0,end=32)
        i_weekly  = i.reshape(32,-1).sum(-1)
        return i_weekly

    from joblib import Parallel, delayed
    i_samples = Parallel(n_jobs=10)(delayed(gillespie)(x,y) for x,y in zip(I0s,repos) )

    incs = np.array(i_samples)
        
    return times,incs,samples

if __name__ == "__main__":

    times,i,_ = simulation(N=1000,I0=1,repo=2,infectious_period=2,start=0,end=32)

    weeks = np.arange(0,32)
    i_weekly  = i.reshape(32,-1).sum(-1)

    #--model fit for control
    times,incs,_ = control_fit( i_weekly[:10], weeks[:10], 1000 )

    colors = sns.color_palette("tab10",2)
    plt.style.use("science")
    
    fig, ax = plt.subplots()
    ax.scatter(weeks, i_weekly,s=8, color="black")
    ax.set_xlabel("MMWR week", fontsize=8)
    ax.set_ylabel("Incident cases", fontsize=8)

    lower1,lower2,middle,upper2,upper1 = np.percentile(incs,[2.5,25,50,75,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[0])
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[0])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[0])

    
    plt.show()
        

    

    

