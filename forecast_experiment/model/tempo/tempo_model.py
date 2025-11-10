#mcandrew

class tempo_model4(object):
    """
    Tempo model with hierarchical fitting across seasons.
    
    Differences from tempo_model3:
    - fit_past_seasons: Hierarchical model across all seasons
    - fit_new_season: Same as tempo_model3
    """
    
    def __init__(self, y, X, N, key, nobs=0):
        self.y = y.copy()
        #self.X = X.copy() 
        self.N = N.copy()
        self.nobs = nobs
        self.key = key

    def fit_past_seasons(self):
        """Fit hierarchical model across all past seasons."""
        import jax
        import jax.numpy as jnp
        from jax import vmap
        from jax.scipy.special import logit, expit
        from jax import lax

        jax.clear_caches()
        
        import numpy as np
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer import Predictive

        def model(y, X=None, N=None, nobs=0):
            """Hierarchical model across seasons."""
            nseasons, ntimes = y.shape
            times = np.arange(ntimes)
            eps = 10**-5

            def d_generalized_logistic(t, delta, B, M, Q=1.0, nu=1.0):
                """Generalized logistic derivative (epidemic curve)."""
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return (-delta * B * Q * exp_term) / (nu * denom**(1/nu + 1))
            
            # ========================================================================
            # HIERARCHICAL PRIORS: Population-level hyperparameters
            # ========================================================================
            
            delta_pop_mu = numpyro.sample("delta_pop_mu", dist.Normal( 0, 2))              
            M_pop_mu = numpyro.sample("M_pop_mu"        , dist.Normal( jnp.log(33./2), 2))   
            B_pop_mu = numpyro.sample("B_pop_mu"        , dist.Normal( 0, 2))                 
            nu_pop_mu = numpyro.sample("nu_pop_mu"      , dist.Normal( 0, 2))                 
            Q_pop_mu = numpyro.sample("Q_pop_mu"        , dist.Normal( 0, 2))                 
            
            # AR(1) process parameters - population level
            rho_pop_mu = numpyro.sample("rho_pop_mu", dist.Normal(2.0, 1.0))  # logit scale → ~0.88 after expit
            sigma_ar_pop_mu = numpyro.sample("sigma_ar_pop_mu", dist.Normal(jnp.log(0.05), 0.5))  # log scale, small
            
            # Population-level standard deviations (between-season variability)
            # Use HalfCauchy for heavier tails - allows more variability when needed
            delta_pop_sd = numpyro.sample("delta_pop_sd", dist.HalfNormal(1.0))
            M_pop_sd = numpyro.sample("M_pop_sd"        , dist.HalfNormal(1.0))
            B_pop_sd = numpyro.sample("B_pop_sd"        , dist.HalfNormal(1.0))
            nu_pop_sd = numpyro.sample("nu_pop_sd"      , dist.HalfNormal(1.0))
            Q_pop_sd = numpyro.sample("Q_pop_sd"        , dist.HalfNormal(1.0))
            
            # AR(1) population SDs
            rho_pop_sd      = numpyro.sample("rho_pop_sd", dist.HalfNormal(0.5))
            sigma_ar_pop_sd = numpyro.sample("sigma_ar_pop_sd", dist.HalfNormal(0.5))
            
            # Weather covariates - population level
            #ncovs = X.shape[-1] 
            #F_pop_mu = numpyro.sample("F_pop_mu", dist.Normal(0, 1).expand([ncovs]))
            #F_pop_sd = numpyro.sample("F_pop_sd", dist.HalfNormal(1.0).expand([ncovs]))
            
            # ========================================================================
            # SEASON-SPECIFIC PARAMETERS (hierarchical - NON-CENTERED)
            # Non-centered parameterization for better sampling efficiency
            # ========================================================================
            
            with numpyro.plate("seasons", nseasons):
                # Sample raw (standardized) parameters - these are "white noise"
                delta_season_raw = numpyro.sample("delta_season_raw", dist.Normal(0, 1))
                M_season_raw = numpyro.sample("M_season_raw", dist.Normal(0, 1))
                B_season_raw = numpyro.sample("B_season_raw", dist.Normal(0, 1))
                nu_season_raw = numpyro.sample("nu_season_raw", dist.Normal(0, 1))
                Q_season_raw = numpyro.sample("Q_season_raw", dist.Normal(0, 1))
                
                # AR(1) parameters - non-centered
                rho_season_raw = numpyro.sample("rho_season_raw", dist.Normal(0, 1))
                sigma_ar_season_raw = numpyro.sample("sigma_ar_season_raw", dist.Normal(0, 1))
                
                # F has shape (ncov,) for weather covariates
                #F_season_raw = numpyro.sample("F_season_raw", dist.Normal(0, 1).expand([ncovs]).to_event(1))
            
            # Transform from non-centered to centered (deterministic transformation)
            delta_season = numpyro.deterministic("delta_season", delta_pop_mu + delta_pop_sd * delta_season_raw)
            M_season = numpyro.deterministic("M_season", M_pop_mu + M_pop_sd * M_season_raw)
            B_season = numpyro.deterministic("B_season", B_pop_mu + B_pop_sd * B_season_raw)
            nu_season = numpyro.deterministic("nu_season", nu_pop_mu + nu_pop_sd * nu_season_raw)
            Q_season = numpyro.deterministic("Q_season", Q_pop_mu + Q_pop_sd * Q_season_raw)
            
            # AR(1) parameters - non-centered transformation
            rho_season      = numpyro.deterministic("rho_season", rho_pop_mu + rho_pop_sd * rho_season_raw)
            sigma_ar_season = numpyro.deterministic("sigma_ar_season", sigma_ar_pop_mu + sigma_ar_pop_sd * sigma_ar_season_raw)
            
            #F_season = numpyro.deterministic("F_season", F_pop_mu + F_pop_sd * F_season_raw)
            
            # Transform parameters to constrained scales
            delta_transformed = expit(delta_season)  # [0,1]
            M_transformed = jnp.exp(M_season)        # positive
            B_transformed = jnp.exp(B_season)        # positive
            nu_transformed = jnp.exp(nu_season)      # positive
            Q_transformed = jnp.exp(Q_season)        # positive
            
            # Transform AR(1) parameters
            rho_transformed      = expit(rho_season)  # ∈ (0,1) for stationarity
            sigma_ar_transformed = jnp.exp(sigma_ar_season)  # positive
            
            # ========================================================================
            # LIKELIHOOD: Vectorized computation (much faster than for loop)
            # ========================================================================
            
            #nseasons, ntimes_model, numcovs = X.shape
            
            # Vectorized epidemic curve generation (vmap over seasons)
            def compute_inc_for_season(delta, B, M, Q, nu):
                inc = -d_generalized_logistic(times, delta, B, M, Q, nu)
                return jax.scipy.special.logit(jnp.clip(inc, eps, 1-eps))
            
            # vmap over first dimension (seasons)
            logit_inc_all = vmap(compute_inc_for_season)(
                delta_transformed, B_transformed, M_transformed, Q_transformed, nu_transformed
            )  # Shape: (nseasons, ntimes)
            
            # Vectorized weather effects
            # def compute_weather_effects(X_s, F_s):
            #     weather_raw = X_s @ F_s.reshape(numcovs, 1)
            #     return 2.0 * jnp.tanh(weather_raw / 2.0) #weather_raw#jnp.clip(weather_raw, -2.0, 2.0)
            #weather_effects_all = vmap(compute_weather_effects)(X, F_season)  # (nseasons, ntimes, 1)
            
            # Add weather to baseline
            regression_outputs_all = logit_inc_all.reshape(nseasons, ntimes, 1) #+ weather_effects_all
            
            # AR(1) process - SEASON-SPECIFIC innovations
            # Each season gets its own AR(1) path with its own parameters
            
            # Sample AR innovations for all seasons (374 parameters)
            with numpyro.plate("ar_innovations_plate", nseasons):
                z_all = numpyro.sample("ar_z_seasons", dist.Normal(0, 1).expand([ntimes]).to_event(1))
            # Shape: (nseasons, ntimes)
            
            # Vectorized AR(1) process using vmap
            def compute_ar_drift(rho, sigma_ar, z):
                """Compute AR(1) drift for one season."""
                def ar_step(x_prev, z_t):
                    x_t = rho * x_prev + sigma_ar * z_t
                    return x_t, x_t
                _, ar_drift = lax.scan(ar_step, 0.0, z)
                return ar_drift
            
            # vmap over seasons (each season uses its own rho and sigma)
            ar_drift_all = vmap(compute_ar_drift)(rho_transformed, sigma_ar_transformed, z_all)
            # Shape: (nseasons, ntimes)
            
            # Combine regression + AR(1) drift
            process_for_p_all = expit(regression_outputs_all.squeeze(-1) + ar_drift_all)
            numpyro.deterministic("process_for_p_all", process_for_p_all)
            # Shape: (nseasons, ntimes)
            
            # Vectorized likelihood
            y_reshaped = y.reshape(nseasons, ntimes)
            N_reshaped = N.reshape(nseasons, ntimes)
            
            present_all = ~jnp.isnan(y_reshaped) & ~jnp.isnan(N_reshaped)
            
            # Concentration parameter for overdispersion
            conc = numpyro.sample("conc", dist.Gamma(10, 0.2))  # Mean=50, more stable than LogNormal
            
            # Simple vectorized likelihood (no nested plates for better performance)
            with numpyro.handlers.mask(mask=present_all.reshape(nseasons,ntimes)):
                numpyro.sample("y_seasons", 
                              dist.NegativeBinomial2(N_reshaped * process_for_p_all, conc),
                              obs=y_reshaped)
            
            # Store population-level parameters for later use
            numpyro.deterministic("delta_pop"   , expit(delta_pop_mu))
            numpyro.deterministic("M_pop"       , jnp.exp(M_pop_mu))
            numpyro.deterministic("B_pop"       , jnp.exp(B_pop_mu))
            numpyro.deterministic("nu_pop"      , jnp.exp(nu_pop_mu))
            numpyro.deterministic("Q_pop"       , jnp.exp(Q_pop_mu))
            numpyro.deterministic("rho_pop"     , expit(rho_pop_mu))
            numpyro.deterministic("sigma_ar_pop", jnp.exp(sigma_ar_pop_mu))

        # Run MCMC with increased target_accept_prob for better geometry handling
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=10*10**3, num_samples=10*10**3, num_chains=1)
        mcmc.run(self.key,
                 y=self.y,
                 X=None,
                 N=jnp.nan_to_num(self.N, nan=1),
                 nobs=self.nobs)
        mcmc.print_summary()
        
        print("Hierarchical MCMC completed!")
        samples = mcmc.get_samples()
        return samples

    def fit_new_season(self, prior_mus=None, prior_covs=None, forecast=False, N_pred=None, constraint_mu=None, constraint_sd=None):
        import jax
        import jax.numpy as jnp
        from jax import vmap
        from jax.scipy.special import logit, expit
        
        import numpy as np
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer import Predictive
        from numpyro.infer.reparam import TransformReparam
        from numpyro import handlers
        from jax import lax

        jax.clear_caches()

        def model(y, X=None, N=None, nobs=0, prior_mus=None, prior_covs=None, forecast=False, N_pred=None, constraint_mu=None, constraint_sd=None):
            nseasons, ntimes = y.shape
            times = np.arange(ntimes)
            eps = 10**-5

            def d_generalized_logistic(t, delta, B, M, Q=1.0, nu=1.0):
                """Generalized logistic derivative (epidemic curve)."""
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return (-delta * B * Q * exp_term) / (nu * denom**(1/nu + 1))

            ntimes = len(times)
            nseasons_prior, nparams = prior_mus.shape
            
            # Sample weights for different seasons (mixture of priors)
            season_weights = numpyro.sample("season_weights", dist.Dirichlet(jnp.ones(nseasons_prior)))

            # --- Mixture-of-Gaussians prior via base Gaussian + log-density correction ---
            def chol_reg(S):
                S = 0.5 * (S + S.T)
                evals = jnp.linalg.eigh(S)[0]
                eps_loc = 1e-6
                delta = jnp.maximum(0.0, -evals[0] + eps_loc)
                return jnp.linalg.cholesky(S + delta * jnp.eye(S.shape[0]))

            prior_mus  = jnp.asarray(prior_mus)
            prior_covs = jnp.asarray(prior_covs)
            Lks        = jax.vmap(chol_reg)(prior_covs)

            m0            = (season_weights[:, None] * prior_mus).sum(0)
            second_moment = (season_weights[:, None, None] *
                            (prior_covs + jnp.einsum('si,sj->sij', prior_mus, prior_mus))
                            ).sum(0)
            Sigma0        = 0.5 * ((second_moment - jnp.outer(m0, m0)) + (second_moment - jnp.outer(m0, m0)).T)
            T             = chol_reg(Sigma0)

            d = prior_mus.shape[1]
            eta = numpyro.sample("param_vec_white", dist.Normal(0., 1.).expand([d]).to_event(1))
            param_vec = numpyro.deterministic("param_vec", m0 + T @ eta)

            def mvn_logpdf_chol(x, mu, L):
                y = jax.scipy.linalg.solve_triangular(L, x - mu, lower=True)
                return -0.5 * (x.size * jnp.log(2 * jnp.pi) + 2 * jnp.sum(jnp.log(jnp.diag(L))) + jnp.dot(y, y))

            logps    = jax.vmap(lambda mu, Lk: mvn_logpdf_chol(param_vec, mu, Lk))(prior_mus, Lks)
            log_mix  = jax.scipy.special.logsumexp(jnp.log(season_weights + 1e-32) + logps)
            base_log = mvn_logpdf_chol(param_vec, m0, T)
            alpha_mog = 1.0
            numpyro.factor("mixture_prior", alpha_mog * (log_mix - base_log))
          
            # Extract parameters (AR instead of OU)
            delta_empirical = param_vec[0]
            M_empirical = param_vec[1]
            B_empirical = param_vec[2]
            nu_empirical = param_vec[3]
            Q_empirical = param_vec[4]
            rho_empirical = param_vec[5]
            sigma_ar_empirical = param_vec[6]
               
            # Sample individual priors
            delta_individual = numpyro.sample("delta_individual", dist.Normal(0, 1))
            M_individual = numpyro.sample("M_individual", dist.Normal(0, 1))
            B_individual = numpyro.sample("B_individual", dist.Normal(0, 1))
            nu_individual = numpyro.sample("nu_individual", dist.Normal(0, 1))
            Q_individual = numpyro.sample("Q_individual", dist.Normal(0, 1))
            rho_individual = numpyro.sample("rho_individual", dist.Normal(0, 1))
            sigma_ar_individual = numpyro.sample("sigma_ar_individual", dist.Normal(0, 1))

            noise_scale = 0.01
            delta    = numpyro.deterministic("delta", delta_empirical + noise_scale * delta_individual)
            M        = numpyro.deterministic("M", M_empirical + noise_scale * M_individual)
            B        = numpyro.deterministic("B", B_empirical + noise_scale * B_individual)
            nu       = numpyro.deterministic("nu", nu_empirical + noise_scale * nu_individual)
            Q        = numpyro.deterministic("Q", Q_empirical + noise_scale * Q_individual)

            noise_scale = 0.1
            rho      = numpyro.deterministic("rho", rho_empirical + noise_scale * rho_individual)
            sigma_ar = numpyro.deterministic("sigma_ar", sigma_ar_empirical + noise_scale * sigma_ar_individual)

            # Transform parameters
            delta_transformed = numpyro.deterministic("delta_t", expit(delta))
            M_transformed = numpyro.deterministic("Mt", jnp.exp(M))
            nu_transformed = numpyro.deterministic("nut", jnp.exp(nu))
            Q_transformed = numpyro.deterministic("Qt", jnp.exp(Q))
            B_transformed = numpyro.deterministic("Bt", jnp.exp(B))
            rho_transformed = numpyro.deterministic("rho_t", expit(rho))
            sigma_ar_transformed = numpyro.deterministic("sigma_ar_t", jnp.exp(sigma_ar))

            inc = -d_generalized_logistic(times, delta_transformed, B_transformed, M_transformed, Q_transformed, nu_transformed)
            eps = 1e-9
            logit_inc = jax.scipy.special.logit(jnp.clip(inc, eps, 1-eps))

            
            regression_outputs = logit_inc.reshape(ntimes, 1)

            # AR(1) Process
            # Sample innovations for new season
            z = numpyro.sample("ar_z", dist.Normal(0, 1).expand([ntimes]))
            
            # Build AR(1) path
            def ar_step(x_prev, z_t):
                x_t = rho_transformed * x_prev + sigma_ar_transformed * z_t
                return x_t, x_t
            _, ar_drift = lax.scan(ar_step, 0.0, z)
            
            # Combine regression + AR drift
            process_for_p = expit(regression_outputs.reshape(1, ntimes)  + ar_drift.reshape(1, ntimes))

            present = ~jnp.isnan(y) & ~jnp.isnan(N)

            cases_predicted = numpyro.deterministic("cases_predicted", process_for_p * N)

            with numpyro.handlers.mask(mask=present.reshape(1,ntimes)):
                numpyro.sample("y", dist.NegativeBinomial2((N * process_for_p).reshape(1, ntimes), 150), 
                             obs=y.reshape(1, ntimes))
           
            # Soft constraint on the final time point (end of forecast horizon)
            if constraint_mu is not None and constraint_sd is not None:
                # Expected count at the last time point in the sequence
                expected_last = N[0, -1] * process_for_p[0, -1]
                # Soft Gaussian constraint
                numpyro.sample("forecast_end_constraint", 
                             dist.Normal(constraint_mu, constraint_sd),
                             obs=expected_last)
           
            if forecast:
                inc_pred = numpyro.sample("inc_pred", 
                                         dist.NegativeBinomial2((N * process_for_p).reshape(1, ntimes), 150))

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=7000, num_samples=7000, num_chains=1)
        mcmc.run(self.key,
                 y=self.y,
                 X=None,
                 N=self.N,
                 nobs=self.nobs,
                 prior_mus=prior_mus,
                 prior_covs=prior_covs,
                 forecast=False,
                 N_pred=N_pred,
                 constraint_mu=constraint_mu,
                 constraint_sd=constraint_sd)
        mcmc.print_summary()
        
        print("MCMC completed!")
        samples = mcmc.get_samples()
        self.samples = samples
        
        if forecast == True:
            predictive = Predictive(model, posterior_samples=mcmc.get_samples())
            samples = predictive(jax.random.PRNGKey(42),
                               y=self.y,
                               X=None,
                               N=self.N,
                               nobs=self.nobs,
                               prior_mus=prior_mus,
                               prior_covs=prior_covs,
                               forecast=True,
                               N_pred=N_pred,
                               constraint_mu=constraint_mu,
                               constraint_sd=constraint_sd)
            self.predictive_samples = samples
        return samples








if __name__ == "__main__":
    pass
