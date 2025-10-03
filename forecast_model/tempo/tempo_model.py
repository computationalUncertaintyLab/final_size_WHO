#mcandrew

class tempo_model1(object):
    def __init__(self, y, N, nobs=0):
        self.y           = y.copy()
        self.N           = N.copy()
        self.nobs        = nobs

    def fit_past_seasons(self):
        import jax
        import jax.numpy         as jnp
        from   jax               import vmap
        from   jax.scipy.special import logit, expit
        import numpy             as np
        import numpyro
        import numpyro.distributions as dist
        from   numpyro.infer import MCMC, NUTS
        from   numpyro.infer import Predictive


        def model(y,N,nobs=0):
            nseasons, ntimes   = y.shape
            times              = np.arange(ntimes)
            eps                = 10**-5

            #--contagion with season-specific parameters
            def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

            A = 1.
            
            # Hierarchical model for incidence parameters with more informative priors
            # Global (population) level parameters - more constrained priors
            K = numpyro.sample("K", dist.Normal(0, 1))  # logit scale, ~0.1 prob
            M = numpyro.sample("M_mu", dist.Normal(0, 1))  # log scale, ~20 weeks
            B = numpyro.sample("B_mu"  , dist.Normal(0, 1))  # log scale, ~0.14
            B2 = numpyro.sample("B2_mu", dist.Normal(0, 1))  # log scale, ~0.14
            nu = numpyro.sample("nu_mu", dist.Normal(0, 1))  # log scale, ~1
            Q = numpyro.sample("Q_mu"  , dist.Normal(0, 1))  # log scale, ~1

            K  =   expit(K)
            M  = jnp.exp(M)
            nu = jnp.exp(nu)
            Q  = jnp.exp(Q)

            # Smooth transition instead of sharp cutoff
            transition_point = numpyro.sample("transition_point", dist.Uniform(0, ntimes))
            transition_width = numpyro.sample("transition_width", dist.Gamma(2, 1))  # Controls smoothness

            # Sigmoid transition: smooth from 0 to 1
            transition_weight = jax.nn.sigmoid((times - transition_point) / transition_width)
            B_over_time = numpyro.deterministic("B_over_time", jnp.exp(B + B2 * transition_weight))
                
            def inc_t(t,A,K,B,M,Q,nu):
                return -d_generalized_logistic(t, A, K, B, M, Q, nu)
            inc                  = numpyro.deterministic("inc", vmap(inc_t, in_axes=(0,None,None,0,None,None,None))(times, A, K, B_over_time, M, Q, nu)  )

            #s = numpyro.sample("s", dist.HalfNormal(1./10))
            #noise = numpyro.sample("noise", dist.Normal(0, s).expand([ntimes]))
            #inc_logit = logit(inc) + noise
            #inc = expit(inc_logit)

            cases_predicted = numpyro.deterministic("cases_predicted", inc * N)
            
            # Likelihood for observed cases using predicted N and season-specific incidence
            # Mask for valid observations (both y and N must be non-NaN)
            present = ~jnp.isnan(y) & ~jnp.isnan(N)
            # Use more constrained prior for observation noise
            # Replace NaN values with dummy values for the Binomial distribution
            # (they will be masked out anyway, but Binomial needs valid inputs)
            y_safe   = jnp.where(present, y, 0.0)
            N_safe   = jnp.where(present, N, 1.0)
            inc_safe = jnp.where(present, inc, 0.5)
            
            with numpyro.handlers.mask(mask=present):
                numpyro.sample("inc_ll", dist.Binomial(total_count=N_safe.astype(int), probs=inc_safe), obs=y_safe.astype(int))

        # Use MCMC with tuned parameters for hierarchical model
        # Use more conservative MCMC settings to reduce divergences
        nuts_kernel = NUTS(model, target_accept_prob=0.85)
        mcmc = MCMC(nuts_kernel , num_warmup=2500, num_samples=1500, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42),
                y    = self.y,
                N    = jnp.nan_to_num(self.N,nan=1),
                nobs = self.nobs)
        mcmc.print_summary()
        
        print("MCMC completed!")
        samples = mcmc.get_samples()
    
        return samples

    def fit_new_season(self, prior_tensor=None, forecast=False, N_pred=None):

        import jax
        import jax.numpy as jnp
        from   jax               import vmap
        from   jax.scipy.special import logit, expit
        import numpy             as np
        import numpyro
        import numpyro.distributions as dist
        from   numpyro.infer import MCMC, NUTS
        from   numpyro.infer import Predictive

        def model(y,N,nobs=0,prior_tensor=None,forecast=False,N_pred=None):
            nseasons, ntimes   = y.shape
            times              = np.arange(ntimes)
            eps                = 10**-5

            #--contagion with season-specific parameters
            def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

            A = 1.
            
            # Use joint empirical distribution prior if prior_tensor is provided
            nseasons_prior, nparams, nsamples = prior_tensor.shape
            
            # Sample weights for different seasons (how much to trust each historical season)
            season_weights = numpyro.sample("season_weights", dist.Dirichlet(jnp.ones(nseasons_prior)))
            
            # Weight the seasons: compute weighted average across seasons for each parameter and sample
            # prior_tensor: (nseasons_prior, nparams, nsamples)
            # season_weights: (nseasons_prior,)
            # Result: (nparams, nsamples)
            weighted_prior = jnp.einsum('s,spn->pn', season_weights, prior_tensor)
            
            # Now select from samples using Gumbel-Softmax
            gumbel_noise = numpyro.sample("gumbel_noise", dist.Gumbel(0, 1).expand([nsamples]))
            logits = jnp.zeros(nsamples)  # Uniform over samples
            soft_selection = jax.nn.softmax((logits + gumbel_noise) / 0.1)
            
            # Select from weighted samples for each parameter
            # weighted_prior: (nparams, nsamples), soft_selection: (nsamples,)
            # Result: (nparams,)
            empirical_sample = jnp.dot(weighted_prior, soft_selection)
            
            # Extract individual parameters while preserving correlations
            K_empirical  = empirical_sample[0]  # logit scale
            M_empirical  = empirical_sample[1]  # log scale
            B_empirical  = empirical_sample[2]  # log scale
            B2_empirical = empirical_sample[3]  # log scale
            nu_empirical = empirical_sample[4]  # log scale
            Q_empirical  = empirical_sample[5]  # log scale
            transition_point_empirical  = empirical_sample[6]  # log scale
            transition_width_empirical = empirical_sample[7]  # log scale
                
            # Sample individual priors (on same scales as empirical)
            K_individual  = numpyro.sample("K_individual"  , dist.Normal(0, 1))  # logit scale
            M_individual  = numpyro.sample("M_individual"  , dist.Normal(0, 1))  # log scale
            B_individual  = numpyro.sample("B_individual"  , dist.Normal(0, 1))  # log scale
            B2_individual = numpyro.sample("B2_individual" , dist.Normal(0, 1))  # log scale
            nu_individual = numpyro.sample("nu_individual" , dist.Normal(0, 1))  # log scale
            Q_individual  = numpyro.sample("Q_individual"  , dist.Normal(0, 1))  # log scale
            transition_point_individual  = numpyro.sample("transition_point_individual"  , dist.Normal(0, 1))  # log scale
            transition_width_individual  = numpyro.sample("transition_width_individual" , dist.Normal(0, 1))  # log scale

            # Weight for mixture (closer to 1 = more empirical, closer to 0 = more individual)
            #mixture_weight = numpyro.sample("mixture_weight", dist.Beta(10, 1))  # Bias toward empirical

            # Add small noise around empirical samples (keep them as the main component)
            noise_scale = 0.1  # Small perturbation around empirical samples
            K  = numpyro.deterministic("K" , K_empirical  + noise_scale * K_individual)
            M  = numpyro.deterministic("M" , M_empirical  + noise_scale * M_individual)
            B  = numpyro.deterministic("B" , B_empirical  + noise_scale * B_individual)
            B2 = numpyro.deterministic("B2", B2_empirical + noise_scale * B2_individual)
            nu = numpyro.deterministic("nu", nu_empirical + noise_scale * nu_individual)
            Q  = numpyro.deterministic("Q" , Q_empirical  + noise_scale * Q_individual)
            transition_point = numpyro.deterministic("transition_point" , transition_point_empirical  + noise_scale * transition_point_individual)
            transition_width = numpyro.deterministic("transition_width" , transition_width_empirical + noise_scale * transition_width_individual)

            # Apply transformations to get final parameter values
            K_transformed  = numpyro.deterministic("Kt" , expit(K))    # logit -> probability
            M_transformed  = numpyro.deterministic("Mt" , jnp.exp(M))  # log -> positive
            nu_transformed = numpyro.deterministic("nut", jnp.exp(nu)) # log -> positive  
            Q_transformed  = numpyro.deterministic("Qt" , jnp.exp(Q))  # log -> positive

            # Sigmoid transition: smooth from 0 to 1
            transition_weight = jax.nn.sigmoid((times - transition_point) / transition_width)
            B_over_time = numpyro.deterministic("B_over_time", jnp.exp(B + B2 * transition_weight))
 

            def inc_t(t,A,K,B,M,Q,nu):
                return -d_generalized_logistic(t, A, K, B, M, Q, nu)
            inc = numpyro.deterministic("inc", vmap(inc_t, in_axes=(0,None,None,0,None,None,None))(times, A, K_transformed, B_over_time, M_transformed, Q_transformed, nu_transformed))

            s         = numpyro.sample("s", dist.HalfNormal(1./10))
            noise     = numpyro.sample("noise", dist.Normal(0, s).expand([ntimes]))
            inc_logit = logit(inc) + noise
            inc       = expit(inc_logit)

            cases_predicted = numpyro.deterministic("cases_predicted", inc * N)
            
            # Likelihood for observed cases using predicted N and season-specific incidence
            # Mask for valid observations (both y and N must be non-NaN)
            present = ~jnp.isnan(y) & ~jnp.isnan(N)
            # Use more constrained prior for observation noise
            # Replace NaN values with dummy values for the Binomial distribution
            # (they will be masked out anyway, but Binomial needs valid inputs)
            y_safe = jnp.where(present, y, 0.0)
            N_safe = jnp.where(present, N, 1.0)
            
            with numpyro.handlers.mask(mask=present):
                numpyro.sample("inc_ll", dist.Binomial(total_count=N_safe.astype(int), probs=inc), obs=y_safe.astype(int))

            if forecast:
                noise     = numpyro.sample("noise_pred", dist.Normal(0, s).expand([ntimes]))
                inc_logit = logit(inc) + noise
                inc       = numpyro.deterministic("inc_pred", expit(inc_logit))

                #nsamples = len(N_pred)
                #idx = numpyro.sample("idx", dist.Categorical(jnp.ones(nsamples)))
                #numpyro.sample("forecast", dist.Binomial(total_count=N_pred[idx], probs=inc[idx]) )

        nuts_kernel = NUTS(model, target_accept_prob=0.85)
        mcmc = MCMC(nuts_kernel , num_warmup=2500, num_samples=1500, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42),
                y            = self.y,
                N            = self.N,
                nobs         = self.nobs,
                prior_tensor = prior_tensor,
                forecast     = forecast,
                N_pred       = N_pred)
        mcmc.print_summary()
        
        print("MCMC completed!")
        samples = mcmc.get_samples()
        self.samples = samples
        
        if forecast==True:
            predictive = Predictive(model, posterior_samples=mcmc.get_samples())
            samples = predictive(jax.random.PRNGKey(42),
                    y            = self.y,
                    N            = self.N,
                    nobs         = self.nobs,
                    prior_tensor = prior_tensor,
                    forecast     = forecast,
                    N_pred       = N_pred)
            self.predictive_samples = samples
        return samples

    def generate_forecast(self, N_samples):
        import jax
        from numpyro.distributions import dist
        cases_samples = dist.Binomial(total_count  = N_samples.astype(int)
                                        , probs    = self.samples["inc"]).sample(jax.random.PRNGKey(321))
        return {"inc": self.samples["inc"], "cases_predicted": cases_samples}

        
class tempo_model2(object):
    def __init__(self, y, N, nobs=0):
        self.y           = y.copy()
        self.N           = N.copy()
        self.nobs        = nobs

    def fit_past_seasons(self):
        import jax
        import jax.numpy         as jnp
        from   jax               import vmap
        from   jax.scipy.special import logit, expit
        import numpy             as np
        import numpyro
        import numpyro.distributions as dist
        from   numpyro.infer import MCMC, NUTS
        from   numpyro.infer import Predictive


        def model(y,N,nobs=0):
            nseasons, ntimes   = y.shape
            times              = np.arange(ntimes)
            eps                = 10**-5

            #--contagion with season-specific parameters
            def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

            A = 1.
            
            # Hierarchical model for incidence parameters with more informative priors
            # Global (population) level parameters - more constrained priors
            K                = numpyro.sample("K"     , dist.Normal(0, 1))  # logit scale, ~0.1 prob
            M                = numpyro.sample("M_mu"  , dist.Normal(0, 1))  # log scale, ~20 weeks
            B                = numpyro.sample("B_mu"  , dist.Normal(0, 1))  # log scale, ~0.14
            B2               = numpyro.sample("B2_mu", dist.Normal(0, 1))  # log scale, ~0.14
            nu               = numpyro.sample("nu_mu", dist.Normal(0, 1))  # log scale, ~1
            Q                = numpyro.sample("Q_mu"  , dist.Normal(0, 1))  # log scale, ~1
            transition_width = numpyro.sample("transition_width", dist.Normal(0,1))

            K                = expit(K)
            M                = jnp.exp(M)
            nu               = jnp.exp(nu)
            Q                = jnp.exp(Q)
            B                = jnp.exp(B)
            B2               = jnp.exp(B2)
            transition_width = jnp.exp(transition_width)

            # Sigmoid transition: smooth from 0 to 1
            transition_weight = jax.nn.sigmoid((times - M) / transition_width)
            B_over_time       = numpyro.deterministic("B_over_time", B + (B+B2) * transition_weight)
                
            def inc_t(t,A,K,B,M,Q,nu):
                return -d_generalized_logistic(t, A, K, B, M, Q, nu)
            inc                  =  vmap(inc_t, in_axes=(0,None,None,0,None,None,None))(times, A, K, B_over_time, M, Q, nu)
            inc = numpyro.deterministic("inc_base", inc)

            present = ~jnp.isnan(y) & ~jnp.isnan(N)

            eps = 1e-9
            eta_base = jax.scipy.special.logit(jnp.clip(inc, eps, 1-eps))
            sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1./10))
            eta       = eta_base + numpyro.sample("eps_t",
                                            dist.Normal(0., sigma_obs).expand([ntimes]).to_event(1))
            p_obs = expit(eta)
            inc   = numpyro.deterministic("inc", p_obs)

            cases_predicted = numpyro.deterministic("cases_predicted", p_obs * N)
            with numpyro.handlers.mask(mask=present):
                numpyro.sample("y", dist.Binomial(total_count=N, probs=p_obs), obs=y)

            
        # Use MCMC with tuned parameters for hierarchical model
        # Use more conservative MCMC settings to reduce divergences
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel , num_warmup=2500, num_samples=1500, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42),
                y    = self.y,
                N    = jnp.nan_to_num(self.N,nan=1),
                nobs = self.nobs)
        mcmc.print_summary()
        
        print("MCMC completed!")
        samples = mcmc.get_samples()
    
        return samples

    def fit_new_season(self, prior_mus=None, prior_covs=None, forecast=False, N_pred=None):

        import jax
        import jax.numpy as jnp
        from   jax               import vmap
        from   jax.scipy.special import logit, expit
        import numpy             as np
        import numpyro
        import numpyro.distributions as dist
        from   numpyro.infer import MCMC, NUTS
        from   numpyro.infer import Predictive
        from numpyro.infer.reparam import TransformReparam
        from numpyro import handlers

        def model(y,N,nobs=0,prior_mus=None, prior_covs=None,forecast=False,N_pred=None):
            nseasons, ntimes   = y.shape
            times              = np.arange(ntimes)
            eps                = 10**-5

            #--contagion with season-specific parameters
            def d_generalized_logistic(t, A, K, B, M, Q=1.0, nu=1.0):
                import jax
                import jax.numpy as jnp

                exp_term = jnp.exp(-B * (t - M))
                denom = (1 + Q * exp_term)
                return ((K - A) * B * Q * exp_term) / (nu * denom**(1/nu + 1))

            A = 1.

            ntimes = len(times)
            # Use joint empirical distribution prior if prior_tensor is provided
            nseasons_prior, nparams = prior_mus.shape
            
            # Sample weights for different seasons (how much to trust each historical season)
            season_weights = numpyro.sample("season_weights", dist.Dirichlet(5*jnp.ones(nseasons_prior)))

            temp = 0.1
            log_weights = jnp.log(season_weights)
            g           = numpyro.sample("gumbel_season", dist.Gumbel(0.0, 1.0).expand([nseasons_prior]))

            z = jax.nn.softmax((log_weights + g)/temp)
            #zz = z.reshape(nseasons_prior,1,1)
            
            m = (z[:,None]*prior_mus).sum(0)

            second_moment = (z[:, None, None] *
                             (prior_covs + jnp.einsum('si,sj->sij', prior_mus, prior_mus))
                             ).sum(0)                                    # (d,d)
            Sigma = second_moment - jnp.outer(m, m)
            Sigma = 0.5*(Sigma + Sigma.T)

            jitter = 10**-3
            L = jnp.linalg.cholesky(Sigma + jitter * jnp.eye(m.shape[0]))
            #param_vec = numpyro.sample("param_vec", dist.MultivariateNormal(m, scale_tril=C))

            eps       = numpyro.sample("param_vec_white", dist.Normal(0., 1.).expand([m.shape[0]]).to_event(1))
            param_vec = numpyro.deterministic("param_vec", m + L @ eps)

            #--sample param vector
            #base_loc = prior_mus.mean(0)
            #base_cov = prior_covs.mean(0) #jnp.cov(prior_mus.T) + 1*jnp.eye(nparams)
            #param_vec = numpyro.sample("param_vec", dist.MultivariateNormal( base_loc, base_cov ) )

            #prior_scale_tril = jnp.linalg.cholesky(prior_covs + 1e-6*jnp.eye(nparams))
            #mog              = dist.MultivariateNormal( prior_mus, scale_tril = prior_scale_tril )
           
            #numpyro.factor( "prior_mix"  , jax.scipy.special.logsumexp( jnp.log(season_weights) +  mog.log_prob( param_vec ))  )
            #numpyro.factor( "base_sample", -dist.MultivariateNormal(base_loc, base_cov ).log_prob(param_vec)  )
           
            # Extract individual parameters while preserving correlations
            K_empirical                 = param_vec[0]
            M_empirical                 = param_vec[1]
            B_empirical                 = param_vec[2]
            B2_empirical                = param_vec[3]
            nu_empirical                = param_vec[4]
            Q_empirical                 = param_vec[5]
            transition_width_empirical  = param_vec[6]
                
            # Sample individual priors (on same scales as empirical)
            K_individual  = numpyro.sample("K_individual"  , dist.Normal(0, 1))  # logit scale
            M_individual  = numpyro.sample("M_individual"  , dist.Normal(0, 1))  # log scale
            B_individual  = numpyro.sample("B_individual"  , dist.Normal(0, 1))  # log scale
            B2_individual = numpyro.sample("B2_individual" , dist.Normal(0, 1))  # log scale
            nu_individual = numpyro.sample("nu_individual" , dist.Normal(0, 1))  # log scale
            Q_individual  = numpyro.sample("Q_individual"  , dist.Normal(0, 1))  # log scale
            transition_width_individual  = numpyro.sample("transition_width_individual" , dist.Normal(0, 1))  # log scale

            # Weight for mixture (closer to 1 = more empirical, closer to 0 = more individual)
            #mixture_weight = numpyro.sample("mixture_weight", dist.Beta(10, 1))  # Bias toward empirical

            # Add small noise around empirical samples (keep them as the main component)
            #noise_scale = 0.1  # Small perturbation around empirical samples

            noise_scale = 0.01#numpyro.sample("noise_scale", dist.Beta(1,10))
            K  = numpyro.deterministic("K" , K_empirical  + noise_scale * K_individual)
            M  = numpyro.deterministic("M" , M_empirical  + noise_scale * M_individual)
            B  = numpyro.deterministic("B" , B_empirical  + noise_scale * B_individual)
            B2 = numpyro.deterministic("B2", B2_empirical + noise_scale * B2_individual)
            nu = numpyro.deterministic("nu", nu_empirical + noise_scale * nu_individual)
            Q  = numpyro.deterministic("Q" , Q_empirical  + noise_scale * Q_individual)
            transition_width = numpyro.deterministic("transition_width" , transition_width_empirical + noise_scale * transition_width_individual)

            # Apply transformations to get final parameter values
            K_transformed    = numpyro.deterministic("Kt" , expit(K))    # logit -> probability
            M_transformed    = numpyro.deterministic("Mt" , jnp.exp(M))  # log -> positive
            nu_transformed   = numpyro.deterministic("nut", jnp.exp(nu)) # log -> positive  
            Q_transformed    = numpyro.deterministic("Qt" , jnp.exp(Q))  # log -> positive
            B_transformed    =  numpyro.deterministic("Bt" , jnp.exp(B))  # log -> positive
            B2_transformed    =  numpyro.deterministic("B2t" , jnp.exp(B2))  # log -> positive
            transition_width_transformed = numpyro.deterministic("transition_width_t" , jnp.exp(transition_width))  # log -> positive
            
            # Sigmoid transition: smooth from 0 to 1
            transition_weight = jax.nn.sigmoid((times - M_transformed) / transition_width_transformed)
            B_over_time       = numpyro.deterministic("B_over_time", B_transformed + (B_transformed+B2_transformed) * transition_weight)

            def inc_t(t,A,K,B,M,Q,nu):
                return -d_generalized_logistic(t, A, K, B, M, Q, nu)
            inc =  vmap(inc_t, in_axes=(0,None,None,0,None,None,None))(times, A, K_transformed, B_over_time, M_transformed, Q_transformed, nu_transformed)

            # Likelihood for observed cases using predicted N and season-specific incidence
            # Mask for valid observations (both y and N must be non-NaN)
            present = ~jnp.isnan(y) & ~jnp.isnan(N)
            # Use more constrained prior for observation noise
            # Replace NaN values with dummy values for the Binomial distribution
            # (they will be masked out anyway, but Binomial needs valid inputs)
            y_safe = jnp.where(present, y, 0.0)
            N_safe = jnp.where(present, N, 1.0)

            eps = 1e-9
            # tau_shock = numpyro.sample("tau_shock", dist.HalfNormal(1./10))
            # lam_t     = numpyro.sample("lam_shock", dist.HalfCauchy(1./2).expand([ntimes]))
            # z_t       = numpyro.sample("z_shock", dist.Normal(0., 1.).expand([ntimes]))
            # log_s_t   = tau_shock * lam_t * z_t                   # horseshoe shrinkage
            # s_t       = jnp.exp(log_s_t)
            # inc_spiky = numpyro.deterministic(
            #     "inc_spiky", jnp.clip(inc * s_t, eps, 1.0 - eps)
            # )

            eta_base = logit(jnp.clip(inc, eps, 1-eps))
            sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.6))
            eta       = eta_base + numpyro.sample("eps_t",
                                            dist.Normal(0., sigma_obs).expand([ntimes]).to_event(1))
            p_obs = expit(eta)
            inc   = numpyro.deterministic("inc", p_obs)
            
            cases_predicted = numpyro.deterministic("cases_predicted", p_obs * N)
            
            with numpyro.handlers.mask(mask=present):
                numpyro.sample("inc_ll", dist.Binomial(total_count=N_safe.astype(int), probs=p_obs), obs=y_safe.astype(int))

            if forecast:
                #noise     = numpyro.sample("noise_pred", dist.Normal(0, s).expand([ntimes]))
                #inc_logit = logit(inc) + noise
                #inc       = numpyro.deterministic("inc_pred", expit(inc_logit))

                inc        = numpyro.deterministic("inc_pred"  , p_obs)
                #inc_smooth = numpyro.deterministic("inc_smooth", inc_spiky)
                
                #nsamples = len(N_pred)
                #idx = numpyro.sample("idx", dist.Categorical(jnp.ones(nsamples)))
                #numpyro.sample("forecast", dist.Binomial(total_count=N_pred[idx], probs=inc[idx]) )

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel , num_warmup=5000, num_samples=1500, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42),
                y              = self.y
                 ,N            = self.N
                 ,nobs         = self.nobs
                 ,prior_mus = prior_mus
                 ,prior_covs = prior_covs
                 ,forecast     = forecast
                 ,N_pred       = N_pred)
        mcmc.print_summary()
        
        print("MCMC completed!")
        samples = mcmc.get_samples()
        self.samples = samples
        
        if forecast==True:
            predictive = Predictive(model, posterior_samples=mcmc.get_samples())
            samples = predictive(jax.random.PRNGKey(42),
                                 y            = self.y
                                 ,N            = self.N
                                 ,nobs         = self.nobs
                                 ,prior_mus = prior_mus
                                 ,prior_covs = prior_covs
                                 ,forecast     = forecast
                                 ,N_pred       = N_pred)
            self.predictive_samples = samples
        return samples

    def generate_forecast(self, N_samples):
        import jax
        from numpyro.distributions import dist
        cases_samples = dist.Binomial(total_count  = N_samples.astype(int)
                                        , probs    = self.samples["inc"]).sample(jax.random.PRNGKey(321))
        return {"inc": self.samples["inc"], "cases_predicted": cases_samples}




if __name__ == "__main__":
    pass
