#' R6 classes for my MCMC samplers
#'
#' First basic MCMC structure to hold parameters and init
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
MCMC = R6Class(
  "MCMC",
  public = list(
    N.iter = NULL, # number of iterations
    N.params = NULL, # number of parameters
    data = NULL,
    iteration = NULL, # current iteration
    samples = NULL, # object for storing samples
    initialize = function(N.iter = 1000,
                          N.params = NULL,
                          data = NULL,
                          init = NULL){
      self$N.iter = N.iter
      self$N.params = N.params
      self$data = data
      self$iteration = 1
      self$samples = array(NA, dim=c(N.iter,N.params))
      # Init
      if (!is.null(N.params)){
        if (!is.null(init)){
          self$samples[1,] = init
        } else {self$samples[1,] = runif(N.params,min=-2,max=2)}
      }
    }
  )
)
#' HMC sampler with automatic adaptation of chain
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
HMC = R6Class(
  "HMCSampler",
  inherit = MCMC,
  public = list(
    control = list( # List of parameters that control the sampler
      init_buffer = 75, # width of first fast adaptation window
      term_buffer = 50, # width of final fast adaptation window 
      init_window = 25, # initial window size for slow adaptation
      window = NULL, # window size
      traj_length = 1.0, # target trajectory length
      max_L = 2^6, # maximum number of leapfrog steps to perform
      t0 = 10, # Adaptation iteration offset
      delta = 0.8, # Target acceptance probability
      kappa = 0.75, # Adapatation relaxation exponent
      gamma = 0.05, # adaptation regularization scale
      warmup = NULL,
      L = NULL,
      epsilon = NULL,
      alpha = NULL,
      mass_matrix = NULL,
      # Setting flags for adaptation
      adapt = T, # Should adaptation be on?
      adapt_epsilon = NULL, # Should we adapt epsilon
      adapt_L = NULL, # Should we adapt the number of leapfrog steps
      adapt_mass = NULL, # Should we adapt the mass_matrix?
      # Some more detailed stuff below here for the adaptation of epsilon
      x = NULL,
      mu = NULL, # Ideally I'd want to use few leapfrog iterations, say 10
      xbar = NULL,
      H = 0,
      t = 1, # warmup counter
      # And denote the various phases
      phases = NULL,
      window_counter = 0, # counting windows
      reject_counter = 0 # count rejections
    ),
    initialize = function(#N.iter = 1000,
      #N.params = NULL,
      #data = NULL,
      control = NULL,
      #init = NULL,
      L0 = self$control$max_L, epsilon0 = 0.0001, 
      alpha0 = 1, ...){
      if (!is.null(control)){
        self$control = modifyList(control, self$control)
      }
      # Some derived parameters that need to be set manually in init
      self$control$window = 2^(seq(0,4))*self$control$init_window
      self$control$warmup = self$control$init_buffer + self$control$term_buffer + sum(self$control$window)
      
      # Now init super
      super$initialize(...)
      # And check that the number of iterations is larger than warmup
      if (self$N.iter < self$control$warmup){
        print(paste0("N.iter must be larger than ", self$control$warmup))
        break
      }
      
      if (self$control$adapt){
        self$control$adapt_epsilon = T
        self$control$adapt_L = T
        self$control$adapt_mass = F # don't adapt mass matrix from initial samples
      }
      if (!is.null(self$data$t)){self$data$tlist = lapply(1:nrow(self$data$t), function(i) matrix(self$data$t[i, ], ncol = 1))}
      # Containers
      self$control$L = rep(0.0, self$N.iter)
      self$control$epsilon = rep(0.0, self$N.iter)
      self$control$alpha = rep(0.0, self$N.iter)
      # Initialise
      self$control$L[1] = L0
      self$control$epsilon[1] = epsilon0
      self$control$alpha[1] = alpha0
      if (!is.null(self$N.params)){
        self$control$mass_matrix = rep(1,self$N.params)
      }
      self$control$x = log(epsilon0)
      self$control$mu = log(10*epsilon0)
      self$control$xbar = self$control$x
      self$control$phases =  c(self$control$init_buffer, self$control$init_buffer + cumsum(self$control$window), self$control$init_buffer + sum(self$control$window) + self$control$term_buffer)
    },
    sample = function(){
      print("Not implemented, must be implemented in inhereted subclass")
      break
    }
  ),
  active = list(
    current_epsilon = function() self$control$epsilon[self$iteration],
    current_L = function() self$control$L[self$iteration]
  ),
  private = list(
    adapt = function() {
      # Main function for iterating the sampler and updating the sampler's hyperparameters
      # First thing is we increase the current iteration
      self$iteration = self$iteration + 1
      # Adapt epsilon
      if (self$control$adapt & self$control$adapt_epsilon){
        # We adapt epsilon
        eta = self$control$t^(-self$control$kappa)
        self$control$H = self$control$H + (self$control$delta-self$control$alpha[self$iteration])
        self$control$x = self$control$mu - (sqrt(self$control$t)/self$control$gamma)*(self$control$H/(self$control$t+self$control$t0))
        self$control$xbar = eta*self$control$x + (1-eta)*self$control$xbar
        self$control$t = self$control$t + 1
        # And store epsilon
        self$control$epsilon[self$iteration] = exp(self$control$x)
      } else{
        # We keep epsilon fixed
        self$control$epsilon[self$iteration] = exp(self$control$xbar)
      }
      if (self$control$adapt & self$control$adapt_L){
        # Adapting L according to desired trajectory length
        self$control$L[self$iteration] = min(max(floor(self$control$traj_length/exp(self$control$xbar)),1),self$control$max_L)
      } else{
        self$control$L[self$iteration] = self$control$L[self$iteration-1]
      }
      print(paste0("Sampling! ", "current epsilon = ", round(self$current_epsilon,4),
                   " current L: ", self$current_L,
                   " Acceptance probability: ", round(self$control$alpha[self$iteration],2),
                   " iteration=",self$iteration-1))
      
      # Check if we are changing phases
      if (self$iteration %in% self$control$phases){
        if (self$iteration == self$control$phases[2]){ 
          # Starting windowed adaptation of covariance
          self$control$adapt_mass = T
        }
        if (self$iteration == self$control$phases[6]){
          # Stopping windows adaptation of covariance
          self$control$adapt_mass = F
        }
        if (self$iteration == self$control$phases[7]){
          # Stopping adaptation of epsilon and L
          self$control$adapt_epsilon = F
          self$control$adapt_L = F
        }
        
        print("Next phase!")
        # Reset some stuff for estimation of epsilon
        # (as long as we are still adapting this)
        if (self$control$adapt & self$control$adapt_epsilon){
          print("Epsilon stuff reset")
          self$control$x = log(self$control$epsilon[1]) # epsilon0
          self$control$mu = log(10*self$control$epsilon[1]) # epsilon0
          self$control$xbar = self$control$x
          self$control$H = 0
          self$control$t = 1
        }
        
        # Estimate mass matrix using buffer
        if (self$control$adapt & self$control$adapt_mass){
          # print(paste0("old mass_matrix: ", self$control$mass_matrix))
          self$control$window_counter = self$control$window_counter + 1
          n_samples = self$control$window[self$control$window_counter]
          # Estimate covariance of window data
          current_mad = apply(self$samples[(self$iteration-self$control$window[self$control$window_counter]):self$iteration,],2,mad)
          current_var = (1.4826*current_mad)^2 # small correction here
          # Regularise
          regularised_variance = n_samples/(n_samples+5)*current_var + (5/(n_samples+5))*1
          # Construct and standardise
          self$control$mass_matrix = 1/regularised_variance
        }
      }
    }
  )
) 
#' HMC sampler with for F hypers
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
HMC_samplerF = R6Class("HMCSampler",
                       inherit = HMC,
                       public = list(
                         initialize = function(slab_scale = 4, slab_df = 4.0, 
                                               nu_local = 3, nu_global = 3,...){
                           super$initialize(...)
                           self$data$slab_scale = slab_scale
                           self$data$slab_df = slab_df
                           self$data$nu_local = nu_local
                           self$data$nu_global = nu_global
                         },
                         sample = function(){
                           eigKt = private$.eigKt()
                           Qt = eigKt$vectors
                           Dt = eigKt$values
                           step = tryCatch(
                             sample_f_hypers(self$data$X, Qt, Dt, 
                                             self$data$Y,self$data$gamma,
                                             self$data$tau0_prime,
                                             self$data$nugget, self$data$ell, 
                                             self$samples[self$iteration,], 
                                             self$control$mass_matrix, 
                                             self$current_epsilon,
                                             self$current_L,
                                             self$data$slab_scale,
                                             self$data$slab_df,
                                             self$data$nu_local,
                                             self$data$nu_global),
                             error = function(e) {
                               warning(paste0("Divergence! ", e))
                               # print(e)
                               NULL
                             }
                           )
                           reject = F
                           if (is.null(step)){ # If this is null, then auto-reject
                             reject = T
                           } else{
                             if (any(is.infinite(exp(step$theta)))){ # If any of these are too large, reject
                               reject = T
                             }
                             if (any(abs(step$theta)>100)){
                               reject = T
                             }
                           }
                           if (reject){ # Reject
                             self$samples[self$iteration+1,] = self$samples[self$iteration,]
                             self$control$alpha[self$iteration+1] = 0
                             self$control$reject_counter = self$control$reject_counter + 1
                           } else{ # Accept
                             self$samples[self$iteration+1,] = step$theta
                             self$control$alpha[self$iteration + 1] = step$accept_prob
                             self$control$reject_counter = 0
                           }
                           if (self$control$reject_counter > 5){
                             # Go back to a point where things worked
                             self$samples[self$iteration+1,] = self$samples[self$iteration-6,]
                           }
                           # Adapt stuff
                           super$adapt()
                         }
                       ),
                       active = list(
                         p = function() {
                           as.integer(self$N.params - 4)/2 # Change this to n-4 after
                         },
                         tau = function() {
                           Dt = private$.eigKt()$values
                           Dt = pmax(Dt,0)
                           log_tau0 = log(self$data$tau0_prime)-log(sum(sqrt(Dt)))
                           tau = exp(self$samples[,1] + 0.5*self$samples[,2] + log_tau0)
                           return(tau)
                         },
                         lambda = function() {
                           num = sweep(self$lambda_tilde,1,self$u,FUN="*")
                           denom_A = sweep(self$lambda_tilde,1,self$tau,FUN="*")^2
                           denom = sqrt(sweep(denom_A,1,self$u^2,FUN="+"))
                           num / denom
                         },
                         u = function(){
                           u = exp(log(self$data$slab_scale) + 0.5*self$samples[,2*self$p+4])
                           return(u)
                         },
                         c = function(){
                           exp(self$samples[,2*self$p+3])
                         },
                         lambda_tilde = function(){
                           exp(self$samples[,3:(self$p+2)]+0.5*self$samples[,(self$p+3):(2*self$p+2)])
                         }
                       ),
                       private = list(
                         .Kt = function(){
                           t1 = as.matrix(self$data$t)
                           t2 = as.matrix(self$data$t)
                           n1 = nrow(t1)
                           n2 = nrow(t2)
                           K = matrix(NA,nrow=n1,ncol=n2)
                           for (i in 1:n1){
                             for (j in 1:n2){
                               K[i,j] = exp(-0.5*abs(t1[i,]-t2[j,])^2/self$data$ell^2)
                             }
                           }
                           return(K+1e-6*diag(n1))
                         },
                         .eigKt = function(){
                           return(eigen(private$.Kt(),symmetric = T))
                         }
                       )
)
#' HMC sampler with for Z hypers
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
HMC_samplerZ = R6Class("HMCSampler",
                       inherit = HMC,
                       public = list(
                         sample = function(){
                           eigKt = private$.eigKt()
                           Qt = eigKt$vectors
                           Dt = eigKt$values
                           step = tryCatch(
                             sample_z_hypers(Qt, Dt,
                                             self$data$Y,
                                             self$data$temperature[self$iteration],
                                             self$data$nugget,
                                             self$data$eta[self$iteration],
                                             self$data$beta_a,
                                             self$data$beta_b,
                                             self$data$dir_a,
                                             self$samples[self$iteration,], 
                                             self$control$mass_matrix, 
                                             self$current_epsilon,
                                             self$current_L),
                             error = function(e) {
                               warning(paste0("Divergence! ", e))
                               # print(e)
                               NULL
                             }
                           )
                           reject = F
                           if (is.null(step)){
                             reject = T
                           } else{
                             if (any(is.infinite(exp(step$theta)))){
                               reject = T
                             }
                             if (any(abs(step$theta)>100)){
                               reject = T
                             }
                           }
                           if (reject){ # Reject
                             self$samples[self$iteration+1,] = self$samples[self$iteration,]
                             self$control$alpha[self$iteration+1] = 0
                             self$control$reject_counter = self$control$reject_counter + 1
                           } else{ # Accept
                             self$samples[self$iteration+1,] = step$theta
                             self$control$alpha[self$iteration + 1] = step$accept_prob
                             self$control$reject_counter = 0
                           }
                           if (self$control$reject_counter > 5){
                             # Go back to a point where things worked
                             self$samples[self$iteration+1,] = self$samples[self$iteration-6,]
                           }
                           
                           # Adapt stuff
                           super$adapt()
                         }
                       ),
                       active = list(
                         n = function(){
                           self$N.params - 1
                         },
                         log_phi_tilde = function(){
                           self$samples[,1:self$n]
                         },
                         logit_u = function(){
                           self$samples[,(self$n + 1)]
                         },
                         phi_tilde = function(){
                           exp(self$log_phi_tilde)
                         },
                         phi = function(){
                           sweep(self$phi_tilde,1,rowSums(self$phi_tilde),FUN="/")
                         },
                         u = function(){
                           1/(1+exp(-self$logit_u))
                         },
                         omega = function(){
                           exp(self$logit_u)
                         },
                         omega_scaled = function(){
                           self$omega * self$data$eta
                         },
                         gamma = function(){
                           sqrt(sweep(self$phi,1,self$omega_scaled,FUN="*"))
                         }
                       ),
                       private = list(
                         .Kt = function(){
                           t1 = as.matrix(self$data$t)
                           t2 = as.matrix(self$data$t)
                           n1 = nrow(t1)
                           n2 = nrow(t2)
                           K = matrix(NA,nrow=n1,ncol=n2)
                           for (i in 1:n1){
                             for (j in 1:n2){
                               K[i,j] = exp(-0.5*abs(t1[i,]-t2[j,])^2/self$data$ell^2)
                             }
                           }
                           return(K+1e-6*diag(n1))
                         },
                         .eigKt = function(){
                           return(eigen(private$.Kt(),symmetric = T))
                         }
                       )
)
#' HMC sampler with for F hypers with SKIM kernel
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
HMC_samplerSKIM = R6Class("HMCSampler",
                          inherit = HMC_samplerF,
                          public = list(
                            sample = function(){
                              eigKt = private$.eigKt()
                              Qt = eigKt$vectors
                              Dt = eigKt$values
                              step = tryCatch(
                                sample_f_hypers_SKIM(self$data$X, Qt, Dt, 
                                                     self$data$Y,
                                                     self$data$tau0_prime,
                                                     self$data$nugget, self$data$ell, 
                                                     self$samples[self$iteration,], 
                                                     self$control$mass_matrix, 
                                                     self$current_epsilon,
                                                     self$current_L,
                                                     self$data$slab_scale,
                                                     self$data$slab_df,
                                                     self$data$nu_local,
                                                     self$data$nu_global),
                                error = function(e) {
                                  warning(paste0("Divergence! ", e))
                                  # print(e)
                                  NULL
                                }
                              )
                              reject = F
                              if (is.null(step)){ # If this is null, then auto-reject
                                reject = T
                              } else{
                                if (any(is.infinite(exp(step$theta)))){ # If any of these are too large, reject
                                  reject = T
                                }
                                if (any(abs(step$theta)>100)){
                                  reject = T
                                }
                              }
                              if (reject){ # Reject
                                self$samples[self$iteration+1,] = self$samples[self$iteration,]
                                self$control$alpha[self$iteration+1] = 0
                                self$control$reject_counter = self$control$reject_counter + 1
                              } else{ # Accept
                                self$samples[self$iteration+1,] = step$theta
                                self$control$alpha[self$iteration + 1] = step$accept_prob
                                self$control$reject_counter = 0
                              }
                              if (self$control$reject_counter > 5){
                                # Go back to a point where things worked
                                self$samples[self$iteration+1,] = self$samples[self$iteration-6,]
                              }
                              # Adapt stuff
                              super$adapt()
                            }
                          ),
                          active = list(
                            p = function() {
                              as.integer(self$N.params - 5)/2 # Change this to n-4 after
                            },
                            v = function(){ # And comment out this
                              v = exp(log(self$data$slab_scale) + 0.5*self$samples[,2*self$p+5])
                              return(v)
                            },
                            tau2 = function(){
                              tau2 = self$v * (self$tau/self$u)^2
                              return(tau2)
                            }
                          )
)
#' Matheron sampler using the Kronecker structure to sample F
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
KroneckerMatheronSamplerF = R6Class("KroneckerMatheronSampler",
                                    # Samples from the posterior of a GP with Kronecker-structured covariance
                                    # using Matheron's rule.
                                    public = list(
                                      N.iter = NULL,
                                      thinning = NULL,
                                      N.params = NULL,
                                      # samples = NULL,
                                      unthinned_samples = NULL,
                                      iteration = 1,
                                      data = NULL,
                                      control = list(
                                        window_counter = 0, # counting windows
                                        reject_counter = 0 # count rejections
                                      ),
                                      initialize = function(N.iter = 1000, N.params = NULL,
                                                            data = NULL, thinning = 10){
                                        self$N.iter = N.iter
                                        self$N.params = N.params
                                        self$unthinned_samples = array(NA,dim=c(N.iter*thinning,N.params))
                                        self$data = data
                                        self$thinning = thinning
                                        self$sample() # Init this by sampling
                                      },
                                      sample = function(){
                                        # I'll create the kernels here as well and do everything
                                        # First thing we do is check if hypers are changed
                                        if (self$hypers_changed){
                                          # print("shouldn't be here often")
                                          eigKt = self$eigKt
                                          eigKx = self$eigKx
                                          eigKg = self$eigKg
                                          eigKz = self$eigKz
                                          # And I'll store them in cache
                                          private$.cache$eigKt = eigKt
                                          private$.cache$eigKx = eigKx
                                          private$.cache$eigKg = eigKg
                                          private$.cache$eigKz = eigKz
                                          # And I'll store the current hypers in cache
                                          private$.cache$tau = self$data$tau
                                          private$.cache$lambda = self$data$lambda
                                          private$.cache$c = self$data$c
                                          private$.cache$ell = self$data$ell
                                          private$.cache$gamma = self$data$gamma
                                        } else{
                                          eigKt = private$.cache$eigKt
                                          eigKx = private$.cache$eigKx
                                          eigKg = private$.cache$eigKg
                                          eigKz = private$.cache$eigKz
                                        }
                                        
                                        
                                        # Try-catch here
                                        step = tryCatch(
                                          {
                                            Y = self$data$Y
                                            sigma = self$data$sigma
                                            
                                            Qt = eigKt$vectors
                                            Qx = eigKx$vectors
                                            Qg = eigKg$vectors
                                            Qz = eigKz$vectors
                                            Dt = pmax(eigKt$values,0)
                                            Dx = pmax(eigKx$values,0)
                                            Dg = pmax(eigKg$values,0)
                                            Dz = pmax(eigKz$values,0)
                                            Kx = Qx%*%diag(Dx)%*%t(Qx)
                                            Kt = Qt%*%diag(Dt)%*%t(Qt)
                                            Kg = Qg%*%diag(Dg)%*%t(Qg)
                                            Kz = Qz%*%diag(Dz)%*%t(Qz)
                                            n = nrow(Qx)
                                            m = nrow(Qt)
                                            # Matheron's rule
                                            eta1 = matrix(rnorm(n*m),ncol=m)
                                            eta1 = sweep(eta1,1,sqrt(Dx),'*')
                                            eta1 = sweep(eta1,2,sqrt(Dt),'*')
                                            f1.prior = Qx%*%eta1%*%t(Qt)
                                            eta2 = matrix(rnorm(n*m),ncol=m)
                                            eta2 = sweep(eta2,1,sqrt(Dz),'*')
                                            eta2 = sweep(eta2,2,sqrt(Dt),'*')
                                            z1.prior = Qz%*%eta2%*%t(Qt)
                                            error = sigma*matrix(rnorm(n*m),ncol=m)
                                            eta = Y-f1.prior-z1.prior-error
                                            # Now compute correction, note subtraction of c^2 meaning no intercept
                                            correction = ((Kx%*%Qg)%*%((1/(Dg%*% t(Dt)+sigma^2))*t(Qg)%*%eta%*%Qt)%*%t(Kt%*%Qt))
                                            f1.prior + correction
                                          },
                                          error = function(e) {
                                            warning(paste0("Error in Matheron step! ", e))
                                            # print(e)
                                            NULL
                                          }
                                        )
                                        reject = F
                                        if (is.null(step)){
                                          reject = T
                                        }
                                        if (reject){ # Reject
                                          self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-1,,]
                                          self$control$reject_counter = self$control$reject_counter + 1
                                        } else{ # Accept
                                          self$unthinned_samples[self$iteration,,] = step
                                          self$control$reject_counter = 0
                                        }
                                        if (self$control$reject_counter > 5){
                                          # Go back to a point where things worked
                                          self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-6,,]
                                        }
                                        # And increase iteration
                                        self$iteration = self$iteration + 1
                                      },
                                      skip = function(){
                                        self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-1,,]
                                        self$iteration = self$iteration + 1
                                      }
                                    ),
                                    active = list(
                                      p = function() {
                                        as.integer(ncol(self$data$X))
                                      },
                                      m = function() {
                                        as.integer(nrow(self$data$t))
                                      },
                                      Kx = function(){
                                        # Implements linear kernel with inner product wrt diag(alpha)
                                        x1 = as.matrix(self$data$X)
                                        x2 = as.matrix(self$data$X)
                                        n1 = nrow(x1)
                                        n2 = nrow(x2)
                                        K = self$data$c^2 + x1%*%(t(x2)*(self$data$tau*self$data$lambda)^2)
                                        return(K+1e-9*diag(n1))
                                      },
                                      Kg = function(){
                                        K = self$Kx + diag(self$data$gamma^2)
                                        return(K+1e-9*diag(length(self$data$gamma)))
                                      },
                                      Kz = function(){
                                        K = diag(self$data$gamma^2)
                                        return(K+1e-9*diag(length(self$data$gamma)))
                                      },
                                      Kt = function(){
                                        # This implements the Matern3/2 kernel
                                        t1 = as.matrix(self$data$t)
                                        t2 = as.matrix(self$data$t)
                                        n1 = nrow(t1)
                                        n2 = nrow(t2)
                                        K = matrix(NA,nrow=n1,ncol=n2)
                                        for (i in 1:n1){
                                          for (j in 1:n2){
                                            K[i,j] = exp(-0.5*abs(t1[i,]-t2[j,])^2/self$data$ell^2)
                                          }
                                        }
                                        return(K+1e-9*diag(n1))
                                      },
                                      eigKt = function(){
                                        return(eigen(self$Kt,symmetric = T))
                                      },
                                      eigKx = function(){
                                        return(eigen(self$Kx,symmetric = T))
                                      },
                                      eigKg = function(){
                                        return(eigen(self$Kg,symmetric = T))
                                      },
                                      eigKz = function(){
                                        return(eigen(self$Kz,symmetric = T))
                                      },
                                      hypers_changed = function(){
                                        old = c(private$.cache$tau, private$.cache$lambda,
                                                private$.cache$c, private$.cache$ell,
                                                private$.cache$gamma)
                                        current = c(self$data$tau, self$data$lambda,
                                                    self$data$c, self$data$ell,
                                                    self$data$gamma)
                                        if (!identical(old,current)){
                                          TRUE
                                        } else {
                                          FALSE
                                        }
                                      },
                                      samples = function(){
                                        self$unthinned_samples[seq(1,dim(self$unthinned_samples)[1],by=self$thinning),,]
                                      }
                                    ),
                                    private = list(
                                      .cache = list()
                                    )
)
#' Matheron sampler using the Kronecker structure to sample Z
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
KroneckerMatheronSamplerZ = R6Class("KroneckerMatheronSampler",
                                    # Samples from the posterior of a GP with Kronecker-structured covariance
                                    # using Matheron's rule.
                                    public = list(
                                      N.iter = NULL,
                                      thinning = NULL,
                                      N.params = NULL,
                                      # samples = NULL,
                                      unthinned_samples = NULL,
                                      iteration = 1,
                                      data = NULL,
                                      control = list(
                                        window_counter = 0, # counting windows
                                        reject_counter = 0 # count rejections
                                      ),
                                      initialize = function(N.iter = 1000, N.params = NULL,
                                                            data = NULL, thinning = 10){
                                        self$N.iter = N.iter
                                        self$N.params = N.params
                                        self$unthinned_samples = array(NA,dim=c(N.iter*thinning,N.params))
                                        self$data = data
                                        self$thinning = thinning
                                        self$sample() # Init this by sampling
                                      },
                                      sample = function(){
                                        # I'll create the kernels here as well and do everything
                                        # First thing we do is check if hypers are changed
                                        if (self$hypers_changed){
                                          # print("shouldn't be here often")
                                          eigKt = self$eigKt
                                          eigKx = self$eigKx
                                          # And I'll store them in cache
                                          private$.cache$eigKt = eigKt
                                          private$.cache$eigKx = eigKx
                                          # And I'll store the current hypers in cache
                                          private$.cache$ell = self$data$ell
                                          private$.cache$gamma = self$data$gamma
                                        } else{
                                          eigKt = private$.cache$eigKt
                                          eigKx = private$.cache$eigKx
                                        }
                                        
                                        
                                        # Try-catch here
                                        step = tryCatch(
                                          {
                                            Y = self$data$Y
                                            sigma = self$data$sigma
                                            
                                            Qt = eigKt$vectors
                                            Qx = eigKx$vectors
                                            Dt = pmax(eigKt$values,0)
                                            Dx = pmax(eigKx$values,0)
                                            Kx = Qx%*%diag(Dx)%*%t(Qx)
                                            Kt = Qt%*%diag(Dt)%*%t(Qt)
                                            n = nrow(Qx)
                                            m = nrow(Qt)
                                            # Matheron's rule
                                            eta1 = matrix(rnorm(n*m),ncol=m)
                                            eta1 = sweep(eta1,1,sqrt(Dx),'*')
                                            eta1 = sweep(eta1,2,sqrt(Dt),'*')
                                            error = sigma*matrix(rnorm(n*m),ncol=m)
                                            f1.prior = Qx%*%eta1%*%t(Qt)
                                            eta = Y-f1.prior-error
                                            # Now compute correction, note subtraction of c^2 meaning no intercept
                                            correction = (((Kx)%*%Qx)%*%((1/(Dx%*% t(Dt)+sigma^2))*t(Qx)%*%eta%*%Qt)%*%t(Kt%*%Qt))
                                            f1.prior + correction
                                          },
                                          error = function(e) {
                                            warning(paste0("Error in Matheron step! ", e))
                                            # print(e)
                                            NULL
                                          }
                                        )
                                        reject = F
                                        if (is.null(step)){
                                          reject = T
                                        }
                                        if (reject){ # Reject
                                          self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-1,,]
                                          self$control$reject_counter = self$control$reject_counter + 1
                                        } else{ # Accept
                                          self$unthinned_samples[self$iteration,,] = step
                                          self$control$reject_counter = 0
                                        }
                                        if (self$control$reject_counter > 5){
                                          # Go back to a point where things worked
                                          self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-6,,]
                                        }
                                        # And increase iteration
                                        self$iteration = self$iteration + 1
                                      },
                                      skip = function(){
                                        self$unthinned_samples[self$iteration,,] = self$unthinned_samples[self$iteration-1,,]
                                        self$iteration = self$iteration + 1
                                      }
                                    ),
                                    active = list(
                                      p = function() {
                                        as.integer(ncol(self$data$X))
                                      },
                                      m = function() {
                                        as.integer(nrow(self$data$t))
                                      },
                                      Kx = function(){
                                        # Implements linear kernel with inner product wrt diag(alpha)
                                        x1 = as.matrix(self$data$X)
                                        x2 = as.matrix(self$data$X)
                                        n1 = nrow(x1)
                                        n2 = nrow(x2)
                                        K = diag(self$data$gamma^2)
                                        return(K+1e-9*diag(n1))
                                      },
                                      Kt = function(){
                                        # This implements the Matern3/2 kernel
                                        t1 = as.matrix(self$data$t)
                                        t2 = as.matrix(self$data$t)
                                        n1 = nrow(t1)
                                        n2 = nrow(t2)
                                        K = matrix(NA,nrow=n1,ncol=n2)
                                        for (i in 1:n1){
                                          for (j in 1:n2){
                                            K[i,j] = exp(-0.5*abs(t1[i,]-t2[j,])^2/self$data$ell^2)
                                          }
                                        }
                                        return(K+1e-9*diag(n1))
                                      },
                                      eigKt = function(){
                                        return(eigen(self$Kt,symmetric = T))
                                      },
                                      eigKx = function(){
                                        return(eigen(self$Kx,symmetric = T))
                                      },
                                      hypers_changed = function(){
                                        old = c(private$.cache$ell,
                                                private$.cache$gamma)
                                        current = c(self$data$ell,
                                                    self$data$gamma)
                                        if (!identical(old,current)){
                                          TRUE
                                        } else {
                                          FALSE
                                        }
                                      },
                                      samples = function(){
                                        self$unthinned_samples[seq(1,dim(self$unthinned_samples)[1],by=self$thinning),,]
                                      }
                                    ),
                                    private = list(
                                      .cache = list()
                                    )
)
#' Matheron sampler using the Kronecker structure to sample F with SKIM kernel
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
KroneckerMatheronSamplerSKIM = R6Class("HMCSampler",
                                       inherit = KroneckerMatheronSamplerF,
                                       active = list(
                                         Kx = function(){
                                           # Overwriting the linear kernel with SKIM
                                           x1 = as.matrix(self$data$X)
                                           x2 = as.matrix(self$data$X)
                                           n1 = nrow(x1)
                                           n2 = nrow(x2)
                                           xlxt = x1%*%(t(x2)*(self$data$lambda^2))
                                           x2lx2t = (x1^2)%*%(t((x2)^2)*(self$data$lambda^2))
                                           K = 0.5*self$data$tau2^2*(1+xlxt)^2  -
                                             0.5*self$data$tau2^2*(x2lx2t) +
                                             (self$data$tau1^2-self$data$tau2^2)*(xlxt) +
                                             self$data$c^2 - 0.5*self$data$tau2^2
                                           return(K+1e-6*diag(n1))
                                         }
                                       )
)
#' Gibbs sampler for variance
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
GibbsSamplerVariance = R6Class("Gibbs",
                               inherit = MCMC,
                               public = list(
                                 initialize = function(n, m,
                                                       sigma_sq_a = 2, 
                                                       sigma_sq_b = 0.1,
                                                       ...){
                                   super$initialize(...)
                                   self$data$n = n
                                   self$data$m = m
                                   self$data$sigma_sq_a = sigma_sq_a
                                   self$data$sigma_sq_b = sigma_sq_b
                                   self$sample()
                                 },
                                 sample = function(){
                                   S = sum((self$data$Y - (self$data$F + self$data$Z))^2)
                                   self$samples[self$iteration] = 1/rgamma(1,
                                                                           shape = self$data$sigma_sq_a + self$data$n*self$data$m/2,
                                                                           rate = self$data$sigma_sq_b + S/2)
                                   # And increase iteration
                                   self$iteration = self$iteration + 1
                                 }
                               )
)
#' Metropolis-Hastings sampler for length scale
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
MHSamplerEll = R6Class("MH",
                       inherit = MCMC,
                       public = list(
                         initialize = function(Y, Kx, Kz, t, s2, ell0,
                                               prop_sigma = 0.05,
                                               target_rate = 0.44, ...){
                           super$initialize(...)
                           self$data$Y = Y
                           self$data$prop_sigma = prop_sigma
                           self$data$Kx = Kx
                           self$data$Kz = Kz
                           self$data$t = t
                           self$data$s2 = s2
                           self$samples[1] = ell0
                           # self$iteration = self$iteration
                           self$data$target_rate = target_rate
                         },
                         sample = function(){
                           # First compute all the old stuff
                           Ky = self$data$Kx + self$data$Kz
                           Kt = private$Kt(ell=self$ell[self$iteration])
                           eigKy = eigen(Ky + 1e-9*diag(nrow(Ky)))
                           eigKt = eigen(Kt + 1e-9*diag(nrow(Kt)))
                           # Now a proposal
                           log_ell = log(self$ell[self$iteration])
                           log_ell_star = log_ell + rnorm(1,mean=0,sd=self$data$prop_sigma)
                           ell_star = exp(log_ell_star)
                           Kt_star = private$Kt(ell=ell_star)
                           eigKt_star = eigen(Kt_star + 1e-9*diag(nrow(Kt_star)))
                           # Acceptance ratio
                           Qt = eigKt$vectors
                           Qt_star = eigKt_star$vectors
                           Dt = eigKt$values
                           Dt_star = eigKt_star$values
                           Qg = eigKy$vectors
                           Dg = eigKy$values
                           # Helper quantities
                           Z = t(Qg)%*%self$data$Y%*%Qt
                           Z_star = t(Qg)%*%self$data$Y%*%Qt_star
                           # Inverse solves
                           inv_solve = sum((1/(Dt %*% t(Dg) + self$data$s2)) * t(Z^2))
                           inv_solve_star = sum((1/(Dt_star %*% t(Dg) + self$data$s2)) * t(Z_star^2))
                           # Log determinants
                           log_det = sum(log(Dt %*% t(Dg) + self$data$s2))
                           log_det_star = sum(log(Dt_star %*% t(Dg) + self$data$s2))
                           # Priors
                           prior = -0.5*log_ell^2
                           prior_star = -0.5*log_ell_star^2
                           # Posteriors
                           log_post = -0.5*(log_det + inv_solve) + prior
                           log_post_star = -0.5*(log_det_star + inv_solve_star) + prior_star
                           # Acceptance ratio
                           log_acc = log_post_star - log_post
                           acc = min(1,exp(log_acc))
                           # print(paste0("MH acce prob: ", round(acc,4)))
                           # Do we accept
                           if (rbinom(1,1,acc)){
                             self$samples[self$iteration+1] = exp(log_ell_star)
                           } else {
                             self$samples[self$iteration+1] = self$samples[self$iteration]
                           }
                           
                           # Update proposal variance Robbins-Monro
                           if (self$iteration < 1000){
                             c = 1
                             t0 = 50
                             a = 0.6
                             gamma_t = c / (self$iteration + t0)^a
                             self$data$prop_sigma = exp(log(self$data$prop_sigma) + gamma_t * (acc - self$data$target_rate))
                           }
                           
                           # And increase iteration
                           self$iteration = self$iteration + 1
                         }
                       ),
                       private = list(
                         Kt = function(ell){
                           # print(ell)
                           t1 = as.matrix(self$data$t)
                           t2 = as.matrix(self$data$t)
                           n1 = nrow(t1)
                           n2 = nrow(t2)
                           K = matrix(NA,nrow=n1,ncol=n2)
                           for (i in 1:n1){
                             for (j in 1:n2){
                               K[i,j] = exp(-0.5*abs(t1[i,]-t2[j,])^2/ell^2) # RBF
                             }
                           }
                           return(K+1e-9*diag(n1))
                         }),
                       active = list(
                         ell = function(){
                           return(self$samples)
                         }
                       )
)
#' Metropolis-Hastings sampler for sigma
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
MHSamplerSigma = R6Class("MH",
                         inherit = MCMC,
                         public = list(
                           initialize = function(Y, F, Z, s0,
                                                 prop_sigma = 0.05,
                                                 target_rate = 0.44, ...){
                             super$initialize(...)
                             self$data$Y = Y
                             self$data$prop_sigma = prop_sigma
                             self$data$F = F
                             self$data$Z = Z
                             self$samples[1] = s0
                             # self$iteration = self$iteration
                             self$data$target_rate = target_rate
                           },
                           sample = function(){
                             # Current value and proposal 
                             log_sigma = log(self$sigma[self$iteration])
                             log_sigma_star = log_sigma + rnorm(1,mean=0,sd=self$data$prop_sigma)
                             # MH ratio
                             log_acc = -(prod(dim(self$data$Y))+2*2)*(log_sigma_star - log_sigma) - (0.5*self$S+0.1)*(exp(-2*log_sigma_star)-exp(-2*log_sigma))
                             acc = min(1,exp(log_acc))
                             # print(paste0("MH acce prob: ", round(acc,4)))
                             # Do we accept
                             if (rbinom(1,1,acc)){
                               self$samples[self$iteration+1] = exp(log_sigma_star)
                             } else {
                               self$samples[self$iteration+1] = self$samples[self$iteration]
                             }
                             
                             # Update proposal variance Robbins-Monro
                             if (self$iteration < 1000){
                               c = 1
                               t0 = 50
                               a = 0.6
                               gamma_t = c / (self$iteration + t0)^a
                               self$data$prop_sigma = exp(log(self$data$prop_sigma) + gamma_t * (acc - self$data$target_rate))
                             }
                             
                             # And increase iteration
                             self$iteration = self$iteration + 1
                           }
                         ),
                         active = list(
                           S = function(){
                             return(sum((self$data$Y-(self$data$F + self$data$Z))^2))
                           },
                           sigma = function(){
                             return(self$samples)
                           }
                         )
)