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
    N_iter = NULL, # number of iterations
    N_params = NULL, # number of parameters
    data = NULL,
    iteration = NULL, # current iteration
    samples = NULL, # object for storing samples
    h5file = NULL,
    verbose = F,
    # has_h5file = FALSE,
    initialize = function(N_iter = 2000,
                          N_params = NULL,
                          data = NULL,
                          init = NULL,
                          h5file = NULL,
                          verbose = NULL){
      self$N_iter = N_iter
      self$N_params = N_params
      self$data = data
      self$iteration = 1
      self$samples = array(NA, dim=c(N_iter,N_params))
      # Init
      if (!is.null(N_params)){
        if (!is.null(init)){
          self$samples[1,] = init
        } else {self$samples[1,] = runif(N_params,min=-2,max=2)}
      }
      if (!is.null(h5file)){
        self$h5file <- h5file
      }
      if (!is.null(verbose)){
        self$verbose = verbose
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
    initialize = function(#N_iter = 1000,
      #N_params = NULL,
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
      if (self$N_iter < self$control$warmup){
        print(paste0("N_iter must be larger than ", self$control$warmup))
        stop()
      }
      
      if (self$control$adapt){
        self$control$adapt_epsilon = T
        self$control$adapt_L = T
        self$control$adapt_mass = F # don't adapt mass matrix from initial samples
      }
      if (!is.null(self$data$t)){self$data$tlist = lapply(1:nrow(self$data$t), function(i) matrix(self$data$t[i, ], ncol = 1))}
      # Containers
      self$control$L = rep(0.0, self$N_iter)
      self$control$epsilon = rep(0.0, self$N_iter)
      self$control$alpha = rep(0.0, self$N_iter)
      # Initialise
      self$control$L[1] = L0
      self$control$epsilon[1] = epsilon0
      self$control$alpha[1] = alpha0
      if (!is.null(self$N_params)){
        self$control$mass_matrix = rep(1,self$N_params)
      }
      self$control$x = log(epsilon0)
      self$control$mu = log(10*epsilon0)
      self$control$xbar = self$control$x
      self$control$phases =  c(self$control$init_buffer, self$control$init_buffer + cumsum(self$control$window), self$control$init_buffer + sum(self$control$window) + self$control$term_buffer)
    },
    sample = function(){
      print("Not implemented, must be implemented in inhereted subclass")
      stop()
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
      if(self$verbose){
        print(paste0("Sampling! ", "current epsilon = ", round(self$current_epsilon,4),
                     " current L: ", self$current_L,
                     " Acceptance probability: ", round(self$control$alpha[self$iteration],2),
                     " iteration=",self$iteration-1))
      }
      
      
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
        if (self$verbose){
          print("Next phase!")
        }
        # Reset some stuff for estimation of epsilon
        # (as long as we are still adapting this)
        if (self$control$adapt & self$control$adapt_epsilon){
          if (self$verbose){
            print("Epsilon stuff reset")
          }
          self$control$x = log(self$control$epsilon[1]) # epsilon0
          self$control$mu = log(10*self$control$epsilon[1]) # epsilon0
          self$control$xbar = self$control$x
          self$control$H = 0
          self$control$t = 1
        }
        
        # Estimate mass matrix using buffer
        if (self$control$adapt & self$control$adapt_mass){
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
HMC_samplerF = R6Class("F_hypers",
                       inherit = HMC,
                       public = list(
                         initialize = function(slab_scale = 4, slab_df = 4.0, 
                                               nu_local = 1, nu_global = 1,...){
                           super$initialize(...)
                           self$data$slab_scale = slab_scale
                           self$data$slab_df = slab_df
                           self$data$nu_local = nu_local
                           self$data$nu_global = nu_global
                           # Init h5 file here
                           # handle this logic here
                         },
                         sample = function(){
                           eigKt = private$.eigKt()
                           Qt = eigKt$vectors
                           Dt = eigKt$values
                           step = tryCatch(
                             sample_f_hypers(self$data$X, Qt, Dt, 
                                             self$data$Y,self$data$gamma,
                                             self$data$tau0_prime[self$iteration],
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
                           as.integer(self$N_params - 4)/2 # Change this to n-4 after
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
HMC_samplerZ = R6Class("Z_hypers",
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
                           self$N_params - 1
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
HMC_samplerSKIM = R6Class("F_hypers",
                          inherit = HMC_samplerF,
                          public = list(
                            sample = function(){
                              eigKt = private$.eigKt()
                              Qt = eigKt$vectors
                              Dt = eigKt$values
                              step = tryCatch(
                                sample_f_hypers_SKIM(self$data$X, Qt, Dt, 
                                                     self$data$Y,
                                                     self$data$tau0_prime[self$iteration],
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
                              as.integer(self$N_params - 5)/2 # Change this to n-4 after
                            },
                            v = function(){ # And comment out this
                              # v = exp(0.5*self$samples[,2*self$p+5])
                              v = exp(self$samples[,2*self$p+5])
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
KroneckerMatheronSamplerF = R6Class("F_sampler",
                                    # Samples from the posterior of a GP with Kronecker-structured covariance
                                    # using Matheron's rule.
                                    public = list(
                                      N_iter = NULL,
                                      thinning = NULL,
                                      N_params = NULL,
                                      # samples = NULL,
                                      unthinned_samples = NULL,
                                      iteration = 1,
                                      data = NULL,
                                      # Deal with local cache for eigendecomps
                                      h5file = NULL,
                                      h5Qx = NULL,
                                      h5Dx = NULL,
                                      h5Qt = NULL,
                                      h5Dt = NULL,
                                      control = list(
                                        window_counter = 0, # counting windows
                                        reject_counter = 0 # count rejections
                                      ),
                                      initialize = function(N_iter = 2000, N_params = NULL,
                                                            data = NULL, thinning = 10,
                                                            init = NULL, h5file = NULL){
                                        self$N_iter = N_iter
                                        self$N_params = N_params
                                        self$unthinned_samples = array(NA,dim=c(N_iter*thinning,N_params))
                                        self$data = data
                                        self$thinning = thinning
                                        self$unthinned_samples[1,,] = init
                                        self$iteration = 1 # Current iteration
                                        # H5 stuff
                                        if (!is.null(h5file)){
                                          self$h5file = h5file
                                          self$h5file = self$h5file$create_group("F_sampler")
                                          self$h5Qx = self$h5file$create_dataset(name="Qx",dims=c(N_iter,self$N_params[1],self$N_params[1]),
                                                                                       chunk_dims = c(1,self$N_params[1],self$N_params[1]),
                                                                                       dtype = h5types$double)
                                          self$h5Dx = self$h5file$create_dataset(name="Dx",dims=c(N_iter,self$N_params[1]),
                                                                                       chunk_dims = c(1,self$N_params[1]),
                                                                                       dtype = h5types$double)
                                          self$h5Qt = self$h5file$create_dataset(name="Qt",dims=c(N_iter,self$N_params[2],self$N_params[2]),
                                                                                       chunk_dims = c(1,self$N_params[2],self$N_params[2]),
                                                                                       dtype = h5types$double)
                                          self$h5Dt = self$h5file$create_dataset(name="Dt",dims=c(N_iter,self$N_params[2]),
                                                                                       chunk_dims = c(1,self$N_params[2]),
                                                                                       dtype = h5types$double)
                                        }
                                      },
                                      sample = function(){
                                        # I'll create the kernels here as well and do everything
                                        # First thing we do is check if hypers are changed
                                        if (self$hypers_changed){
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
                                        
                                        # And in any case we store local versions here
                                        if (!is.null(self$h5file)){
                                          self$h5Qx[self$iteration+1,,] = eigKx$vectors
                                          self$h5Dx[self$iteration+1,] = eigKx$values
                                          self$h5Qt[self$iteration+1,,] = eigKt$vectors
                                          self$h5Dt[self$iteration+1,] = eigKt$values 
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
                                          self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration,,]
                                          self$control$reject_counter = self$control$reject_counter + 1
                                        } else{ # Accept
                                          self$unthinned_samples[self$iteration+1,,] = step
                                          self$control$reject_counter = 0
                                        }
                                        if (self$control$reject_counter > 5){
                                          # Go back to a point where things worked
                                          self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration-5,,]
                                        }
                                        # And increase iteration
                                        self$iteration = self$iteration + 1
                                      },
                                      skip = function(){
                                        self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration,,]
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
                                        return(K)
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
KroneckerMatheronSamplerZ = R6Class("Z_sampler",
                                    # Samples from the posterior of a GP with Kronecker-structured covariance
                                    # using Matheron's rule.
                                    public = list(
                                      N_iter = NULL,
                                      thinning = NULL,
                                      N_params = NULL,
                                      # samples = NULL,
                                      unthinned_samples = NULL,
                                      iteration = 1,
                                      data = NULL,
                                      control = list(
                                        window_counter = 0, # counting windows
                                        reject_counter = 0 # count rejections
                                      ),
                                      initialize = function(N_iter = 1000, N_params = NULL,
                                                            data = NULL, thinning = 10,
                                                            init = NULL){
                                        self$N_iter = N_iter
                                        self$N_params = N_params
                                        self$unthinned_samples = array(NA,dim=c(N_iter*thinning,N_params))
                                        self$data = data
                                        self$thinning = thinning
                                        self$unthinned_samples[1,,] = init
                                        # self$sample() # Init this by sampling
                                        self$iteration = 1 # Current iteration
                                      },
                                      sample = function(){
                                        # I'll create the kernels here as well and do everything
                                        # First thing we do is check if hypers are changed
                                        if (self$hypers_changed){
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
                                            NULL
                                          }
                                        )
                                        reject = F
                                        if (is.null(step)){
                                          reject = T
                                        }
                                        if (reject){ # Reject
                                          self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration,,]
                                          self$control$reject_counter = self$control$reject_counter + 1
                                        } else{ # Accept
                                          self$unthinned_samples[self$iteration+1,,] = step
                                          self$control$reject_counter = 0
                                        }
                                        if (self$control$reject_counter > 5){
                                          # Go back to a point where things worked
                                          self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration-5,,]
                                        }
                                        # And increase iteration
                                        self$iteration = self$iteration + 1
                                      },
                                      skip = function(){
                                        self$unthinned_samples[self$iteration+1,,] = self$unthinned_samples[self$iteration,,]
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
KroneckerMatheronSamplerSKIM = R6Class("F_sampler",
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
#' Metropolis-Hastings sampler for length scale
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
MHSamplerEll = R6Class("Ell_sampler",
                       inherit = MCMC,
                       public = list(
                         initialize = function(Y, Kx, Kz, t, s2, ell0,
                                               prop_sigma = 0.5,
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
                           inv_solve = sum((1/(Dg %*% t(Dt) + self$data$s2)) * (Z^2))
                           inv_solve_star = sum((1/(Dg %*% t(Dt_star) + self$data$s2)) * (Z_star^2))
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
                           # acc_ratio = min(1,exp(log_acc))
                           # acc = rbinom(1,1,acc_ratio)
                           accept = is.finite(log_acc) && (log(runif(1)) < log_acc)
                           acc = as.integer(accept)
                           # Do we accept
                           if (acc){
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
                             # gamma_t = 1/self$iteration
                             self$data$prop_sigma = exp(log(self$data$prop_sigma) + gamma_t * (acc - self$data$target_rate))
                           }
                           
                           # And increase iteration
                           self$iteration = self$iteration + 1
                         }
                       ),
                       private = list(
                         Kt = function(ell){
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
#' HMC sampler for Z hypers and noise term
#'
#'@keywords internal
#'@noRd
#'@importFrom R6 R6Class
HMC_samplerZ_noise = R6Class("Z_hypers",
                             inherit = HMC,
                             public = list(
                               sample = function(){
                                 eigKt = private$.eigKt()
                                 Qt = eigKt$vectors
                                 Dt = eigKt$values
                                 step = tryCatch(
                                   sample_z_hypers_and_noise(Qt, Dt,
                                                             self$data$Y,
                                                             self$samples[self$iteration,], 
                                                             self$data$temperature[self$iteration],
                                                             self$control$mass_matrix, 
                                                             self$current_epsilon,
                                                             self$current_L,
                                                             self$data$eta[self$iteration],
                                                             self$data$beta_gamma_a,
                                                             self$data$beta_gamma_b,
                                                             self$data$dir_a
                                   ),
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
                                 self$N_params - 2
                               },
                               log_phi_tilde = function(){
                                 self$samples[,1:self$n]
                               },
                               logit_u = function(){
                                 self$samples[,(self$n + 1)]
                               },
                               log_sigma = function(){
                                 self$samples[,(self$n + 2)]
                               },
                               sigma = function(){
                                 exp(self$log_sigma)
                               },
                               sigma_sq = function(){
                                 self$sigma^2
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
