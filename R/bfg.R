#' Main function for fitting a function-on-scalar regression using the BFG model
#'
#'@param Y : n x m matrix with observations
#'@param X : n x p design matrix
#'@param t : m vector of inputs
#'@param p0 : expected number of non-zero coefficients
#'@param data_generated : list returned by gen_data, for plotting simulations
#'@param interactions : bool, estimate model with interactions? (default F)
#'@param thinning : int, how many Gibbs samples of F and Z per HMC draw of hypers (default 1)
#'@param N_iter : number of MCMC iterations (default 2000)
#'@param plotting : bool, plot fitted curves and variable selection on the fly?
#'@param compute_betas : bool, compute posterior samples of beta on the fly (not reccomended for large p)
#'@param verbose : bool, print messages from samplers?
#'
#'@export
bfg = function(Y,X,t,p0,data_generated=NULL,
               interactions=F,thinning=1,N_iter=2000, 
               plotting=F, compute_betas = F,
               verbose = F){
  # TODO check inputs are correctly formatted and dimensioned
  X = as.matrix(X)
  Y = as.matrix(Y)
  
  # Handle missing values in y
  missing = which(is.na(Y),arr.ind=T)
  # print(nrow(missing))
  working_Y = Y
  # Overwrite working Y with current imputed values (init from N(0,1))
  working_Y[missing] = rnorm(nrow(missing))
  
  # Pick out some key numbers and quantities and define sensible inits
  p = ncol(X)
  m = length(t)
  n = nrow(Y)
  
  # Inits
  ell0 = median(abs(outer(t,t,FUN="-")))/2
  sigma0 = sqrt(mean(apply(Y,1,var))/4)
  Z0 = matrix(0,ncol=m,nrow=n)
  F0 = Z0
  
  # Set up tau0_prime0 for global scale parameter
  tau0_prime0 = m*(p0/p)*(1/sqrt(n))
  
  # Set up caches
  clear_cache()
  prepare_cache(X)
  # Set up hdf5r the samplers can use to write to file
  bfg_file = tempfile(fileext = ".h5")
  bfg_h5 = hdf5r::H5File$new(bfg_file,mode="w")
  print("Temporary file set up:")
  print(bfg_h5)
  
  # Set up tau_prime sequence
  tau0_prime = rep(NA,N_iter)
  tau0_prime[1:2] = tau0_prime0*sigma0
  # Set up samplers for F
  if (!interactions){
    F_hypers = HMC_samplerF$new(N_params=2*p+4,data = list(X = X,
                                                           t = t, 
                                                           Y = F0,
                                                           tau0_prime = tau0_prime,
                                                           nugget = 1e-6, ell = ell0),
                                N_iter = N_iter,verbose=verbose)
    F_sampler = KroneckerMatheronSamplerF$new(data = list(X=X,
                                                          t=t,
                                                          Y=working_Y,
                                                          ell=ell0,
                                                          c=F_hypers$c[1],
                                                          tau = F_hypers$tau[1],
                                                          lambda = F_hypers$lambda[1,],
                                                          gamma = rep(0,n),
                                                          sigma = sigma0),
                                              N_params = c(n,m),
                                              N_iter = F_hypers$N_iter,
                                              thinning = thinning,
                                              init = F0,
                                              h5file = bfg_h5)
  } else{
    F_hypers = HMC_samplerSKIM$new(N_params=2*p+5,data = list(X = X,
                                                              t = t, 
                                                              Y = F0,
                                                              tau0_prime = tau0_prime,
                                                              nugget = 1e-6, ell = ell0),
                                   N_iter = N_iter,verbose=verbose)
    F_hypers$control$max_L = 2^10
    F_hypers$control$delta = 0.95
    F_sampler = KroneckerMatheronSamplerSKIM$new(data = list(X=X,
                                                             t=t,
                                                             Y=working_Y,
                                                             ell=ell0,
                                                             c=F_hypers$c[1],
                                                             tau1 = F_hypers$tau[1],
                                                             tau2 = F_hypers$tau2[1],
                                                             lambda = F_hypers$lambda[1,],
                                                             gamma = rep(0,n),
                                                             sigma = sigma0),
                                                 N_params = c(n,m),
                                                 N_iter = F_hypers$N_iter,
                                                 thinning = thinning,
                                                 init = F0,
                                                 h5file = bfg_h5)
  }
  
  # Found that it is generally good to init these samplers at large values,
  # such that all parameters are 'active' in the start
  F_hypers$samples[1,] = 2
  if (interactions){
    # but also for the interactions init with no interactions active
    # F_hypers$samples[1,2*p+5] = -2
  }
  
  # Set up eta parameter for r2d2 prior
  eta0 = rep(NA,N_iter)
  eta0[1] = n*(F_hypers$c[1]^2 + sum((F_hypers$tau[1]*F_hypers$lambda[1,])^2) + sigma0^2)
  if (interactions){
    eta0[1] = sum(diag(F_sampler$Kx))+n*sigma0^2
  }
  
  # Set up samplers for Z
  # Temperature scheduler
  temp = rep(1,N_iter) # turn this off for now -- constant temperature
  # temp = c(seq(0,1,length.out=1000)^4,rep(1,N_iter-1000))
  # temp = c(rep(0,100),rep(1,N.iter-100))
  Z_hypers = HMC_samplerZ_noise$new(N_params = (n+2), data = list(X = diag(n),
                                                                  t = t,
                                                                  Y = working_Y-F_sampler$samples[1,,],
                                                                  temperature = temp,
                                                                  nugget = 1e-06, ell = ell0,
                                                                  eta = eta0,
                                                                  beta_gamma_a = 1, beta_gamma_b =  20, dir_a = 1),
                                    N_iter = N_iter,verbose=verbose)
  Z_sampler = KroneckerMatheronSamplerZ$new(data = list(X=diag(n),
                                                        t=t,
                                                        Y=working_Y-F_sampler$samples[1,,],
                                                        ell = ell0,
                                                        gamma = Z_hypers$gamma[1,],
                                                        sigma = sigma0),
                                            N_params = c(n,m),
                                            N_iter = N_iter,
                                            thinning = thinning,
                                            init = Z0)
  # Sampler for lengthscale
  ell_sampler = MHSamplerEll$new(Y = working_Y,Kx = F_sampler$Kx, Kz = Z_sampler$Kx, ell0 = ell0,
                                 t = t, 
                                 s2 = Z_hypers$sigma_sq[1],
                                 prop_sigma = 0.005)
  # Container for betas
  if (compute_betas){
    beta.hat = array(0.0,dim=c(N_iter,p,m))
  } else{
    beta.hat = NULL
  }
  
  # Now return the samplers and create an S4 object
  L = list(samplers = list(F_sampler=F_sampler, F_hypers=F_hypers,
                           Z_sampler=Z_sampler, Z_hypers=Z_hypers,
                           ell_sampler=ell_sampler),
           data = list(Y=Y,X=X,t=t,tau0_prime=tau0_prime,
                       data_generated=data_generated,
                       interactions=interactions,thinning=thinning,
                       N_iter=N_iter, plotting=plotting,warmup = floor(N_iter/2)),
           beta.hat = beta.hat,
           h5file = bfg_file)
  # On exit clause
  on.exit(return(L))
  on.exit(bfg_h5$close_all(),add = TRUE)
  # Now sampling starts
  # Start timer
  t1 = Sys.time()
  for (i in 2:N_iter){
    ############################################################################
    ###############  IMPUTING Y     ############################################
    ############################################################################
    # Start with imputing current values of Y
    imp_Y = matrix(rnorm(n*m,
                         F_sampler$samples[i-1,,]+Z_sampler$samples[i-1,,],
                         Z_hypers$sigma[i-1]),
                   ncol=m,nrow=n)
    working_Y[missing] = imp_Y[missing]
    
    
    ############################################################################
    ############### SAMPLING HYPERS ############################################
    ############################################################################
    # Sample F_hypers
    F_hypers$data$gamma = rep(0,n)
    F_hypers$data$Y = F_sampler$samples[i-1,,]
    F_hypers$data$tau0_prime[i] = tau0_prime0*Z_hypers$sigma[i-1]
    F_hypers$sample()
    # Set up hypers for F
    F_sampler$data$c = F_hypers$c[i]
    if (!interactions){
      F_sampler$data$tau = F_hypers$tau[i]
    } else{
      F_sampler$data$tau1 = F_hypers$tau[i]
      F_sampler$data$tau2 = F_hypers$tau2[i]
    }
    F_sampler$data$lambda = F_hypers$lambda[i,]
    F_sampler$data$Y = working_Y
    
    
    
    
    # Sample Z_hypers
    eta = n*(F_hypers$c[i]^2 + sum((F_hypers$tau[i]*F_hypers$lambda[i,])^2) + Z_hypers$sigma[i-1]^2)
    # TODO change this for interactions!
    if (interactions){
      eta = sum(diag(F_sampler$Kx))+n*Z_hypers$sigma[i-1]^2
    }
    Z_hypers$data$eta[i] = eta
    Z_hypers$data$Y = working_Y - F_sampler$samples[i-1,,]
    Z_hypers$sample()
    
    # Print proportion of variance explained
    gamma_sum = sum(Z_hypers$gamma[i,]^2)
    eta_ratio = (gamma_sum)/(eta + gamma_sum)
    if (verbose){
      print(paste0("Variance explained by random effects: ", round(100*eta_ratio,2),"%"))
    }
    
    # Set up hypers for F and Z
    F_sampler$data$gamma = Z_hypers$gamma[i,]
    Z_sampler$data$gamma = Z_hypers$gamma[i,]
    
    ############################################################################
    ############### SAMPLING lengthscale #######################################
    ############################################################################
    ell_sampler$data$Kx = F_sampler$Kx
    ell_sampler$data$Kz = Z_sampler$Kx
    ell_sampler$data$s2 = Z_hypers$sigma_sq[i]
    ell_sampler$data$Y = working_Y
    ell_sampler$sample()
    
    # Update hypers in other samplers
    F_hypers$data$ell = ell_sampler$ell[i]
    Z_hypers$data$ell = ell_sampler$ell[i]
    F_sampler$data$ell = ell_sampler$ell[i]
    Z_sampler$data$ell = ell_sampler$ell[i]
    F_sampler$data$sigma = Z_hypers$sigma[i]
    Z_sampler$data$sigma = Z_hypers$sigma[i]
    
    ############################################################################
    ############### SAMPLING FUNCTIONS #########################################
    ############################################################################
    
    for (k in 0:(thinning-1)){
      # Sample F
      F_sampler$sample()
      # Sample Z conditional on F
      Z_sampler$data$Y = working_Y - F_sampler$unthinned_samples[F_sampler$iteration,,]
      Z_sampler$sample()
      
      # Sum-to-zero correction here
      B = diag(n) - 1/n*rep(1,n)%*%t(rep(1,n))
      tmp = Z_sampler$unthinned_samples[Z_sampler$iteration-1,,]
      tmp = B%*%tmp
      Z_sampler$unthinned_samples[Z_sampler$iteration-1,,] = tmp
    }
    
    ############################################################################
    ###############   COMPUTING BETA   #########################################
    ############################################################################
    if (!interactions){
      if (compute_betas){
        eigKt = F_sampler$eigKt
        fi = F_sampler$samples[i,,]
        lambda = F_hypers$lambda[i,]
        tau = F_hypers$tau[i]
        Lt = t(t(eigKt$vectors)*sqrt(eigKt$values))
        # Sample from prior
        Z = matrix(rnorm(p*m),ncol=m,nrow=p)
        B = (Z*(tau*lambda))%*%t(Lt)
        # Prior sample from f | B is deterministic
        f = X%*%B
        # Now I have a joint sample from \pi(B,f), apply Matheron
        Kx = F_sampler$Kx
        Kx_star = (t(X)*(tau*lambda)^2) # TODO: This needs to look a bit different with interactions
        error = fi - f 
        correction = Kx_star%*%solve(Kx+1e-6*diag(nrow(Kx)),error)
        L$beta.hat[i,,] = B + correction
      }
    } else{
      print("Computing coefficients not enabled for interaction models")
    }
    
    
    
    
    # PLOTTING
    if (plotting & !is.null(data_generated)){
      idx = 11
      lag = 100
      par(mfrow=c(2,2))
      plot(data_generated$F.true[idx,],col="blue",lty=2,type="l",main=paste0("F + Z"," iter=",i))
      points(data_generated$Y[idx,])
      lines(F_sampler$samples[i,idx,]+Z_sampler$samples[i,idx,],col="red")
      if (i > lag){
        lines(apply(F_sampler$samples[(i-lag):i,idx,] + Z_sampler$samples[(i-lag):i,idx,],2,mean),col="black",lty=2)
      }
      # lines(F_sampler$samples[i,idx,],col="red")
      plot((data_generated$F.true - data_generated$Z)[idx,],type="l",col="blue",lty=2, main=paste0("F"," iter=",i))
      points((data_generated$Y - data_generated$Z)[idx,])
      points((data_generated$Y - Z_sampler$samples[i,,])[idx,],col="green")
      lines(F_sampler$samples[i,idx,],col="red")
      lines(F_sampler$samples[i,idx,] + 2*(Z_hypers$sigma[i]),col="red",lty=2)
      lines(F_sampler$samples[i,idx,] - 2*(Z_hypers$sigma[i]),col="red",lty=2)
      if (i > lag){
        lines(apply(F_sampler$samples[(i-lag):i,idx,],2,mean),col="black",lty=2)
      }
      plot(data_generated$Z[idx,],type="l",col="blue",lty=2,main=paste0("Z"," iter=",i))
      points((data_generated$Y - ((data_generated$F.true - data_generated$Z)))[idx,])
      points((data_generated$Y - F_sampler$samples[i,,])[idx,],col="green")
      lines(Z_sampler$samples[i,idx,],col="red")
      lines(Z_sampler$samples[i,idx,] + 2*(Z_hypers$sigma[i]),col="red",lty=2)
      lines(Z_sampler$samples[i,idx,] - 2*(Z_hypers$sigma[i]),col="red",lty=2)
      if (i > lag){
        lines(apply(Z_sampler$samples[(i-lag):i,idx,],2,mean),col="black",lty=2)
      }
      plot(F_hypers$lambda[i,])
      Sys.sleep(0.1)
      par(mfrow=c(1,1))
    }
    #### Manage time expectations
    if (i %% 100 == 0){
      newtime = Sys.time()
      dt = as.numeric(difftime(newtime,t1,units="secs"))
      time_remaining = (dt / i)  * (N_iter-i)
      print(paste0("Time remaining: ", round(time_remaining / 60, 2), " minutes"))
    }
    
  }
  t2 = Sys.time()
  L$time_elapsed = t2 - t1
}
  
  
  
  
  