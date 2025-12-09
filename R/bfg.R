#' Main function for fitting a function-on-scalar regression using the BFG model
#'
#'@param y: n x m matrix with observations
#'@param X: n x p design matrix
#'@param t: m vector of inputs
#'@param tau0_prime: scaling factor for the global horseshoe parameter
#'@param interactions: bool, estimate model with interactions? (default F)
#'@param thinning: int, how many Gibbs samples of F and Z per HMC draw of hypers (default 1)
#'@param N_iter: number of MCMC iterations (default 2000)
#'
#'@export
bfg = function(Y,X,t,tau0_prime,data_generated,interactions=F,thinning=1,N.iter=2000, plotting=F){
  # TODO check inputs are correctly formatted and dimensioned
  X = as.matrix(X)
  Y = as.matrix(Y)
  
  # Handle missing values in y
  missing = which(is.na(Y),arr.ind=T)
  print(sum(missing))
  working_Y = Y
  # Overwrite working Y with current imputed values (init from N(0,1))
  working_Y[missing] = rnorm(sum(missing))
  
  # Pick out some key numbers and quantities and define sensible inits
  p = ncol(X)
  m = length(t)
  n = nrow(Y)
  # ell0 = median(abs(outer(t,t,FUN="-")))
  ell0 = 0.1
  print(ell0)
  sigma0 = 0.22
  eta0 = 1.182
  F0 = working_Y
  Z0 = matrix(0,ncol=m,nrow=n)
  
  
  
  # Set up cache
  clear_cache()
  prepare_cache(X)
  # Set up samplers for F
  if (!interactions){
    F_hypers = HMC_samplerF$new(N.params=2*p+4,data = list(X = X,
                                                           t = t, 
                                                           Y = F0,
                                                           tau0_prime = tau0_prime,
                                                           nugget = 1e-6, ell = ell0),
                                N.iter = N.iter)
    F_sampler = KroneckerMatheronSampler$new(data = list(X=X,
                                                         t=t,
                                                         Y=working_Y,
                                                         ell=ell0,
                                                         c=F_hypers$c[1],
                                                         tau = F_hypers$tau[1],
                                                         lambda = F_hypers$lambda[1,],
                                                         sigma = sigma0),
                                             N.params = c(n,m),
                                             N.iter = F_hypers$N.iter,
                                             thinning = thinning)
  } else{
    F_hypers = HMC_samplerSKIM$new(N.params=2*p+5,data = list(X = X,
                                                              t = t, 
                                                              Y = F0,
                                                              tau0_prime = tau0_prime,
                                                              nugget = 1e-6, ell = ell0),
                                   N.iter = N_iter)
    F_sampler = KroneckerMatheronSamplerSKIM$new(data = list(X=X,
                                                             t=t,
                                                             Y=working_Y,
                                                             ell=ell0,
                                                             c=F_hypers$c[1],
                                                             tau1 = F_hypers$tau[1],
                                                             tau2 = F_hypers$tau2[1],
                                                             lambda = F_hypers$lambda[1,],
                                                             sigma = sigma0),
                                                 N.params = c(n,m),
                                                 N.iter = F_hypers$N.iter,
                                                 thinning = thinning)
  }
  
  # Found that it is generally good to init these samplers at large values,
  # such that all parameters are 'active' in the start
  F_hypers$samples[1,] = 2
  if (interactions){
    # but also for the interactions init with no interactions active
    F_hypers$samples[1,2*p+5] = - 5
  }
  
  # Set up samplers for Z
  Z_hypers = HMC_samplerZ$new(N.params = (n+1), data = list(X = diag(n),
                                                            t = t,
                                                            Y = Z0,
                                                            nugget = 1e-06, ell = ell0,
                                                            eta = eta0, 
                                                            beta_a = 1, beta_b =  200, dir_a = 0.1),
                              N.iter = N.iter)
  Z_sampler = KroneckerMatheronSampler$new(data = list(X=diag(n),
                                                       t=t,
                                                       Y=working_Y,
                                                       ell = ell0,
                                                       c=0,
                                                       tau = 1,
                                                       lambda = Z_hypers$gamma[1,],
                                                       sigma = sigma0),
                                           N.params = c(n,m),
                                           N.iter = F_hypers$N.iter+F_hypers$control$warmup,
                                           thinning = thinning)
  # Sampler for variance
  s2_sampler = GibbsSamplerVariance$new(n = n, m = m, sigma_sq_a = 20, sigma_sq_b = 1e-5,
                                        data = list(Y = working_Y, F = F_sampler$samples[1,,], 
                                                    Z = Z_sampler$samples[1,,]))
  # Sampler for lengthscale
  ell_sampler = MHSamplerEll$new(Y = working_Y,Kx = F_sampler$Kx, Kz = Z_sampler$Kx, ell0 = ell0, 
                                 t = data_generated$T, s2 = s2_sampler$samples[1],
                                 prop_sigma = 0.005)
  
  
  # Now sampling starts
  for (i in 2:N.iter){
    ############################################################################
    ###############  IMPUTING Y     ############################################
    ############################################################################
    # Start with imputing current values of Y
    imp_Y = matrix(rnorm(n*m,
                         F_sampler$samples[i-1,,]+Z_sampler$samples[i-1,,],
                         sqrt(s2_sampler$samples[i-1])),
                   ncol=m,nrow=n)
    working_Y[missing] = imp_Y[working_Y[missing]]
    
    
    ############################################################################
    ############### SAMPLING HYPERS ############################################
    ############################################################################
    F_hypers$data$Y = F_sampler$samples[i-1,,]
    # # Sample F_hypers
    F_hypers$sample()
    # # Sample Z_hypers
    eta = n*(F_hypers$c[i]^2 + sum((F_hypers$tau[i]*F_hypers$lambda[i,])^2) + s2_sampler$samples[i-1])
    Z_hypers$data$eta[i] = eta
    Z_hypers$data$Y = Z_sampler$samples[i-1,,]
    Z_hypers$sample()
    
    ############################################################################
    ############### SAMPLING FUNCTIONS #########################################
    ############################################################################
    # # Set up hypers 
    F_sampler$data$c = F_hypers$c[i]
    if (!interactions){
      F_sampler$data$tau = F_hypers$tau[i]
    } else{
      F_sampler$data$tau1 = F_hypers$tau[i]
      F_sampler$data$tau2 = F_hypers$tau2[i]
    }
    F_sampler$data$lambda = F_hypers$lambda[i,]
    Z_sampler$data$lambda = Z_hypers$gamma[i,]
    for (k in 0:(thinning-1)){
      Z_sampler$data$Y = working_Y - F_sampler$unthinned_samples[F_sampler$iteration-1,,]
      Z_sampler$sample()
      
      # # Sum-to-zero correction here # TRY TURNING THIS OFF!
      # B = diag(n) - 1/n*rep(1,n)%*%t(rep(1,n))
      # tmp = Z_sampler$unthinned_samples[Z_sampler$iteration-1,,]
      # tmp = B%*%tmp
      # Z_sampler$unthinned_samples[Z_sampler$iteration-1,,] = tmp
      
      F_sampler$data$Y = working_Y - Z_sampler$unthinned_samples[Z_sampler$iteration-1,,]
      F_sampler$sample()
    }
    ############################################################################
    ############### SAMPLING sigma_sq ##########################################
    ############################################################################
    s2_sampler$data$F = F_sampler$samples[i,,]
    s2_sampler$data$Z = Z_sampler$samples[i,,]
    s2_sampler$data$Y = working_Y
    s2_sampler$sample()
    # Update hypers in other samplers
    F_sampler$data$sigma = sqrt(s2_sampler$samples[i])
    Z_sampler$data$sigma = sqrt(s2_sampler$samples[i])
    ############################################################################
    ############### SAMPLING lengthscale #######################################
    ############################################################################
    ell_sampler$data$Kx = F_sampler$Kx
    ell_sampler$data$Kz = Z_sampler$Kx
    ell_sampler$data$s2 = s2_sampler$samples[i]
    ell_sampler$data$Y = working_Y
    ell_sampler$sample()
    # Update hypers in other samplers
    F_hypers$data$ell = ell_sampler$ell[i]
    Z_hypers$data$ell = ell_sampler$ell[i]
    F_sampler$data$ell = ell_sampler$ell[i]
    Z_sampler$data$ell = ell_sampler$ell[i]
    
    ## PLOTTING
    if (plotting){
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
      lines(F_sampler$samples[i,idx,] + 2*sqrt(s2_sampler$samples[i]),col="red",lty=2)
      lines(F_sampler$samples[i,idx,] - 2*sqrt(s2_sampler$samples[i]),col="red",lty=2)
      if (i > lag){
        lines(apply(F_sampler$samples[(i-lag):i,idx,],2,mean),col="black",lty=2)
      }
      plot(data_generated$Z[idx,],type="l",col="blue",lty=2,main=paste0("Z"," iter=",i))
      points((data_generated$Y - ((data_generated$F.true - data_generated$Z)))[idx,])
      points((data_generated$Y - F_sampler$samples[i,,])[idx,],col="green")
      lines(Z_sampler$samples[i,idx,],col="red")
      lines(Z_sampler$samples[i,idx,] + 2*sqrt(s2_sampler$samples[i]),col="red",lty=2)
      lines(Z_sampler$samples[i,idx,] - 2*sqrt(s2_sampler$samples[i]),col="red",lty=2)
      if (i > lag){
        lines(apply(Z_sampler$samples[(i-lag):i,idx,],2,mean),col="black",lty=2)
      }
      plot(F_hypers$tau[i]*F_hypers$lambda[i,])
      Sys.sleep(0.1)
      par(mfrow=c(1,1))
    }
  }
  
  # Now return the samplers and create an S4 object
  
  
}




