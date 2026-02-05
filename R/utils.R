#' Function used to pull out estimated beta coefficients
#'
#' This function takes the fitted model as input and provides 
#' posterior samples of the beta coefficients
#' 
#'
#' @param fit : fitted model object
#' @param k : tuple of indices of betas to pull out, i.e. c(1,2) for beta_1 and beta_2. Default NULL provides all coefficients
#' @param N.samples : number of posterior samples (in range 1:N.iter), if NULL provides all samples
#' 
#' @return beta.hat : array of beta coefficients
#' 
#' @export
get_beta = function(fit, k = NULL, N.samples = NULL){
  # Pull out some stuff
  X = fit$data$X
  p = ncol(X)
  m = nrow(fit$data$t)
  N.iter = fit$data$N.iter
  if (is.null(N.samples)){
    N.samples = N.iter
  }
  F_sampler = fit$samplers$F_sampler
  F_hypers = fit$samplers$F_hypers
  ell_sampler = fit$samplers$ell_sampler
  # Which coefficients are we looking to pull out?
  if (is.null((k))){
    # Then we pull out everything
    id = seq(1,p)
  } else{
    id = k
  }
  k = length(id)
  beta.hat = array(0.0,dim=c(k,m,N.samples))
  # Set up matrix of coefficients to pull out
  E = matrix(0,ncol=p,nrow=k)
  for (kk in 1:k){
    E[kk,id[kk]] = 1
  }
  # Now iterate from final sample
  for (ii in 1:N.samples){
    i = (N.iter-N.samples) + ii
    # Pull out current values
    fi = F_sampler$samples[i,,]
    lambda = F_hypers$lambda[i,]
    tau = F_hypers$tau[i]
    c = F_hypers$c[i]
    ell = ell_sampler$ell[i]
    F_sampler$data$tau = tau
    F_sampler$data$lambda = lambda
    F_sampler$data$c = c
    F_sampler$data$ell = ell
    # Construct and decomp matrices
    Kt = F_sampler$Kt
    Lt = t(chol(Kt))
    # Prior sample from B
    Z = matrix(rnorm(k*m),ncol=m,nrow=k)
    B = (Z*(tau*lambda)[id])%*%t(Lt)
    # Prior sample from f | B is deterministic
    f = X[,id]%*%B
    # Now I have a joint sample from \pi(B,f), apply Matheron
    Kx = F_sampler$Kx
    Kx_star = E%*%(t(X)*(tau*lambda)^2) # TODO: This needs to look a bit different with interactions
    error = fi - f 
    correction = Kx_star%*%solve(Kx+1e-6*diag(nrow(Kx)),error)
    beta.hat[,,ii] = B + correction
  }
  return(beta.hat)
}