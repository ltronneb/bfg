#' Function used to pull out estimated beta coefficients
#'
#' This function takes the fitted model as input and provides 
#' posterior samples of the beta coefficients
#' 
#'
#' @param fit : fitted model object
#' @param k : tuple of indices of betas to pull out, i.e. c(1,2) for beta_1 and beta_2. Default NULL provides all coefficients
#' @param N_samples : number of posterior samples (in range 1:N_iter), if NULL provides all samples
#' 
#' @return beta.hat : array of beta coefficients
#' 
#' 
#' 
#' @export
get_beta = function(fit, k = NULL, N_samples = NULL){
  # Pull out some stuff
  X = fit$data$X
  p = ncol(X)
  m = nrow(fit$data$t)
  N_iter = fit$data$N_iter
  if (is.null(N_samples)){
    N_samples = N_iter
  }
  F_sampler = fit$samplers$F_sampler
  F_hypers = fit$samplers$F_hypers
  # ell_sampler = fit$samplers$ell_sampler
  # Which coefficients are we looking to pull out?
  if (is.null((k))){
    # Then we pull out everything
    id = seq(1,p)
  } else{
    id = k
  }
  k = length(id)
  beta.hat = array(0.0,dim=c(N_samples,k,m))
  # Set up matrix of coefficients to pull out
  E = matrix(0,ncol=p,nrow=k)
  for (kk in 1:k){
    E[kk,id[kk]] = 1
  }
  # Set up cached eigen decomps
  bfg_h5 = hdf5r::H5File$new(fit$h5file, mode="r")
  F.grp = bfg_h5$open(("F_sampler"))
  Qx = F.grp[["Qx"]]
  Qt = F.grp[["Qt"]]
  Dt = F.grp[["Dt"]]
  Dx = F.grp[["Dx"]]
  # Now iterate from final sample
  for (ii in 1:N_samples){
    # print(i)
    i = (N_iter-N_samples) + ii
    # Pull out current values
    fi = F_sampler$samples[i,,]
    lambda = F_hypers$lambda[i,]
    tau = F_hypers$tau[i]
    # c = F_hypers$c[i]
    # ell = ell_sampler$ell[i]
    Qti = Qt[i,,]
    Dti = Dt[i,]
    Qxi = Qx[i,,]
    Dxi = Dx[i,]
    # Construct and decomp matrices
    Lt = t(sqrt(Dti)*t(Qti))
    # Prior sample from B
    Z = matrix(rnorm(k*m),ncol=m,nrow=k)
    B = (Z*(tau*lambda)[id])%*%t(Lt)
    # Prior sample from f | B is deterministic
    f = X[,id]%*%B
    # Now I have a joint sample from \pi(B,f), apply Matheron
    Kx_star = E%*%(t(X)*(tau*lambda)^2) # TODO: This needs to look a bit different with interactions
    error = fi - f 
    correction = Kx_star%*%t(1/Dxi*t(Qxi))%*%(t(Qxi)%*%error)
    beta.hat[ii,,] = B + correction
  }
  return(beta.hat)
}


#' Function for finding non-zero coefficient
#'
#' This function applies the 'decoupling shrinkage and selection' (DSS) framework
#' of XXX, as adopted for functional data by Kowal 2020.
#' 
#' @param fit : bfg object
#' @param max_model_size : the maximum size of the model
#' @param N_samples : number of posterior samples used to compute rho
#' @param N_samples_weights : number of posterior samples used to compute weights
#' @param plot : boolean, display plot or not?
#' @param alpha : alpha level for variable selection
#' 
#' @return selected : list of selected coefficients
#'
#'@export
select_betas = function(fit, max_model_size = 100, N_samples = NULL, N_samples_weights = 10, plot=T, alpha=0.1){
  # Do some initial setup
  X = fit$data$X
  p = ncol(X)
  n = nrow(X)
  if (max_model_size > p){
    max_model_size = p
  }
  m = nrow(fit$data$t)
  N_iter = fit$data$N_iter
  warmup = fit$data$warmup
  if (is.null(N_samples)){
    N_samples = N_iter
  }
  F_sampler = fit$samplers$F_sampler
  F_hypers = fit$samplers$F_hypers
  Z_hypers = fit$samplers$Z_hypers
  # Pick out most likely active coefs
  id_j = order(apply(F_hypers$lambda[warmup:N_iter,],2,mean),decreasing = T)[1:max_model_size]
  beta.hat = bfg::get_beta(fit,k = id_j,N_samples = N_samples_weights)
  beta.mean = apply(beta.hat,c(2,3),mean,na.rm=T)
  # Pull out full model fit
  F.mean = apply(F_sampler$samples[warmup:N_iter,,],c(2,3),mean)
  # Set up design matrices
  w = 1/rowSums(beta.mean^2)
  X_sub = X[,id_j]
  X_sub0 = cbind(1,X_sub)
  # Fit glmnet
  glmnet_fit = glmnet::glmnet(x=X_sub,y=F.mean,family="mgaussian",
                              penalty.factor = w,
                              alpha = 1,
                              intercept = T)
  # Pull out coefs and reshape
  beta_lam = glmnet_fit$beta
  L = length(glmnet_fit$lambda) # Just get the number of lambda coefficients
  tmp = array(0.0,c(L,max_model_size+1,m)) # Now build the beta matrix per m
  for (i in 1:m){
    tmp[,,i] = t(rbind(glmnet_fit$a0[i,],(as.matrix(beta_lam[[i]]))))
  }
  beta_lam = tmp
  # Now compute proportion of variance explained
  rho2 = matrix(NA,nrow=N_samples,ncol=1)
  rho2_lam = matrix(NA,nrow=N_samples,ncol=L)
  for (ii in 1:N_samples){
    i = ii+warmup
    XB = F_sampler$samples[i,,]
    XB2 = sum((XB)^2)
    
    # Random effect trace
    post_trace = m*sum(Z_hypers$gamma[i,]^2) + n*m*Z_hypers$sigma_sq[i]
    
    # Zeroth-version
    rho2[ii] = XB2 / (XB2 + post_trace)
    
    # lam version
    err_lam <- numeric(L) # Summing over every lambda value
    for (ell in 1:L) {
      B_lam <- beta_lam[ell,,]              # (1 + K) x m
      XB_sparse <- X_sub0 %*% B_lam         # n x m
      err_lam[ell] <- sum((XB - XB_sparse)^2)
    }
    rho2_lam[ii,] <- XB2 / (XB2 + post_trace + err_lam)
  }
  # Plot in terms of model size
  # And for model size
  model_size = apply(beta_lam,1,function(b) sum(rowSums(b) != 0))
  # Replot with model size as xlab
  unq_size = unique(model_size)
  LL = length(unq_size)
  # Will collect some things here
  rho2_lam_quantiles = matrix(NA,ncol=LL,nrow=3)
  if (plot){
    plot(NA,ylim=c(0,1),xlim=c(0,max(model_size)-1),xlab="Model Size")
    abline(h=mean(rho2))
    abline(h=quantile(rho2,probs=c(alpha/2,1-alpha/2)))
  }
  for (l in 1:LL){
    ind_max = max(which(model_size == unq_size[l]))
    rr = quantile(rho2_lam[,ind_max],probs=c(alpha/2,1-alpha/2))
    rho2_lam_quantiles[c(1,2),l] = rr
    rho2_lam_quantiles[3,l] = ind_max
    if (plot){
      points(unq_size[l]-1,mean(rho2_lam[,ind_max]))
      segments(unq_size[l]-1,rr[1],unq_size[l]-1,rr[2])
    }
  }
  # Check if a model reaches the threshold?
  tresh_reached = which(rho2_lam_quantiles[2,]>=mean(rho2))
  if (length(tresh_reached)==0){
    # We were unable to reach threshold
    n_selected = max(unq_size) - 1
    opt_lam = rho2_lam_quantiles[3,LL]
    print("DSS was unable to reach predictive performance of posterior mean!")
  } else {
    id_selected = min(tresh_reached)
    n_selected = unq_size[id_selected] - 1 # Minus intercept
    opt_lam = rho2_lam_quantiles[3,id_selected]
  }
  selected = sort(id_j[which(rowSums(beta_lam[opt_lam,-1,])!=0)])
  print(paste0("DSS selected ", n_selected, " variables (excluding intercept)"))
  print("Selected coefficients:")
  print(selected)
  # I'll also return the full solution path for ROC curves
  models = list()
  for (kk in 1:length(model_size)){
    models[[kk]] = sort(id_j[which(rowSums(beta_lam[kk,-1,])!=0)])
  }
  return(list(selected = selected,glmnet_fit = glmnet_fit, models=unique(models)))
}

