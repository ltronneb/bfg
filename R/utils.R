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
  # ell_sampler = fit$samplers$ell_sampler
  # Which coefficients are we looking to pull out?
  if (is.null((k))){
    # Then we pull out everything
    id = seq(1,p)
  } else{
    id = k
  }
  k = length(id)
  beta.hat = array(0.0,dim=c(N.samples,k,m))
  # Set up matrix of coefficients to pull out
  E = matrix(0,ncol=p,nrow=k)
  for (kk in 1:k){
    E[kk,id[kk]] = 1
  }
  # Set up cached eigen decomps
  bfg_h5 = H5File$new(fit$h5file, mode="r")
  F.grp = bfg_h5$open(("F_sampler"))
  Qx = F.grp[["Qx"]]
  Qt = F.grp[["Qt"]]
  Dt = F.grp[["Dt"]]
  Dx = F.grp[["Dx"]]
  # Now iterate from final sample
  for (ii in 1:N.samples){
    # print(i)
    i = (N.iter-N.samples) + ii
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
    # Kt = F_sampler$Kt
    # Lt = t(chol(Kt))
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
#'@export
select_betas = function(fit, max_model_size = 100, N.samples = NULL, plot=T){
  # Do some initial setup
  X = fit$data$X
  p = ncol(X)
  m = nrow(fit$data$t)
  N.iter = fit$data$N.iter
  warmup = fit$data$warmup
  if (is.null(N.samples)){
    N.samples = N.iter
  }
  F_sampler = fit$samplers$F_sampler
  F_hypers = fit$samplers$F_hypers
  Z_hypers = fit$samplers$Z_hypers
  # Pick out most likely active coefs
  id_j = order(apply(F_hypers$lambda[warmup:N.iter,],2,mean),decreasing = T)[1:max_model_size]
  beta.hat = bfg::get_beta(fit,k = id_j,N.samples = N.samples)
  beta.mean = apply(beta.hat,c(2,3),mean,na.rm=T)
  # Pull out full model fit
  F.mean = apply(F_sampler$samples[warmup:N.iter,,],c(2,3),mean)
  # Set up design matrices
  w = 1/rowSums(beta.mean^2)
  X_sub = X[,id_j]
  X_sub0 = cbind(1,X_sub)
  # Fit glmnet
  glmnet_fit = glmnet::glmnet(x=X_sub,y=F.mean,family="mgaussian",
                              penalty.factor = w,
                              alpha = 1,
                              intercept = T,
                              lambda.min.ratio = 0,
                              nlambda = 100)
  # Pull out coefs and reshape
  beta_lam = glmnet_fit$beta
  L = length(glmnet_fit$lambda) # Just get the number of lambda coefficients
  tmp = array(0.0,c(L,max_model_size+1,m)) # Now build the beta matrix per m
  for (i in 1:m){
    tmp[,,i] = t(rbind(glmnet_fit$a0[i,],(as.matrix(beta_lam[[i]]))))
  }
  beta_lam = tmp
  # Now compute proportion of variance explained
  rho2 = matrix(NA,nrow=N.samples,ncol=1)
  rho2_lam = matrix(NA,nrow=N.samples,ncol=L)
  for (ii in 1:N.samples){
    i = ii+warmup
    XB = F_sampler$samples[i,,]
    XB2 = sum((XB)^2)
    
    # Random effect trace
    post_trace = m*sum(Z_hypers$gamma[i,]^2) + n*m*Z_hypers$sigma_sq[i]
    
    # Zeroth-version
    rho2[ii] = XB2 / (XB2 + post_trace)
    
    # lam version
    rho2_lam[ii,] = XB2 / (XB2 + post_trace + apply(beta_lam,1,function(b) sum((XB-(X_sub0%*%b))^2)))
  }
  # Plot in terms of model size
  # And for model size
  model_size = apply(beta_lam,1,function(b) sum(rowSums(b) != 0))
  # Replot with model size as xlab
  unq_size = unique(model_size)
  LL = length(unq_size)
  # Will collect some thigns here
  rho2_lam_quantiles = matrix(NA,ncol=LL,nrow=3)
  if (plot){
    plot(NA,ylim=c(0,1),xlim=c(0,LL-1),xlab="Model Size")
    abline(h=mean(rho2))
    abline(h=quantile(rho2,probs=c(0.05,0.95)))
  }
  for (l in 1:LL){
    ind_max = max(which(model_size == unq_size[l]))
    rr = quantile(rho2_lam[,ind_max],probs=c(0.05,0.95))
    rho2_lam_quantiles[c(1,2),l] = rr
    rho2_lam_quantiles[3,l] = ind_max
    if (plot){
      points(unq_size[l]-1,mean(rho2_lam[,ind_max]))
      segments(unq_size[l]-1,rr[1],unq_size[l]-1,rr[2])
    }
  }
  n_selected = min(which(rho2_lam_quantiles[2,]>mean(rho2))) - 1 # Minus intercept
  # Which ones selected?
  opt_lam = rho2_lam_quantiles[3,n_selected+1]
  selected = sort(id_j[which(rowSums(beta_lam[opt_lam,-1,])!=0)])
  print(paste0("DSS selected ", n_selected+1, " variables (including intercept)"))
  print("Selected coefficients:")
  print(selected)
}

