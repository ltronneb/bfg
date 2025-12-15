#' Generate simulated data
#' 
#' This function generates simulated data for function-on-scalar regression
#' 
#' 
#'@param n : number of observations
#'@param m : number of timepoints
#'@param p : number of covariates
#'@param p0 : number of true non-zero covariates
#'@param RSNR : root-signal-to-noise-ratio
#'@param ell : length-scale controlling smoothness in time
#'@param rho : correlation between covariates x_i, x_j
#'@param re : bool (simulate with random effects, default=T)
#'
#'@export
gen_data = function(n,m,p,p0,RSNR,ell,rho,re=T){
  # Function to generate a sample dataset for simulation study
  # n: number of parameters
  # m: number of unique input locations (say time)
  # p: number of coefficients
  # p0: number of true non-zero coefficients
  # RSNR: Root signal to noise ratio, for sigma
  # ell: double, lengthscale of the underlying GP over T
  # rho: covariance between covariates in X
  # re: bool, generate random effects?
  # Generate inputs
  t.seq = seq(0,1,length.out=m)
  X = generate_X(n,p,rho)
  
  # Generate coefficients
  B = matrix(0,ncol=m,nrow=p)
  idx_b = floor(seq(1,p,length.out=p0)) # evenly spaced
  Kt = kern_t(t.seq,t.seq,ell)
  Lt = t(chol(Kt+1e-9*diag(m)))
  B0 = Lt%*%rnorm(m)
  for (k in idx_b){
    B[k,] = Lt%*%rnorm(m)
  }
  
  # Generate random effects
  Z = matrix(0,ncol=m,nrow=n)
  if (re){ # Generate random effects
    for (i in 1:n){
      Z[i,] = Lt%*%rnorm(m) 
    }
  }
  
  # Generate true functions per individual
  F.true = cbind(1,X)%*%rbind(t(B0),B) + Z
  
  # Noise levels set according to RSNR
  s = sqrt(sum((F.true-mean(F.true))^2)/(n*m-1))/RSNR
  print(s)
  
  # Generate data
  Y = F.true + rnorm(n*m,sd=s)
  
  # Convert t to list
  tvec = as.matrix(t.seq)
  t_list <- lapply(1:nrow(tvec), function(i) matrix(tvec[i, ], ncol = 1))

  
  return(list(N=n,M=m,p=p,Y=Y,X=X,T=tvec,t_list=t_list,
              tau0_prime = m*(p0/p)*(1/sqrt(n)),B=B,B0 = B0,Z=Z,sigma=s,
              F.true = F.true))
  
}

#' Internal helper functions for kernel over t
#' 
#' @keywords internal
#' @noRd
kern_t = function(x1,x2,ell){
  # This implemenets the Matern3/2 kernel
  x1 = as.matrix(x1)
  x2 = as.matrix(x2)
  n1 = nrow(x1)
  n2 = nrow(x2)
  K = matrix(NA,nrow=n1,ncol=n2)
  for (i in 1:n1){
    for (j in 1:n2){
      K[i,j] = exp(-0.5*abs(x1[i,]-x2[j,])^2/ell^2)
    }
  }
  return(K)
}
#' Internal helper functions for kernel over x
#' 
#' @keywords internal
#' @noRd
kern_x = function(x1,x2,alpha,c,gamma){
  # Implements linear kernel with inner product wrt diag(alpha)
  # plus random effects
  x1 = as.matrix(x1)
  x2 = as.matrix(x2)
  K = c^2 + x1%*%(t(x2)*alpha^2)
  if (!is.null(gamma)){
    K = K + diag(gamma^2) 
  }
  return(K)
}
#' Internal helper functions to generate correlated design matrix
#' 
#' @keywords internal
#' @noRd
#' @importFrom  MASS mvrnorm
generate_X = function(n,p,rho){
  S = outer(1:p,1:p,function(i,j) rho^abs(i-j))
  X = mvrnorm(n,mu=rep(0,p),Sigma=S)
  return(scale(X))
}

