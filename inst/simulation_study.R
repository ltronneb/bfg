##### Simulation study for the BFG paper
library(fosr)
library(bfg)
# Set number of threads to be friendly
RcppParallel::setThreadOptions(numThreads = 4)
# Set up some parameters shared for all settings
N.simulations = 100
n = 100
m = 30
ell0 = 0.1
p0 = 10
RSNR = 5
RZ2 = 0.1
rho = 0.75

p_list = list(200,500,1000)

for (p in p_list){
  for (sim in 1:N.simulations){
    # Set seed here based on current settings
    set.seed(as.numeric(gsub("[.]","",paste0(n,m,ell0,p0,RSNR,RZ2,rho,p,sim)))/1e18)
    file_name_settings = paste0("n=",n,",m=",m,",ell0=",ell0,",p0=",p0,
                                ",RSNR=",RSNR,",RZ2=",RZ2,",rho=",rho,",p=",p,",sim=",sim)
    # Generate data
    data_generated = gen_data(n,m,p,p0,RSNR,ell0,rho,re = T,re_prop = RZ2)
    Y = data_generated$Y
    X = data_generated$X
    t = data_generated$T
    true = which(apply(data_generated$B,1,function(x) any(x != 0)))
    tryCatch({
      # Run bfg
      fit = bfg(Y,X,t,p0=p0,data_generated=data_generated,
                interactions = F, plotting = F,thinning = 1, N_iter=2000, verbose=F,
                compute_betas = T)
      # Pull out betas
      beta.hat = fit$beta.hat[1001:2000,,]
      beta.mean = apply(beta.hat,c(2,3),mean)
      beta.quantiles = apply(beta.hat,c(2,3),quantile, probs=c(0.025,0.975))
      # Compute metrics
      rmse = sqrt(mean((beta.mean - data_generated$B)^2))
      mciw = mean(beta.quantiles[2,,] - beta.quantiles[1,,])
      coverage = mean((beta.quantiles[1,,] <= data_generated$B) & (data_generated$B <= beta.quantiles[2,,]))
      # Select coefs
      id_j = select_betas(fit,max_model_size=50,N_samples=1000,N_samples_weights=1000,plot=F)
      # Compute FPR and TPR for ROC curves
      FPR_TPR = matrix(NA,ncol=50,nrow=2)
      for (i in 1:length(id_j$models)){
        indices = id_j$models[[i]]
        model_size = length(indices)
        FPR_TPR[1,model_size] = length(intersect(true,indices)) / length(true)
        FPR_TPR[2,model_size] = length(setdiff(indices,true)) / (p-length(true))
      }
      plot(FPR_TPR[2,],FPR_TPR[1,],type="l",ylim=c(0,1),xlim=c(0,1))
      
      # Compute 
      # Store results
      fit$rmse = rmse
      fit$mciw = mciw
      fit$coverage = coverage
      fit$selected = id_j$selected
      fit$true = true
      fit$FPR_TPR = FPR_TPR
      # And I'll also store this file
      save(fit, file = paste0("bfg_",file_name_settings))
    },error = function(msg){
      return(NA)
    })
    
    
    # Now do the same thing for the fosr model
    tryCatch({
      out = fosr(
        Y = data_generated$Y,
        tau = data_generated$T,
        X = data_generated$X,
        K = 15,
        mcmc_params = list("fk", "alpha","gamma", "Yhat", "sigma_e", "sigma_g"))
      alpha = array(0,dim=c(dim(out$alpha)[1],p,m))
      for (k in 1:p){
        alpha[,k,] = get_post_alpha_tilde(out$fk,out$alpha[,1+k,])
      }
      alpha.mean = apply(alpha,c(2,3),mean)
      alpha.quantiles = apply(alpha,c(2,3),quantile, probs=c(0.025,0.975))
      rmse = sqrt(mean((alpha.mean - data_generated$B)^2))
      mciw = mean(alpha.quantiles[2,,] - alpha.quantiles[1,,])
      coverage = mean((alpha.quantiles[1,,] <= data_generated$B) & (data_generated$B <= alpha.quantiles[2,,]))
      # Select coefs
      alpha_dss = fosr_select(
        X = cbind(1,data_generated$X),
        post_alpha = out$alpha,
        post_trace_sigma_2 = n*m*out$sigma_e^2 + apply(out$sigma_g^2, 1, sum),
        weighted = TRUE,
        alpha_level = 0.10,
        remove_int = T,
        include_plot = F,
        include_model_list = T)
      pos_select_dss = which(apply(alpha_dss$alpha_dss, 1, function(x) any(x != 0)))[-1] - 1 # no intercept
      # Compute FPR and TPR for ROC curves
      FPR_TPR = matrix(NA,ncol=50,nrow=2)
      for (i in 1:nrow(alpha_dss$model_list)){
        indices = which(alpha_dss$model_list[i,-1])
        model_size = length(indices)
        FPR_TPR[1,model_size] = length(intersect(true,indices)) / length(true)
        FPR_TPR[2,model_size] = length(setdiff(indices,true)) / (p-length(true))
      }
      lines(FPR_TPR[2,],FPR_TPR[1,],col="red")
      # Store results
      out$rmse = rmse
      out$mciw = mciw
      out$coverage = coverage
      out$selected = pos_select_dss
      out$true = true
      out$FPR_TPR = FPR_TPR
      # And I'll also store this file
      save(out, file = paste0("fosr_",file_name_settings))
    }, error = function(msg){
      return(NA)
    })
  }
}


