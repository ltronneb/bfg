##### Simulation study for the BFG paper
library(fosr)
library(bfg)
library(foreach)
library(doParallel)
library(doRNG)
# Set number of threads to be friendly
#RcppParallel::setThreadOptions(numThreads = 4)
cores <- 20
cl <- makeCluster(cores,outfile="out.txt") # Create a cluster object, using one less than total cores
registerDoParallel(cl) # Register the parallel backend
# Set up some parameters shared for all settings
N.simulations = 100
n = 100
m = 30
ell0 = 0.1
p0 = 10
RSNR = 5
rho = 0.75

rz2_list = list(0,0.1,0.5)
p_list = list(200,500,1000)

p = 1000
RZ2 = 0.1

set.seed(42)

for (RZ2 in rz2_list){
  for (p in p_list){
    foreach(sim = 1:N.simulations, .packages = c('bfg','fosr','glmnet','hdf5r'),.options.RNG = 42) %dorng% {
      # Set seed here based on current settings
      file_name_settings = paste0("n=",n,",m=",m,",ell0=",ell0,",p0=",p0,
                                  ",RSNR=",RSNR,",RZ2=",RZ2,",rho=",rho,",p=",p,",sim=",sim)
      # Generate data
      data_generated = gen_data(n,m,p,p0,RSNR,ell0,rho,re = T,re_prop = RZ2)
      Y = data_generated$Y
      X = data_generated$X
      t = data_generated$T
      true = which(apply(data_generated$B,1,function(sx) any(x != 0)))
      tryCatch({
        # Run bfg
        fit = bfg(Y,X,t,p0=floor(0.2*p),data_generated=data_generated,
                  interactions = F, plotting = T,thinning = 1, N_iter=2000, verbose=T,
                  compute_betas = F, beta_gamma_a = 1, beta_gamma_b = 100, slab_scale = 0.5,
                  temp_schedule = c(seq(0,1,length.out=200)^2,rep(1,2000-200))
                  # temp_schedule = c(rep(0,100),rep(1,2000-100))
                  )
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
        
        # Compute
        # Store results
        fit$rmse = rmse
        fit$mciw = mciw
        fit$coverage = coverage
        fit$selected = id_j$selected
        fit$true = true
        fit$FPR_TPR = FPR_TPR
        # And I'll also store this file
        save(fit, file = paste0("results/bfg_",file_name_settings))
        # And do some cleaning
        file.remove(fit$h5file)
        rm(list=c("fit","beta.hat","id_j"))
        gc()
      },error = function(msg){
        print(msg)
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
        # Store results
        out$rmse = rmse
        out$mciw = mciw
        out$coverage = coverage
        out$selected = pos_select_dss
        out$true = true
        out$data_generated = data_generated
        out$FPR_TPR = FPR_TPR
        # And I'll also store this file
        save(out, file = paste0("results/fosr_",file_name_settings))
      }, error = function(msg){
        print(msg)
      })
    }
  }
  
}
stopCluster(cl)
