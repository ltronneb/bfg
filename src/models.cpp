// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(StanHeaders)]]
#include <stan/math.hpp>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

// One thread-local buffer per thread, reused across calls
thread_local std::vector<double> thread_local_buffer;


//////////////////////////////////////////////////////////////////////////////
///////////////////////////    Cache    //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Cache!
struct XCache {
  bool initialized = false;
  int n, p;
  std::vector<double> deriv_data;
  Rcpp::NumericMatrix X_rcpp;
  Rcpp::NumericMatrix X_sq_rcpp;
  // Eigen::MatrixXd X_aug; // This doesn't work!
  
  void initialize(const Eigen::MatrixXd& X){
    n = X.rows();
    p = X.cols();
    Eigen::MatrixXd X_sq = X.array().square();
    X_rcpp = Rcpp::NumericMatrix(n,p);
    X_sq_rcpp = Rcpp::NumericMatrix(n,p);
    std::copy(X.data(), X.data() + X.size(), X_rcpp.begin());
    std::copy(X_sq.data(), X_sq.data() + X_sq.size(), X_sq_rcpp.begin());
    // Matrix needed for low-rank stuff
    // X_aug.resize(n,p+1);
    // X_aug.col(0).setOnes();
    // X_aug.block(0,1,n,p) = X;
    initialized = true;
  }
  
  void clear() {
    n = 0;
    p = 0;
    initialized = false;
  }
};



XCache& get_cache() {
  static XCache cache;
  return cache;
}

// [[Rcpp::export]]
void clear_cache() {
  get_cache().clear();
}

// [[Rcpp::export]]
void prepare_cache(Rcpp::NumericMatrix X) {
  using Rcpp::as;
  Eigen::MatrixXd X_eigen = as<Eigen::MatrixXd>(X);
  get_cache().initialize(X_eigen);
}

//////////////////////////////////////////////////////////////////////////////
///////////////////////////  Fast XLXT  //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Worker and parallel stuff!
struct XLXtWorker : public RcppParallel::Worker {
  const RcppParallel::RMatrix<double> X;
  const double* __restrict__ lambda_ptr;
  int n, p;
  
  std::vector<double> partial_data;
  RcppParallel::RMatrix<double> partial;
  
  XLXtWorker(const Rcpp::NumericMatrix& X_, 
             const double* lambda_ptr_,
             std::vector<double>& shared_buffer)
    : X(X_), 
      lambda_ptr(lambda_ptr_), 
      n(X_.nrow()), 
      p(X_.ncol()),
      partial_data(shared_buffer),
      partial(partial_data.data(), n, n) {}
  
  XLXtWorker(const XLXtWorker& other, RcppParallel::Split)
    : X(other.X), lambda_ptr(other.lambda_ptr), n(other.n), p(other.p),
      partial_data(n * n, 0.0),
      partial(partial_data.data(), n, n) {}
  
  
  inline void operator()(std::size_t begin, std::size_t end) {
    const int block = 64;  // tweak for your CPU cache size
    const double* __restrict__ lam = lambda_ptr;
    const double* __restrict__ Xptr = X.begin();
    double* __restrict__ pdata = partial_data.data();
    for (int kk = begin; kk < end; kk += block) {
      int kmax = std::min<int>(kk + block, end);
      for (int i = 0; i < n; ++i){
        for (int j = 0; j <= i; ++j){
          double sum = 0.0;
          
#pragma GCC ivdep
          for (size_t k = kk; k < kmax; ++k){
            sum += lam[k] * Xptr[i + k*n] * Xptr[j + k*n];
          }
          size_t idx = i + j * n;
          pdata[idx] += sum;
          if (i != j) pdata[j + i*n] += sum;
        }
      }
    }
  }
  
  void join(const XLXtWorker& rhs) {
    for (int i = 0; i < n * n; ++i)
      partial_data[i] += rhs.partial_data[i];
  }
};


Eigen::MatrixXd compute_XLambdaXt_core(const Rcpp::NumericMatrix& X,
                                       const Eigen::VectorXd& lambda_eigen) {
  
  int n = X.nrow();
  std::size_t p = X.ncol();
  std::size_t grain = std::max<std::size_t>(1, p / 100);
  size_t required_size = static_cast<size_t>(n) * n;
  // Reuse buffer
  if (thread_local_buffer.size() != required_size)
    thread_local_buffer.resize(required_size);
  std::fill(thread_local_buffer.begin(), thread_local_buffer.end(), 0.0);
  // Setting up worker and parallel reduce
  XLXtWorker worker(X, lambda_eigen.data(), thread_local_buffer);
  parallelReduce(0, X.ncol(), worker, grain); // grain set as proportion of p
  // Convert result back
  Eigen::Map<Eigen::MatrixXd> result(worker.partial_data.data(), X.nrow(), X.nrow());
  return result;
}


//////////////////////////////////////////////////////////////////////////////
/////////////////////////// HMC KERNEL ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
struct StanNestedScope {
  StanNestedScope() { stan::math::start_nested(); }
  ~StanNestedScope() { stan::math::recover_memory_nested(); }
};
template <typename PotentialFunc>
Rcpp::List HMC_kernel(PotentialFunc U,
                      const Eigen::VectorXd& q_val, 
                      const Eigen::VectorXd& mass_matrix_diag, 
                      double epsilon, int L){
  
  using stan::math::var;
  using Eigen::VectorXd;
  using Eigen::Matrix;
  stan::math::set_zero_all_adjoints();
  // Parameter vector
  VectorXd q_current = q_val;
  VectorXd q = q_current;
  // Momentum vector
  VectorXd eta = Rcpp::as<VectorXd>(Rcpp::rnorm(q.size(),0.0,1.0));
  VectorXd p = mass_matrix_diag.array().sqrt() * eta.array();
  VectorXd p_current = p;
  
  // Half step momentum
  {
    StanNestedScope scope;
    Matrix<var, -1, 1> q_var = q.cast<var>();
    var U_val = U(q_var,1);
    std::vector<var> q_vars(q_var.data(), q_var.data() + q_var.size());
    std::vector<double> grad_U_vec;
    U_val.grad(q_vars,grad_U_vec);
    VectorXd grad_U = Eigen::Map<VectorXd>(grad_U_vec.data(),grad_U_vec.size());
    p -= 0.5*epsilon*grad_U;
  }
  
  // Now full steps
  {
    for (int i = 0; i < L; ++i){
      q = q + epsilon*(p.array() / mass_matrix_diag.array()).matrix();
      if (i != L-1){
        {
          StanNestedScope scope;
          
          Matrix<var, -1, 1> q_var = q.cast<var>();
          var U_val = U(q_var,0);
          std::vector<var> q_vars(q_var.data(), q_var.data() + q_var.size());
          std::vector<double> grad_U_vec;
          U_val.grad(q_vars,grad_U_vec);
          VectorXd grad_U = Eigen::Map<VectorXd>(grad_U_vec.data(), grad_U_vec.size());
          p -= epsilon * grad_U;
        }
        
        
        
      }
    }
  }
  
  // Final half step
  {
    StanNestedScope scope;
    Matrix<var, -1, 1> q_var = q.cast<var>();
    var U_val = U(q_var,0);
    std::vector<var> q_vars(q_var.data(), q_var.data() + q_var.size());
    std::vector<double> grad_U_vec;
    U_val.grad(q_vars, grad_U_vec);
    VectorXd grad_U = Eigen::Map<VectorXd>(grad_U_vec.data(), grad_U_vec.size());
    p -= 0.5 * epsilon * grad_U;
    
  }
  
  // Swap sign on p 
  p = -p;
  
  VectorXd p_current_d = p_current.val();
  VectorXd p_d = p.val();
  
  // Evaluate energies
  double current_U;
  double proposed_U;
  {
    StanNestedScope scope;
    Matrix<var, -1, 1> q_current_var = q_current.cast<var>();
    Matrix<var, -1, 1> q_var = q.cast<var>();
    current_U =  U(q_current_var,0).val();
    proposed_U = U(q_var,0).val();
  }
  // Kinetic
  double current_K = 0.5 * (p_current_d.array().square() / mass_matrix_diag.array()).sum();
  double proposed_K = 0.5 * (p_d.array().square() / mass_matrix_diag.array()).sum();
  
  // // Acceptance probability
  double log_accept_prob = current_U - proposed_U + current_K - proposed_K;
  double accept_prob = std::exp(std::min(0.0,log_accept_prob));
  
  Eigen::VectorXd q_return;
  if (Rcpp::runif(1, 0.0, 1.0)[0] < accept_prob)
    q_return = q.val();
  else
    q_return = q_current.val();
  
  // Return objects I care about
  if (stan::math::empty_nested()){
    stan::math::recover_memory();
  }
  
  return Rcpp::List::create(
    Rcpp::_["theta"] = q_return,
    Rcpp::_["accept_prob"] = accept_prob);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////// CUSTOM LLT solver and log-det////////////////////////////
//////////////////////////////////////////////////////////////////////////////
struct LLTBase {
  virtual ~LLTBase() = default;
  virtual Eigen::MatrixXd solve(const Eigen::MatrixXd& Y) const = 0;
  virtual double log_det(int n) const = 0;
};

struct FullLLT : LLTBase {
  Eigen::LLT<Eigen::MatrixXd> llt;
  
  explicit FullLLT(const Eigen::MatrixXd& K) : llt(K) {
    if (llt.info() != Eigen::Success)
      throw std::runtime_error("FullLLT: LLT failed!");
  }
  
  Eigen::MatrixXd solve(const Eigen::MatrixXd& Y) const override {
    return llt.solve(Y);
  }
  
  double log_det(int /*n*/) const override {
    Eigen::VectorXd diagL = llt.matrixL().toDenseMatrix().diagonal();
    return 2.0 * diagL.array().log().sum();
  }
};

struct LowRankLLT : LLTBase {
  Eigen::MatrixXd X_full;
  Eigen::VectorXd diag_full;
  double eps;
  double threshold;
  int r;
  
  Eigen::MatrixXd X_active;
  Eigen::VectorXd diag_active;
  std::vector<int> active_idx;
  Eigen::LLT<Eigen::MatrixXd> lltM;
  
  // Constructor
  explicit LowRankLLT(const Eigen::MatrixXd& X_in,
                      const Eigen::VectorXd& diag,
                      double c_sq,
                      double eps_in = 1e-6,
                      double threshold_in = 1e-4)
    : X_full(X_in), eps(eps_in), threshold(threshold_in)
  {
    r = X_full.cols(); // should be p + 1
    if (diag.size() != r-1){
      throw std::invalid_argument("diag length must equal X.cols()-1");
    }
    if (eps <= 0.0){
      throw std::invalid_argument("eps must be >0");
    }
    
    // Prepend c^2 to diag
    diag_full.resize(r);
    diag_full(0) = c_sq;
    diag_full.tail(r-1) = diag;
    
    // Compute active set once
    compute_active_set();
    // initialise cholesky using current diag
    update_cholesky(diag, c_sq);
  }
  
  void compute_active_set() {
    // select active columns
    active_idx.clear();
    for (int j = 0; j < diag_full.size(); ++j){
      if (diag_full(j) > threshold) active_idx.push_back(j);
    }
    
    int r_active = active_idx.size();
    X_active.resize(X_full.rows(),r_active);
    diag_active.resize(r_active);
    
    for (int i = 0; i < r_active; ++i){
      X_active.col(i) = X_full.col(active_idx[i]);
      diag_active(i) = diag_full(active_idx[i]);
    }
  }
  
  void update_cholesky(const Eigen::VectorXd& diag, double c_sq) {
    // Update diag_full
    diag_full(0) = c_sq;
    diag_full.tail(r-1) = diag;
    // create small matrix D^{-1}
    for (int i = 0; i < active_idx.size(); ++i){
      diag_active(i) = diag_full(active_idx[i]);
    }
    
    Eigen::MatrixXd M = (X_active.transpose() * X_active) / eps;
    M.diagonal().array() += (1.0 / diag_active.array());
    M.diagonal().array() += 1e-4;
    
    // factorise this small matrix 
    lltM.compute(M);
    if (lltM.info() != Eigen::Success)
      throw std::runtime_error("LowRankLLT: LLT failed!");
  }
  
  
  
  // Now the actual Woodbury solve
  Eigen::MatrixXd solve(const Eigen::MatrixXd& Y) const override {
    Eigen::MatrixXd XtY = X_active.transpose() * Y;
    Eigen::MatrixXd tmp = lltM.solve(XtY);
    return (Y / eps) - (X_active * tmp) / (eps * eps);
  }
  
  double log_det(int n) const override {
    Eigen::VectorXd diagL = lltM.matrixL().toDenseMatrix().diagonal();
    double logdetM = 2.0 * diagL.array().log().sum();
    double logdetDinv = diag_active.array().log().sum();
    return double(n) * std::log(eps) + logdetM + logdetDinv;
  }
};



//////////////////////////////////////////////////////////////////////////////
//////////////////// CUSTOM LOG-LIKELIHOODS //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Now here comes the actual definition of the function and the gradient
stan::math::var gp_1dloglik_analyticgrad(const Eigen::MatrixXd& X,
                                         const Eigen::MatrixXd& F,
                                         const Eigen::MatrixXd& Qt, // Eigen-decomp of K_t
                                         const Eigen::VectorXd& Dt, // Eigen-decomp of K_t
                                         const Eigen::VectorXd& gamma,
                                         const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& theta,
                                         bool first_step){
  // I need to compute f^T (K_t \kron K_x)^{-1}f
  // Efficiency is provided by Kronecker product, but to avoid a large AD tape
  // everything is done with doubles, and analytic gradient added at the end
  // theta has the form (\tau, \lambda_1,...,\lambda_p,c)
  
  // Define some constants
  int n = X.rows();
  int p = X.cols();
  int m = Qt.rows();
  
  // First thing I pull out the values of theta
  Eigen::VectorXd theta_val = theta.val(); 
  // And pull out the things I care about
  double tau = theta_val(0);
  Eigen::VectorXd lambda = theta_val.segment(1,p);
  double c = theta_val(p + 1);
  // Define the diagonal term in XLXT
  Eigen::VectorXd diag = (tau * lambda).array().square();
  
  // Kernel construction
  // First init cache
  const auto& cache = get_cache();
  if (!cache.initialized) Rcpp::stop("Cache not initialized.");
  // Compute kernel matrix
  // ------------ ALL BELOW UNCOMMENT FOR NORMAL VERSION---------------
  // ------------------------------------------------------------------
  // Ill just compare my solves to a full just to check
  Eigen::MatrixXd K_val = compute_XLambdaXt_core(cache.X_rcpp, diag);
  K_val.array() += c * c;
  K_val.diagonal().array() += 1e-6;
  // Eigen::LLT<Eigen::MatrixXd> llt_full(K_val);
  
  
  // Logic for low-rank adaptation -- turned off for now
  static std::unique_ptr<LLTBase> llt; // This is static, so persists
  llt = std::make_unique<FullLLT>(K_val);
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  
  
  // -------- ALL BELOW UNCOMMENT FOR LOWRANK ADAPTIVE VERSION---------
  // ------------------------------------------------------------------
  // // Commented this out, low-rank adaptation didn't play nice with HMC 
  // static std::unique_ptr<LLTBase> llt; // This is static, so persists
  // if (first_step){ // Then we init
  //   int r_active = (diag.array() > 1e-4).count() + 1;
  // 
  //   if (n < r_active){ // Construct matrix and take a full Cholesky
  //     Rcpp::Rcout << "full_rank:" << r_active << std::endl;
  //     Eigen::MatrixXd K_val = compute_XLambdaXt_core(cache.X_rcpp, diag);
  //     K_val.array() += c * c;
  //     K_val.diagonal().array() += 1e-6;
  //     llt = std::make_unique<FullLLT>(K_val);
  //   } else{ // Work with the low-rank version
  //     Rcpp::Rcout << "low_rank:" << r_active << std::endl;
  //     llt = std::make_unique<LowRankLLT>(cache.X_aug, diag, c*c);
  //   }
  // } else { // If it is not the first step, we deal with the two cases
  //   auto* lowrank = dynamic_cast<LowRankLLT*>(llt.get());
  //   if (lowrank){ // Cholesky is updated
  //     lowrank->update_cholesky(diag, c*c);
  //   } else { // Reinit the full thing
  //     Eigen::MatrixXd K_val = compute_XLambdaXt_core(cache.X_rcpp, diag);
  //     K_val.array() += c * c;
  //     K_val.diagonal().array() += 1e-6;
  //     llt = std::make_unique<FullLLT>(K_val);
  //   }
  // }
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  
  // Eigen::LLT<Eigen::MatrixXd> llt(K_val);
  ////////////////////////////////////////////////
  // QUADRATIC SOLVE
  ////////////////////////////////////////////////
  // First compute a helper quantity we will use throughout
  Eigen::MatrixXd KFU = llt->solve(F * Qt);
  double quad_solve = (((F * Qt).transpose() * KFU).diagonal().array() / Dt.array()).sum();
  
  //GRADIENTS
  
  // Now computing the quad_solve and log_det gradients iteratively
  Eigen::VectorXd quad_solve_grad(p + 2); // Will store them here
  Eigen::VectorXd log_det_grad(p + 2);
  
  // wrt lambdas (while creating the necessary quantity for tau as well)
  double quad_solve_grad_tau = 0.0;
  double log_det_grad_tau = 0.0;
  for (int j = 0; j < p; ++j){
    Eigen::VectorXd xj = X.col(j);
    Eigen::VectorXd KFUT_xj = KFU.transpose() * xj;
    double scale = 2.0 * tau * tau * lambda(j);
    quad_solve_grad(1 + j) = -scale * (KFUT_xj.array().square() / Dt.array()).sum();
    Eigen::MatrixXd xj_solve_mat = llt->solve(xj);
    Eigen::VectorXd xj_solve = xj_solve_mat.col(0);
    log_det_grad(1 + j) = double(m) * scale * xj.dot(xj_solve);
    // Computing for tau
    quad_solve_grad_tau -= 2.0 * tau * lambda(j) * lambda(j) * (KFUT_xj.array().square() / Dt.array()).sum();
    log_det_grad_tau += double(m) * 2.0 * tau *lambda(j) * lambda(j) * xj.dot(xj_solve);
  }
  
  // wrt tau
  {
    quad_solve_grad(0) = quad_solve_grad_tau;
    log_det_grad(0) = log_det_grad_tau;
  }
  
  
  // wrt c
  {
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd KFUT_ones = KFU.transpose() * ones;
    quad_solve_grad(p + 1) = -2.0 * c * (KFUT_ones.array().square() / Dt.array()).sum();
    Eigen::MatrixXd ones_solve_mat = llt->solve(ones);
    Eigen::VectorXd ones_solve = ones_solve_mat.col(0);
    log_det_grad(p + 1) = double(m) * 2.0 * c * ones.dot(ones_solve);
  }
  
  stan::math::var quad_solve_var = stan::math::precomputed_gradients(quad_solve,theta,quad_solve_grad);
  
  ////////////////////////////////////////////////
  // LOG-DETERMINANT
  ////////////////////////////////////////////////
  // K_f log determinant from Cholesky direct
  double log_det_Kf = llt->log_det(n);
  // K_t log-determinant via eigendecomp and so
  double log_det_Kt = Dt.array().log().sum();
  // And the value of the log determinant of a Kronecker product now
  double log_det_val = (double(m) * log_det_Kf + double(n) * log_det_Kt);
  
  stan::math::var log_det_var = stan::math::precomputed_gradients(log_det_val,theta,log_det_grad);
  
  return -0.5*(quad_solve_var + log_det_var);
}


// And similarly for the SKIM-kernel
stan::math::var gp_1dloglik_analyticgrad_SKIM(const Eigen::MatrixXd& X,
                                              const Eigen::MatrixXd& F,
                                              const Eigen::MatrixXd& Qt, // Eigen-decomp of K_t
                                              const Eigen::VectorXd& Dt, // Eigen-decomp of K_t
                                              const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& theta,
                                              bool first_step){
  
  // Define some constants
  int n = X.rows();
  int p = X.cols();
  int m = Qt.rows();
  
  // First thing I pull out the values of theta
  Eigen::VectorXd theta_val = theta.val(); 
  // And pull out the things I care about
  double tau1 = theta_val(0);
  double tau2 = theta_val(1);
  Eigen::VectorXd lambda = theta_val.segment(2,p);
  double c = theta_val(p + 2);
  // Define the diagonal term in XLXT
  Eigen::VectorXd diag = lambda.array().square();
  
  // Kernel construction
  // First init cache
  const auto& cache = get_cache();
  if (!cache.initialized) Rcpp::stop("Cache not initialised.");
  // Compute kernel matrix
  Eigen::MatrixXd xlxt = compute_XLambdaXt_core(cache.X_rcpp,diag);
  Eigen::MatrixXd x2lx2t = compute_XLambdaXt_core(cache.X_sq_rcpp,diag);
  Eigen::MatrixXd xlxt_poly = (xlxt.array() + 1.0).array().square().matrix();
  Eigen::MatrixXd K_val = (0.5 * tau2 * tau2 * xlxt_poly).array() - 
    (0.5 * tau2 * tau2)*x2lx2t.array() + (tau1 * tau1 - tau2 * tau2)*xlxt.array();
  K_val.array() += (c * c) - (0.5 * tau2 * tau2);
  K_val.diagonal().array() += 1e-6; // little jitter
  // Logic for low-rank adaptation -- turned off for now
  static std::unique_ptr<LLTBase> llt; // This is static, so persists
  llt = std::make_unique<FullLLT>(K_val);
  
  ////////////////////////////////////////////////
  // QUADRATIC SOLVE AND LOG-DET
  ////////////////////////////////////////////////
  // First compute a helper function
  Eigen::MatrixXd KFU = llt->solve(F*Qt);
  double quad_solve = (((F * Qt).transpose() * KFU).diagonal().array() / Dt.array()).sum();
  double log_det_Kf = llt->log_det(n);
  double log_det_Kt = Dt.array().log().sum();
  double log_det_val = (double(m) * log_det_Kf + double(n) * log_det_Kt);
  
  // Now gradients
  Eigen::VectorXd quad_solve_grad(p + 3); // Will store them here
  Eigen::VectorXd log_det_grad(p + 3);
  
  
  // Also here some helper quantities used multiple times
  Eigen::MatrixXd KPOLY = llt->solve((xlxt.array() + 1.0).matrix());
  Eigen::MatrixXd KPOLY2 = llt->solve(xlxt_poly); 
  double log_det_grad_tau1 = 0.0; // these will be built iteratively
  double log_det_grad_tau2 = 0.0; // these will be built iteratively
  
  // wrt c
  {
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
    double scale = 2.0*c;
    Eigen::MatrixXd dkdtheta = scale * (ones * ones.transpose());
    Eigen::MatrixXd ones_solve_mat = llt->solve(ones);
    Eigen::VectorXd ones_solve = ones_solve_mat.col(0);
    quad_solve_grad(p + 2) = -((KFU.transpose() * dkdtheta * KFU).diagonal().array() / Dt.array()).sum();
    log_det_grad(p + 2) = double(m) * scale * ones.dot(ones_solve);
    // build tau gradients iteratively
    log_det_grad_tau2 -= ones.dot(ones_solve);
  }
  
  // wrt lambdas
  for (int j = 0; j < p; ++j){
    Eigen::VectorXd xj = X.col(j);
    Eigen::VectorXd xj_sq = X.col(j).array().square();
    Eigen::MatrixXd xj_outer = 2 * lambda(j) * (xj * xj.transpose());
    Eigen::MatrixXd xj_sq_outer = 2 * lambda(j) * (xj_sq * xj_sq.transpose());
    Eigen::MatrixXd dkdtheta = (tau2 * tau2)*((1.0 + xlxt.array()) * xj_outer.array()).matrix() - (0.5 * tau2 * tau2)*xj_sq_outer + (tau1 * tau1 - tau2 * tau2) * xj_outer;
    quad_solve_grad(2+j) = -((KFU.transpose() * dkdtheta * KFU).diagonal().array() / Dt.array()).sum();
    Eigen::MatrixXd xj_solve_mat = llt->solve(xj);
    Eigen::VectorXd xj_solve = xj_solve_mat.col(0);
    Eigen::MatrixXd xj_sq_solve_mat = llt->solve(xj_sq);
    Eigen::VectorXd xj_sq_solve = xj_sq_solve_mat.col(0);
    double scale = 2.0 * lambda(j) * double(m);
    log_det_grad(2+j) = scale * ((tau2 * tau2)*xj.dot(KPOLY * xj) - (0.5*tau2*tau2)*xj_sq.dot(xj_sq_solve) + (tau1 * tau1 - tau2 * tau2)*xj.dot(xj_solve));
    // build tau gradients iteratively
    log_det_grad_tau1 += 2.0 * tau1 * (lambda(j) * lambda(j)) * xj.dot(xj_solve);
    log_det_grad_tau2 -= 2.0 * (lambda(j) * lambda(j)) * xj.dot(xj_solve) + (lambda(j) * lambda(j))*xj_sq.dot(xj_sq_solve);
  }
  // wrt tau1
  {
    Eigen::MatrixXd dkdtheta = 2.0 * tau1 * xlxt;
    quad_solve_grad(0) = -((KFU.transpose() * dkdtheta * KFU).diagonal().array() / Dt.array()).sum();
    log_det_grad(0) = double(m) * log_det_grad_tau1;
  }
  // wrt tau2
  {
    Eigen::MatrixXd dkdtheta = (tau2 * xlxt_poly) - (tau2 * x2lx2t) - (2 * tau2 *xlxt);
    dkdtheta.array() -= tau2;
    quad_solve_grad(1) = -((KFU.transpose() * dkdtheta * KFU).diagonal().array() / Dt.array()).sum();
    log_det_grad(1) = double(m) * tau2 * (KPOLY2.diagonal().array().sum() + log_det_grad_tau2);
  }
  
  
  // Setting up the var objects
  stan::math::var quad_solve_var = stan::math::precomputed_gradients(quad_solve,theta,quad_solve_grad);
  stan::math::var log_det_var = stan::math::precomputed_gradients(log_det_val,theta,log_det_grad);
  
  return -0.5*(quad_solve_var + log_det_var);
  
}



// A version here for \theta_z and variance
stan::math::var gp_1dloglik_analyticgrad_re_noise(const Eigen::MatrixXd& Y,
                                                   const Eigen::MatrixXd& Qt,
                                                   const Eigen::VectorXd& Dt,
                                                   const Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>& theta,
                                                   const bool first_step){
  // Define some constants
  int n = Y.rows();
  int m = Y.cols();

  // First thing I pull out values I need
  Eigen::VectorXd theta_val = theta.val();
  Eigen::VectorXd gamma_val = theta_val.segment(0,n);
  Eigen::VectorXd gamma_sq_val = gamma_val.array().square();
  double sigma_val = theta_val(n);
  double sigma_sq_val = sigma_val * sigma_val;

  // Helper quantities
  Eigen::MatrixXd Z = Y * Qt;
  Eigen::MatrixXd inv_eigvals = 1.0 / ((gamma_sq_val * Dt.transpose()).array() + sigma_sq_val);
  Eigen::MatrixXd alpha = (inv_eigvals.array() * Z.array()).matrix() * Qt.transpose();
  Eigen::MatrixXd alpha_Kt = alpha * (Qt * Dt.asDiagonal() * Qt.transpose());

  // Quad_solve and log_det values
  double quad_solve = (Y.array() * alpha.array()).sum();
  double log_det = (1.0 / inv_eigvals.array()).log().sum();

  // Gradients following Algorithm 16 of Saatci's thesis
  std::vector<double> quad_solve_grad(n+1);
  std::vector<double> log_det_grad(n+1);

  // For gammas
  for (int j = 0; j < n; ++j){
    Eigen::VectorXd alpha_Kt_j = alpha_Kt.row(j).transpose();
    Eigen::VectorXd alpha_j = alpha.row(j).transpose();
    double scale = 2.0 * gamma_val(j);
    quad_solve_grad[j] = -scale * (alpha_Kt_j.array() * alpha_j.array()).sum();
    log_det_grad[j] = scale * (inv_eigvals.row(j).transpose().array() * Dt.array()).sum();
  }
  // For sigma
  quad_solve_grad[n] = -2.0*sigma_val * (alpha.array().square().sum());
  log_det_grad[n] = 2.0 *sigma_val * inv_eigvals.array().sum();

  // Setting up the var objects
  stan::math::var quad_solve_var = stan::math::precomputed_gradients(quad_solve,theta,quad_solve_grad);
  stan::math::var log_det_var = stan::math::precomputed_gradients(log_det,theta,log_det_grad);

  // Returning
  return -0.5*(quad_solve_var + log_det_var);
}



//////////////////////////////////////////////////////////////////////////////
///////////////////////////// MODELS /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Standard model, no interaction, 1d
template <typename T>
T gp_kron_logpost_1d_horseshoe(// Data
    const Eigen::MatrixXd& X, // Data for linear kernel
    const Eigen::MatrixXd& Qt,
    const Eigen::VectorXd& Dt,
    const Eigen::MatrixXd& F, // The data
    const Eigen::VectorXd& gamma,
    // Hyperparameters
    const double& slab_scale,
    const double& slab_df,
    const int& nu_local,
    const int& nu_global,
    const T& tau0_prime,
    const T& nugget,
    const T& ell, // lengthscale for covariance over t
    // Parameters
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& q, // parameters to sample
    const bool first_step
){
  using stan::math::exp;
  using stan::math::sqrt;
  using stan::math::square;
  using stan::math::elt_divide;
  using stan::math::elt_multiply;
  using stan::math::log_sum_exp;
  using stan::math::log1p_exp;
  using stan::math::eigendecompose_sym;
  using stan::math::add_diag;
  using stan::math::add;
  using stan::math::sum;
  using stan::math::inv_gamma_lpdf;
  using stan::math::std_normal_lpdf;
  using stan::math::student_t_lpdf;
  using stan::math::pow;
  using stan::math::inv_logit;
  using stan::math::log1m;
  
  // Pull out parameters
  T log_tau_aux_1 = q(0);
  T log_tau_aux_2 = q(1);
  int p = X.cols();
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_tilde_aux_1 = q.segment(2, p);
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_tilde_aux_2 = q.segment(2+p,p);
  T log_c = q(2 + 2*p);
  T log_u_aux = q(3 + 2*p);
  
  // Transform parameters into proper scale and make sure bounds hold
  T tau_aux_1 = exp(log_tau_aux_1);
  T tau_aux_2 = exp(log_tau_aux_2);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda_tilde_aux_1 = exp(log_lambda_tilde_aux_1);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda_tilde_aux_2 = exp(log_lambda_tilde_aux_2);
  T c = exp(log_c);
  T u_aux = exp(log_u_aux);
  
  // Defining some hyper-parameters parameters that will be used
  double n = static_cast<double>(X.rows());
  double m = static_cast<double>(Qt.rows());
  double eig_nugget = 1e-9;
  
  
  T log_tau0 = log(tau0_prime) - log_sum_exp(0.5 * Dt.array().log());
  // Constructing higher order parameters to be used
  auto log_u = log(slab_scale) + 0.5*log_u_aux;
  auto log_tau = log_tau_aux_1 + 0.5*log_tau_aux_2 + log_tau0;
  
  auto log_lambda_tilde = add(log_lambda_tilde_aux_1, 0.5*log_lambda_tilde_aux_2);
  
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_num = add(log_u,log_lambda_tilde);
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_den = -0.5*log_sum_exp(2.0*log_u,add(2.0*log_tau,2.0*log_lambda_tilde));
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda = add(log_lambda_num,log_lambda_den);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda = exp(log_lambda);
  auto tau = exp(log_tau);
  
  // Stack all my parameters I care about into a vector
  Eigen::Matrix<T, Eigen::Dynamic, 1> theta(p + 2);
  theta(0) = tau;
  for (int j = 0; j < p; ++j)
    theta(j + 1) = lambda(j);
  theta(p + 1) = c;
  
  // Get the log-likelihood
  auto loglik = gp_1dloglik_analyticgrad(X,F,Qt,Dt,gamma,theta,first_step);
  
  // Add priors for parameters
  // Note Jacobian corrections
  auto log_prior_log_c = std_normal_lpdf(c) + log_c;
  auto log_prior_log_u_aux = inv_gamma_lpdf(u_aux, 0.5*slab_df,0.5*slab_df) + log_u_aux;
  auto log_prior_log_tau_aux_1 = std_normal_lpdf(tau_aux_1) + log_tau_aux_1;
  auto log_prior_log_tau_aux_2 = inv_gamma_lpdf(tau_aux_2, 0.5*nu_global, 0.5*nu_global) + log_tau_aux_2;
  auto log_prior_log_lambda_tilde_aux_1 = std_normal_lpdf(lambda_tilde_aux_1) + sum(log_lambda_tilde_aux_1);
  auto log_prior_log_lambda_tilde_aux_2 = inv_gamma_lpdf(lambda_tilde_aux_2, 0.5*nu_local, 0.5*nu_local) + sum(log_lambda_tilde_aux_2);
  
  // Log posterior returned
  auto logpost = loglik +
    log_prior_log_c +
    log_prior_log_u_aux +
    log_prior_log_tau_aux_1 +
    log_prior_log_tau_aux_2 +
    log_prior_log_lambda_tilde_aux_1 +
    log_prior_log_lambda_tilde_aux_2;
  
  return logpost;
}

// Function to sample hyperparameters of F (minus the lengthscale)
// [[Rcpp::export]]
Rcpp::List sample_f_hypers(Eigen::MatrixXd X,  // Data
                           const Eigen::MatrixXd& Qt,
                           const Eigen::VectorXd& Dt,
                           const Eigen::MatrixXd& F,  // Data
                           const Eigen::VectorXd& gamma,
                           double tau0_prime_in,  // Hypers
                           double nugget_in, // Hypers
                           double ell_in, // Hypers
                           const Eigen::VectorXd& q_val, // Parameters
                           const Eigen::VectorXd& mass_matrix_diag, double epsilon, int L, // Hypers
                           const double& slab_scale = 5.0, // Hypers w/ default values
                           const double& slab_df = 4.0, // Hypers w/ default values
                           const int& nu_local = 1, // Hypers w/ default values
                           const int& nu_global = 1 // Hypers w/ default values
){
  using stan::math::var;
  using Eigen::VectorXd;
  using Eigen::Matrix;
  // Promote some stuff
  var tau0_prime = tau0_prime_in;
  var nugget = nugget_in;
  var ell = ell_in;
  // Set up the potential function lambda
  auto U = [&](const Eigen::Matrix<var, -1, 1> q_val, const bool first_step){
    return -gp_kron_logpost_1d_horseshoe(X, Qt, Dt, F, gamma, //Data
                                         slab_scale, slab_df, nu_local, // Hypers
                                         nu_global, tau0_prime, nugget, ell, // Hypers
                                         q_val, // Parameters
                                         first_step
    );
  };
  
  return HMC_kernel(U, q_val, mass_matrix_diag, epsilon, L);
}

// SKIM model with interactions
template <typename T>
T gp_kron_logpost_1d_SKIM(// Data
    const Eigen::MatrixXd& X, // Data for linear kernel
    const Eigen::MatrixXd& Qt,
    const Eigen::VectorXd& Dt,
    const Eigen::MatrixXd& F, // The data
    // Hyperparameters
    const double& slab_scale,
    const double& slab_df,
    const int& nu_local,
    const int& nu_global,
    const T& tau0_prime,
    const T& nugget,
    const T& ell, // lengthscale for covariance over t
    // Parameters
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& q, // parameters to sample
    const bool first_step
){
  using stan::math::log_sum_exp;
  // Rcpp::Rcout << "is this never called?!!!!!!:" << std::endl;
  
  // Pull out parameters
  T log_tau_aux_1 = q(0);
  T log_tau_aux_2 = q(1);
  int p = X.cols();
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_tilde_aux_1 = q.segment(2,p);
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_tilde_aux_2 = q.segment(2+p,p);
  T log_c = q(2 + 2*p);
  T log_u_aux = q(3 + 2*p);
  T log_v_aux = q(4 + 2*p);
  // Transform parameters into proper scale and make sure bounds hold
  T tau_aux_1 = exp(log_tau_aux_1);
  T tau_aux_2 = exp(log_tau_aux_2);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda_tilde_aux_1 = exp(log_lambda_tilde_aux_1);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda_tilde_aux_2 = exp(log_lambda_tilde_aux_2);
  T c = exp(log_c);
  T u_aux = exp(log_u_aux);
  T v_aux = exp(log_v_aux);
  
  // Construct higher-order parameters
  T log_tau0 = log(tau0_prime) - log_sum_exp(0.5 * Dt.array().log());
  auto log_u = log(slab_scale) + 0.5*log_u_aux;
  // auto log_v = 0.5*log_v_aux;
  auto log_v = log_v_aux;
  
  auto log_tau1 = log_tau_aux_1 + 0.5*log_tau_aux_2 + log_tau0;
  auto log_lambda_tilde = add(log_lambda_tilde_aux_1, 0.5*log_lambda_tilde_aux_2);
  
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_num = add(log_u,log_lambda_tilde);
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda_den = -0.5*log_sum_exp(2.0*log_u,add(2.0*log_tau1,2.0*log_lambda_tilde));
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_lambda = add(log_lambda_num,log_lambda_den);
  Eigen::Matrix<T, Eigen::Dynamic, 1> lambda = exp(log_lambda);
  auto tau1 = exp(log_tau1);
  auto tau2 = exp(2.0 * (log_tau1 - log_u) + log_v);
  
  // Combine parameters into vector
  Eigen::Matrix<T, Eigen::Dynamic, 1> theta(p + 3);
  theta(0) = tau1;
  theta(1) = tau2;
  for (int j = 0; j < p; ++j)
    theta(j + 2) = lambda(j);
  theta(p + 2) = c;
  
  // Likelihood
  auto loglik = gp_1dloglik_analyticgrad_SKIM(X,F,Qt,Dt,theta,first_step);
  
  // Priors
  auto log_prior_log_c = std_normal_lpdf(c) + log_c;
  auto log_prior_log_u_aux = inv_gamma_lpdf(u_aux, 0.5*slab_df,0.5*slab_df) + log_u_aux;
  auto log_prior_log_v_aux = inv_gamma_lpdf(v_aux, 0.5*slab_df,0.5*slab_df) + log_v_aux;
  auto log_prior_log_tau_aux_1 = std_normal_lpdf(tau_aux_1) + log_tau_aux_1;
  auto log_prior_log_tau_aux_2 = inv_gamma_lpdf(tau_aux_2, 0.5*nu_global, 0.5*nu_global) + log_tau_aux_2;
  auto log_prior_log_lambda_tilde_aux_1 = std_normal_lpdf(lambda_tilde_aux_1) + sum(log_lambda_tilde_aux_1);
  auto log_prior_log_lambda_tilde_aux_2 = inv_gamma_lpdf(lambda_tilde_aux_2, 0.5*nu_local, 0.5*nu_local) + sum(log_lambda_tilde_aux_2);
  
  // Log post returned
  // Log posterior returned
  auto logpost = loglik +
    log_prior_log_c +
    log_prior_log_u_aux +
    log_prior_log_v_aux + 
    log_prior_log_tau_aux_1 +
    log_prior_log_tau_aux_2 +
    log_prior_log_lambda_tilde_aux_1 +
    log_prior_log_lambda_tilde_aux_2;
  
  return logpost;
}

// [[Rcpp::export]]
Rcpp::List sample_f_hypers_SKIM(Eigen::MatrixXd X,  // Data
                                const Eigen::MatrixXd& Qt,
                                const Eigen::VectorXd& Dt,
                                const Eigen::MatrixXd& F,  // Data
                                double tau0_prime_in,  // Hypers
                                double nugget_in, // Hypers
                                double ell_in, // Hypers
                                const Eigen::VectorXd& q_val, // Parameters
                                const Eigen::VectorXd& mass_matrix_diag, double epsilon, int L, // Hypers
                                const double& slab_scale = 5.0, // Hypers w/ default values
                                const double& slab_df = 4.0, // Hypers w/ default values
                                const int& nu_local = 1, // Hypers w/ default values
                                const int& nu_global = 1 // Hypers w/ default values
                                  
){
  
  using stan::math::var;
  using Eigen::VectorXd;
  using Eigen::Matrix;
  // Promote some stuff
  var tau0_prime = tau0_prime_in;
  var nugget = nugget_in;
  var ell = ell_in;
  
  // Set up the potential function lambda
  auto U = [&](const Eigen::Matrix<var, -1, 1> q_val, const bool first_step){
    return -gp_kron_logpost_1d_SKIM(X, Qt, Dt, F, //Data
                                    slab_scale, slab_df, nu_local, // Hypers
                                    nu_global, tau0_prime, nugget, ell, // Hypers
                                    q_val, // Parameters
                                    first_step
    );
  };
  
  return HMC_kernel(U, q_val, mass_matrix_diag, epsilon, L);
}




// random effects and noise
template <typename T>
T gp_kron_logpost_1d_re_noise(const Eigen::MatrixXd& Qt,
                              const Eigen::VectorXd& Dt,
                              const Eigen::MatrixXd& Y,
                              const double& temperature,
                              const double& eta,
                              const double& beta_gamma_a,
                              const double& beta_gamma_b,
                              const double& dir_a,
                              const Eigen::Matrix<T, Eigen::Dynamic, 1>& q,
                              const bool first_step
){
  
  // Defining some quantities we use
  int n = Y.rows();
  int m = Y.cols();

  // Pull out parameters
  Eigen::Matrix<T, Eigen::Dynamic, 1> log_phi_tilde = q.segment(0, n);
  T logit_u = q(n);
  T log_sigma = q(n+1);

  // Transform parameters
  Eigen::Matrix<T, Eigen::Dynamic, 1> phi_tilde = exp(log_phi_tilde); // Gamma distributed
  Eigen::Matrix<T, Eigen::Dynamic, 1> phi = exp(log_softmax(log_phi_tilde)); // Dirichlet distributed
  T u = inv_logit(logit_u); // Beta distribute
  T omega = exp(logit_u); // Beta prime distributed
  T omega_scaled = omega * eta; // Scaled beta prime distributed

  // And now create gammas and sigma
  Eigen::Matrix<T, Eigen::Dynamic, 1> gamma = sqrt(phi * omega_scaled);
  T sigma = exp(log_sigma);
  
  // And store in container
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> theta(n+1);
  theta.segment(0,n) = gamma;
  theta(n) = sigma;

  // Get the log-marginal likelihood
  auto loglik = gp_1dloglik_analyticgrad_re_noise(Y, Qt, Dt, theta,first_step);
  // Tack on priors
  auto log_prior_log_phi_tilde = stan::math::gamma_lpdf(phi_tilde,dir_a,1) + sum(log_phi_tilde);
  auto log_prior_logit_u = stan::math::beta_lpdf(u, beta_gamma_a, beta_gamma_b) + log(u) + log1m(u);
  // auto log_prior_log_sigma = 0.0; // Flat prior in log-space
  auto log_prior_log_sigma = stan::math::normal_lpdf(sigma, 0, 1);

  // Log posterior returned
  auto logpost = temperature * loglik +
    log_prior_log_phi_tilde +
    log_prior_logit_u +
    log_prior_log_sigma;
  return logpost;
}





//  [[Rcpp::export]]
Rcpp::List sample_z_hypers_and_noise(const Eigen::MatrixXd& Qt, 
  const Eigen::VectorXd& Dt,
  const Eigen::MatrixXd& Y,
  const Eigen::VectorXd& q_val, // Parameters
  const double& temperature,
  const Eigen::VectorXd& mass_matrix_diag, double epsilon, int L, // Hypers
  const double& eta,
  const double& beta_gamma_a,
  const double& beta_gamma_b,
  const double& dir_a){

  using stan::math::var;
  using Eigen::VectorXd;
  using Eigen::Matrix;
  
  auto U = [&](const Eigen::Matrix<var, -1, 1> q_val, const bool first_step){
    return -gp_kron_logpost_1d_re_noise(Qt, Dt,
                                        Y, temperature, 
                                        eta, beta_gamma_a, beta_gamma_b, dir_a,
                                        q_val, first_step);
  };

  return HMC_kernel(U,q_val,mass_matrix_diag,epsilon,L);
}
