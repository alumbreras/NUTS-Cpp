#include <iostream>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <armadillo>
#include <numeric>
using namespace arma;

// Position q and momentum p
struct pq_point {

  arma::vec q;
  arma::vec p;
  
  explicit pq_point(int n): q(n), p(n) {}
  pq_point(const pq_point& z): q(z.q.size()), p(z.p.size()) {
        q = z.q;
        p = z.p;
  }

  pq_point& operator= (const pq_point& z) {
        if (this == &z)
          return *this;

        q = z.q;
        p = z.p;
        
        return *this;
  }
};


struct nuts_util {
  // Constants through each recursion
  double log_u; // uniform sample
  double H0; 	// Hamiltonian of starting point?
  int sign; 	// direction of the tree in a given iteration/recursion

  // Aggregators through each recursion
  int n_tree;
  double sum_prob; 
  bool criterion;

  // just to guarantee bool initializes to valid value
  nuts_util() : criterion(false) { }
};


struct posterior_params {
 	arma::ivec v_n;
 	arma::mat W;
 	arma::mat norms_W;  
 	double alpha;
 	double beta;
};


// Gamma-Poisson log posterior
double posterior_h_cpp(const arma::vec& h_n, const arma::ivec& v_n, const arma::mat& W, 
                        float alpha, float beta){
  double logp = 0;
  
  if(any(h_n < 0)){
    return - arma::datum::inf;
  }
  
  int K = W.n_cols;
  int F = W.n_rows;
  
  arma::vec lambdas = W * h_n;
  
  for(int f = 0; f < F; f++){
    logp += v_n[f] * log(lambdas[f]) - lambdas[f]; //- std::lgamma(v_n[f]+1);
  }

  for(int k = 0; k < K; k++){
    logp += alpha * log(h_n[k]) - log(h_n[k]) - beta*h_n[k];
  }
  
  return logp;
}

// Transformed Gamma-Poisson log posterior
double posterior_eta_cpp(const arma::vec& eta_n, const arma::ivec& v_n, const arma::mat& W, 
                        float alpha, float beta){
  arma::vec exp_eta = exp(eta_n);
  double logp = posterior_h_cpp(exp_eta, v_n, W, alpha, beta) + sum(eta_n);
  return logp;
}


// Posterior wrapper
double loglike_cpp(const arma::vec& eta_n, const posterior_params& postparams){
  return posterior_eta_cpp(eta_n, 
  						postparams.v_n, 
  						postparams.W, 
  						postparams.alpha, 
  						postparams.beta);
}

// Gradient of the log posterior
arma::vec grad_loglike_cpp(const arma::vec& eta_n, const arma::ivec& v_n, 
                                 const arma::mat& W, const arma::rowvec& norms_W,
                                 float alpha, float beta){
  
  int K = eta_n.size();  
  arma::vec dh = zeros(K);
  arma::vec lambdas = W * exp(eta_n);
  for(int k=0; k < K; k++){
    dh[k] = alpha - exp(eta_n[k])*(beta + norms_W[k]) + sum(v_n % (W.col(k)*exp(eta_n[k])/lambdas));
  }
  return dh;
}


// Performs one leapfrom step (NUTS paper, Algorithm 1)
void leapfrog(pq_point &z, float epsilon, posterior_params& postparams){
  
  z.p += epsilon * 0.5 * grad_loglike_cpp(z.q, 
  										postparams.v_n, 
  										postparams.W, 
  										postparams.norms_W, 
  										postparams.alpha, 
  										postparams.beta);
  z.q += epsilon * z.p;
  z.p += epsilon * 0.5 * grad_loglike_cpp(z.q,									
  										postparams.v_n, 
  										postparams.W, 
  										postparams.norms_W, 
  										postparams.alpha, 
  										postparams.beta);
}

// U-Turn criterion in the generalized form applicable to Riemanian spaces
// See Betancourt's Conceptual HMC (page 58)  
bool compute_criterion(arma::vec& p_sharp_minus, 
                               arma::vec& p_sharp_plus,
                               arma::vec& rho) {
  return  sum(p_sharp_plus % rho)  > 0 && sum(p_sharp_minus % rho) > 0;
      }

//**
// Recursively build a new subtree to completion or until the subtree becomes invalid.
// Returns validity of the resulting subtree
// @param z last visited state?
// @param depth Depth of the desired subtree
// @z_propose State proposed from subtree
// @p_sharp_left p_sharp from left boundary of returned tree (p_sharp = inv(M)*p)
// @p_sharp_right p_sharp from right boundary of returned tree
// @rho Summed momentum accross trajectory (to compute the generalized stoppin criteria)
int BuildTree(pq_point& z, pq_point& z_propose, 
              arma::vec& p_sharp_left, 
              arma::vec& p_sharp_right, 
              arma::vec& rho, 
              nuts_util& util, 
              int depth, float epsilon,
              posterior_params& postparams, 
              std::default_random_engine& generator){

  //std::cout << "\n Tree direction:" << util.sign << " Depth:" << depth << std::endl;

  int K = z.q.n_rows;

  //std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  int F = postparams.W.n_rows;  
  float delta_max = 1000; // Recommended in the NUTS paper: 1000
  
  // Base case - take a single leapfrog step in the direction v
  if(depth == 0){
    leapfrog(z, util.sign * epsilon, postparams);
    float joint = loglike_cpp(z.q, postparams) - 0.5 * sum(z.p % z.p); 
    int valid_subtree = (util.log_u <= joint);    // Is the new point in the slice?
    util.criterion = util.log_u - joint < delta_max; // Is the simulation wildly inaccurate? // TODO: review
    util.n_tree += 1;

    z_propose = z;
    rho += z.p;
    p_sharp_left = z.p;  // p_sharp = inv(M)*p (Betancourt 58)
    p_sharp_right = p_sharp_left;

    return valid_subtree;
  } 

  // General recursion
  arma::vec p_sharp_dummy(K);

  // Build the left subtree
  arma::vec rho_left(K); rho_left.zeros();
  int n1 = BuildTree(z, z_propose, p_sharp_left, p_sharp_dummy, rho_left, util, depth-1, epsilon, postparams, generator);

  if (!util.criterion) return 0; // early stopping

  // Build the right subtree
  pq_point z_propose_right(z);
  arma::vec rho_right(K); rho_left.zeros();
  int n2 = BuildTree(z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right, util, depth-1, epsilon, postparams, generator);

  // Choose which subtree to propagate a sample up from.
  //double accept_prob = static_cast<double>(n2) / static_cast<double>(n1 + n2);
  double accept_prob = static_cast<double>(n2) / std::max((n1 + n2), 1); // avoids 0/0;
  float rand01 = unif01(generator);
  if(util.criterion && (rand01 < accept_prob)){
    z_propose = z_propose_right;
  }

  // Break when NUTS criterion is no longer satisfied
  arma::vec rho_subtree = rho_left + rho_right;
  rho += rho_subtree;
  util.criterion = compute_criterion(p_sharp_left, p_sharp_right, rho);

  int n_valid_subtree = n1 + n2;
  return(n_valid_subtree);
}




arma::mat sample_nuts_cpp(const arma::ivec v_n, const arma::mat& W, arma::vec current_q,
                          double alpha = 1, double beta = 1,
                          float epsilon = 0.01,
                          int iter=100){
  
  int K = W.n_cols;
  int F = W.n_rows;
  int MAXDEPTH = 10;
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  std::normal_distribution<double> normal(0,1);
  std::exponential_distribution<double> exp1(1);

  //const arma::rowvec norms_W = u_ * W;  
  // Store fixed data and parameters
  posterior_params postparams;
  const arma::rowvec u_(F);
  postparams.W 	 	 = W;
  postparams.v_n 	 = v_n;
  postparams.norms_W = u_ * W;
  postparams.alpha 	 = alpha;
  postparams.beta 	 = beta;

  arma::mat h_n_samples(K, iter);   // traces of p
  arma::vec p0(K);                  // initial momentum

  current_q = log(current_q); 		// Transform to unrestricted space
  h_n_samples.col(1) = current_q;
  
  pq_point z(K);

  
  // Used to compute the NUTS generalized stopping criterion
  arma::vec rho(K);


  // Transition
  for(int i=2; i<iter; i++){
    std::cout << "******************* Sample: " << i << std::endl;

    nuts_util util;

    // Sample new momentum (K independent standard normal variates)
    for(int k=0; k<K; k++){
      p0[k] = normal(generator);
    }

    // Initialize the path. Proposed sample,
    // and leftmost/rightmost position and momentum
    ////////////////////////
    z.q = current_q;
    z.p = p0;
    pq_point z_plus(z);
    pq_point z_minus(z);
    pq_point z_propose(z);
    
    // Utils o compute NUTS stop criterion
    arma::vec p_sharp_plus = z.p;
    arma::vec p_sharp_dummy = p_sharp_plus;
    arma::vec p_sharp_minus = p_sharp_plus;
    arma::vec rho(z.p);

    // Hamiltonian
    // Joint logprobability of position q and momentum p
    float joint = loglike_cpp(current_q, postparams) - 0.5* sum(p0 % p0);

    // Slice variable
    ///////////////////////
    // Sample the slice variable: u ~ uniform([0, exp(joint)]). 
    // Equivalent to: (log(u) - joint) ~ exponential(1).
    // logu = joint - exprnd(1);
    std::exponential_distribution<double> exp1(1);
    float random = exp1(generator);
    util.log_u = joint - random;

    int n_valid = 1;
    util.criterion = true;

    // Build a trajectory until the NUTS criterion is no longer satisfied
    int depth_ = 0;
    int divergent_ = 0;
    util.n_tree = 0;
    util.sum_prob = 0;


    // Build a balanced binary tree until the NUTS criterion fails
    while(util.criterion && (depth_ < MAXDEPTH)){
      std::cout << "*****depth : " << depth_  << std::endl;

      // Build a new subtree in the chosen direction
      // (Modifies z_propose, z_minus, z_plus)
      arma::vec rho_subtree(K);
      rho_subtree.zeros();

      // Build a new subtree in a random direction
      util.sign = 2 * (unif01(generator) < 0.5) - 1;
      int n_valid_subtree=0;
      if(util.sign == 1){    
      	   z.pq_point::operator=(z_minus);
           n_valid_subtree = BuildTree(z, z_propose, p_sharp_dummy, p_sharp_plus, rho_subtree, util, depth_, epsilon, postparams, generator);
           z_plus.pq_point::operator=(z);
      } else {  
           z.pq_point::operator=(z_plus);
           n_valid_subtree = BuildTree(z, z_propose, p_sharp_dummy, p_sharp_minus, rho_subtree, util, depth_, epsilon, postparams, generator);
           z_minus.pq_point::operator=(z);
      }
      //if(!valid_subtree) break;
      
      ++depth_;  // Increment depth.
       

      if(util.criterion){ 
        // Use Metropolis-Hastings to decide whether or not to move to a
        // point from the half-tree we just generated.
        double subtree_prob = std::min(1.0, static_cast<double>(n_valid_subtree)/n_valid);

        if(unif01(generator) < subtree_prob){ 
          current_q = z_propose.q; // Accept proposal (it will be THE new sample when s=0)
        }
      }

      // Update number of valid points we've seen.
      n_valid += n_valid_subtree;

      // Break when NUTS criterion is no longer satisfied
      rho += rho_subtree;
      util.criterion = util.criterion && compute_criterion(p_sharp_minus, p_sharp_plus, rho);
    } // end while
    

    h_n_samples.col(i) = current_q;
    
  } // end for

  h_n_samples = h_n_samples.t();
  
  return(exp(h_n_samples));
}




int main(){
  
  int K = 2;
  int F = 100;

  double alpha = 1;
  double beta  = 1;

  std::default_random_engine generator;
  std::gamma_distribution<float> mygamma(alpha, 1/beta);

  
  const arma::mat W = randu<mat>(F, K);
  arma::ivec v_n(F);
  arma::vec h_n(K);
  h_n = h_n.ones()*5;

  for(int f=0; f<F; f++){
    float lambda_f = as_scalar(W.row(f) * h_n);
    std::poisson_distribution<int> mypoisson(lambda_f);
    v_n[f] = mypoisson(generator);
  }

  //std::cout << "Observation:" << std::endl;
  //std::cout << v_n.t() << std::endl;
  //std::cout << "Dictionary W:" << std::endl;
  //std::cout << W << std::endl;


  float epsilon = 0.0001;
  int iter = 10000;
  arma::mat samples = sample_nuts_cpp(v_n, W, h_n+20, alpha, beta, epsilon, iter);

  std::cout << "samples of h_n:" << std::endl;
  std::cout << samples << endl;

  std::cout << "Real h_n:" << std::endl;
  std::cout << h_n.t() << endl;
}

