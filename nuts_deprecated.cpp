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
  pq_point(int k) {q(k); p(k);};
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

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
int BuildTree(pq_point* z, pq_point* z_init_parent, pq_point& z_propose, nuts_util util, int depth, float epsilon,
              posterior_params& postparams, 
              std::default_random_engine& generator){

  int ntabs = std::max(1, 5-depth);

  //std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  int K = postparams.W.n_cols;
  int F = postparams.W.n_rows;  
  float delta_max = 1000; // Recommended in the NUTS paper: 1000
  
  if(depth == 0){
    // Base case - take a single leapfrog step in the direction v
    leapfrog(z_propose, util.sign *epsilon, postparams);

    // If called from a left tree, it modifies also plus and minus. (the other extreme, z_init)
    // Else, only plus or minus
    if (z_init_parent) *z_init_parent = z;

	float joint = loglike_cpp(z_propose.q, postparams) - 0.5 * sum(z_propose.p % z_propose.p); 
    int n_valid_subtree   = (util.log_u <= joint);  	// Is the new point in the slice?
    util.criterion = util.log_u - joint < delta_max; // Is the simulation wildly inaccurate?
	util.n_tree += 1;
    return n_valid_subtree;

  } else {
    // Recursion -- implicitly build the left and right subtrees

    // Left subtree
    pq_point *z_init = *z;
    int n1 = BuildTree(&z, &z_init, z_propose, util, depth-1, epsilon, postparams, generator);
    if (!util.criterion) return 0; // early stopping

	// Right subtree
    pq_point *z_propose_right(K);
	z_propose_right.q = z_propose.q;
	z_propose_right.p = z_propose.p;
	int n2 = BuildTree(z*, &z_propose_right, util, depth-1, epsilon, postparams, generator);

	// Choose which subtree to propagate a sample up from.
	//double accept_prob = static_cast<double>(n_prima2) / std::max((n_prima + n_prima2), 1); // avoids 0/0;
	double accept_prob = static_cast<double>(n2) / static_cast<double>(n1 + n2);
	float rand01 = unif01(generator);
	if(util.criterion && (rand01 < accept_prob)){
	z_propose = z_propose_right;
	}

	arma::vec diff_q = (z_plus.q - z_minus.q); //z_init - z
	int uturn = (sum(diff_q % z_minus.p) >= 0) && (sum(diff_q % z_plus.p) >= 0);
	util.criterion  =  criterion * uturn; // stop if large error or U-turn

    
    return n1 + n2;
  }
}

arma::mat sample_nuts_cpp(const arma::ivec v_n, const arma::mat& W, arma::vec current_q,
                          double alpha = 1, double beta = 1,
                          float epsilon = 0.01,
                          int iter=100){
  
  int K = W.n_cols;
  int F = W.n_rows;
  int MAXDEPTH = 3;
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  std::normal_distribution<double> normal(0,1);

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
  arma::vec current_q(K);			// position

  current_q = log(current_q); 		// Transform to unrestricted space
  h_n_samples.col(1) = current_q;
  
  pq_point z_propose(K);
  pq_point *z(K);
  pq_point z_plus(K);
  pq_point z_minus(K);
  
  arma::vec rho(K);
  arma::vec rho_plus(K);
  arma::vec rho_minus(K);

  for(int i=2; i<iter; i++){
    std::cout << " ************************ Sample: " << i << std::endl;

    nuts_util util;

    // Sample new momentum (K independent standard normal variates)
    for(int k=0; k<K; k++){
      p0[k] = normal(generator);
    }

    // Joint logprobability of position q and momentum p
    float joint = loglike_cpp(current_q, postparams) - 0.5* sum(p0 % p0);

    // Sample the slice variable
    // Resample u ~ uniform([0, exp(joint)]). 
    // double limit_sup = exp( loglike_cpp(current_q, v_n, W, alpha, beta) - 0.5* sum(p % p));
    // std::uniform_real_distribution<double> distribution(0.0,limit_sup); Computational issues
    // Equivalent to (log(u) - joint) ~ exponential(1).
    //logu = joint - exprnd(1);
    std::exponential_distribution<double> distribution(1);
    float random = distribution(generator);
    util.log_u = joint - random;

    // If all fails, the next sample will be the previous on
    // not needed. current_q is, at any moment, the last accepted
    // current_q = samples_q(m-1); 

    // Initialize the tree
    z_propose.q = current_q;
    z_propose.p = p0;
    z_minus.q = current_q;   // position and momentum in the backward path
    z_minus.p = p0;		    
    z_plus.q = current_q;          // momentum in the backward path
    z_plus.p = p0;          // momentum in the forward path

    int j = 0;            // Initial heigth j = 0
    int n = 1;			  
    
    // Build a balanced binary tree until the NUTS criterion fails
    util.criterion = true;
    int n_valid = 0; // Initially the only valid point is the initial point
    int depth_ = 0;
    int divergent_ = 0;

    util.n_tree = 0;
    util.sum_prob = 0;

    // While no U-turn
    while((!util.criterion) && (depth_ < MAXDEPTH)){

      // // Randomly sample a direction. -1 = backwards, 1 = forwards.
      util.sign = 2 * (unif01(generator) < 0.5) - 1;
    
      // Set the variables to update (right or left)
      // z and rho are pointers to the right/left positions
      // Build a new subtree in the chosen direction
      // (Modifies z_propose, z_minus, z_plus)
      pq_point* z = 0;
      arma::vec* rho = 0;
      if(util.sign == -1){    
      	   z   = &z_minus;
           //rho = &rho_minus;
      } else {  
      	   z   = &z_plus;
           //rho = &rho_plus;
      }

      int n_valid_subtree = BuildTree(z, 0, z_propose, util, depth_, epsilon, postparams, generator);
      ++depth_;  // Increment depth.
       
      if(!util.criterion){ 
        // Use Metropolis-Hastings to decide whether or not to move to a
        // point from the half-tree we just generated.
        double subtree_prob = std::min(1.0, static_cast<double>(n_valid_subtree)/n_valid);
        if(unif01(generator) < subtree_prob){ 
          current_q = z_propose.q; // Accept proposal (it will be THE new sample when s=0)
        }
      }

      // Update number of valid points we've seen.
      n_valid += n_valid_subtree;

      // Decide if it's time to stop.
      arma::vec diff_q = (z_plus.q - z_minus.q);
      util.criterion = util.criterion && (sum(diff_q % z_minus.p) >= 0) && (sum(diff_q % z_plus.q) >= 0);

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
  h_n = h_n.ones()*10;

  for(int f=0; f<F; f++){
    float lambda_f = as_scalar(W.row(f) * h_n);
    std::poisson_distribution<int> mypoisson(lambda_f);
    v_n[f] = mypoisson(generator);
  }

  //std::cout << "Observation:" << std::endl;
  //std::cout << v_n.t() << std::endl;
  //std::cout << "Dictionary W:" << std::endl;
  //std::cout << W << std::endl;


  float epsilon = 0.0000001;
  int iter = 3;
  arma::mat samples = sample_nuts_cpp(v_n, W, h_n+10, alpha, beta, epsilon, iter);

  std::cout << "samples of h_n:" << std::endl;
  std::cout << samples << endl;

  std::cout << "Real h_n:" << std::endl;
  std::cout << h_n.t() << endl;
}

