#include <iostream>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <armadillo>
#include <numeric>
using namespace arma;

// Stores position q and momentum p
struct pq_struct {
  arma::vec q;
  arma::vec p;
};

// Stores position-momentum of forward and backwards path
struct Tree {
  arma::vec minus_q;
  arma::vec plus_q;
  arma::vec minus_p;
  arma::vec plus_p;
  arma::vec q_prima;
  int n;
  int s;
};


// Gamma-Poisson posterior
float posterior_h_cpp(const arma::vec& h_n, const arma::ivec& v_n, const arma::mat& W, 
                      float alpha, float beta){
  
  float logp = 0;
  
  if(any(h_n < 0)){
    return - arma::datum::inf;
  }
  
  int K = W.n_cols;
  int F = W.n_rows;
  
  arma::vec lambdas = W * h_n;
  
  for(int f = 0; f < F; f++){
    logp += v_n[f] * log(lambdas[f]) - lambdas[f]; //- log(std::tgamma(v_n[f]+1));  
  }
  for(int k = 0; k < K; k++){
    logp += (alpha-1)*log(h_n[k]) - beta*h_n[k];
  }
  
  // Similar alternative:
  //logp = sum(v_n % log(lambdas)) - beta*sum(lambdas) + 
  //        sum((alpha-1)*log(h_n) - beta*h_n);
  
  return logp;
}

// Posterior wrapper
double loglike_cpp(const arma::vec& h_n, const arma::ivec& v_n, const arma::mat& W, 
             float alpha, float beta){
  return posterior_h_cpp(h_n, v_n, W, alpha, beta);
}

// Gradient of the log posterior
arma::vec grad_loglike_cpp(const arma::vec& h_n, const arma::ivec& v_n, 
                           const arma::mat& W, const arma::rowvec& norms_W,
                           float alpha, float beta){
  
  int K = h_n.size();  
  arma::vec dh = zeros(K);
  arma::vec lambdas = W * h_n; // this is the slowest bottleneck
  for(int k=0; k < K; k++){
    dh[k] = (alpha-1)/h_n[k] - (beta + norms_W[k]) + sum(v_n % (W.col(k)/lambdas));
  }
  return dh;
}


// Performs one leapfrom step (NUTS paper, Algorithm 1)
pq_struct leapfrog(arma::vec q, arma::vec p, float epsilon, 
                   const arma::ivec &v_n, const arma::mat& W, const arma::rowvec& norms_W, 
                   float alpha, float beta){
  
  pq_struct pq;
  //std::cout << "* Leapfrog init" << std::endl;
  //std::cout << "q: " << q.t() << std::endl;
  //std::cout << "p: " << p.t() << std::endl;
  //std::cout << "grad" << p + epsilon * 0.5 * grad_loglike_cpp(q, v_n, W, norms_W, alpha=alpha, beta=beta) << endl;
  p += epsilon * 0.5 * grad_loglike_cpp(q, v_n, W, norms_W, alpha=alpha, beta=beta);
  //std::cout << "p_: " << p.t() << std::endl;
  q += epsilon * p;
  //std::cout << "q_: " << q.t() << std::endl;
  q  = abs(q); // Bouncing. Recommended by Nico (to stay in non-negative values)
  //std::cout << "q_: " << q.t() << std::endl;
  p += epsilon * 0.5 * grad_loglike_cpp(q, v_n, W, norms_W, alpha=alpha, beta=beta);
  //std::cout << "p_: " << p.t() << std::endl;

  pq.q = q; 
  pq.p = p;
  //std::cout << "* Leapfrog end" << std::endl;
  //std::cout << q.t() << std::endl;
  
  return pq;
}

Tree BuildTree(arma::vec& q, arma::vec& p, float logu, int v, int j, float epsilon,
               const arma::ivec& v_n, const arma::mat& W, const arma::rowvec& norms_W, 
               float alpha, float beta, std::default_random_engine& generator){
  int ntabs = 5-j;
  std::cout << std::string(ntabs, '\t') << "TREE j= " << j << std::endl;

  //std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  int K = W.n_cols;
  int F = W.n_rows;

  Tree tree;
  arma::vec q_prima(F);
  arma::vec p_prima(F);
  arma::vec minus_q(F);
  arma::vec minus_p(F);
  arma::vec plus_q(F);
  arma::vec plus_p(F);
  int n_prima;
  int s_prima;
  arma::vec q_prima2(F);
  int n_prima2;
  int s_prima2;
  
  float delta_max = 1000; // Recommended in the NUTS paper: 1000
  
  if(j == 0){
    std::cout << std::string(ntabs, '\t') << "Leapfrog-----------------" << std::endl;
    std::cout << std::string(ntabs, '\t') << "Leapfrog init:" << q.t();

    // Base case - take a single leapfrog step in the direction v
    pq_struct pq = leapfrog(q, p, v*epsilon, v_n, W, norms_W, alpha, beta);
    q_prima = pq.q; 
    p_prima = pq.p;
	float joint = loglike_cpp(q_prima, v_n, W, alpha, beta) - 0.5 * sum(p_prima % p_prima); 
 	
 	// Is the new point in the slice?
    int n_prima   = (logu <= joint);

    // Is the simulation wildly inaccurate?
    int s_prima   = joint - logu > - delta_max;

    // Set the return values 
    // minus=plus for all things here, since the tree is of depth 0.
    tree.minus_q = q_prima;
    tree.minus_p = p_prima;
    tree.plus_q  = q_prima;
    tree.plus_p  = p_prima;
    tree.q_prima = q_prima;
    tree.n = n_prima;
    tree.s = s_prima;

    std::cout << std::string(ntabs, '\t') << "Leapfrog end :" << tree.q_prima.t();
    std::cout << std::string(ntabs, '\t') << "Leapfrog in the slice? ok: 1, ko: 0 :" << n_prima << std::endl;
    std::cout << std::string(ntabs, '\t') << "Simulation wildly inacurate? ok: 1, ko: 0 :" << s_prima << std::endl;
    
    return tree;
    
  } else {
    std::cout << std::string(ntabs, '\t') << "Recursion-----------------" << std::endl;
    std::cout << std::string(ntabs, '\t') << "implicitly build the left and right subtrees-" << std::endl;
    
    std::cout << std::string(ntabs, '\t')  << "from q: " << q.t() << std::endl;
    // Recursion -- implicitly build the left and right subtrees
    Tree tree = BuildTree(q, p, logu, v, j-1, epsilon, v_n, W, norms_W, alpha, beta, generator);
    minus_q = tree.minus_q;
    minus_p = tree.minus_p;
    plus_q  = tree.plus_q;
    plus_p  = tree.plus_p;
    q_prima = tree.q_prima;
    n_prima = tree.n;
    s_prima = tree.s;
    std::cout << std::string(ntabs, '\t')  << "Left:"  << minus_q.t();
    std::cout << std::string(ntabs, '\t')  << "Right:" << plus_q.t();
    std::cout << std::string(ntabs, '\t')  << "s_prima:" << s_prima << std::endl;
    if(s_prima == 1){

       // Double the size of the tree.
      if(v == -1){
        std::cout << std::string(ntabs, '\t')  << "Double to the left from q-: "  << minus_q.t();
        Tree tree = BuildTree(minus_q, minus_p, logu, v, j-1, epsilon, v_n, W, norms_W, alpha, beta, generator);
        minus_q  = tree.minus_q;
        minus_p  = tree.minus_p;
        q_prima2 = tree.q_prima;
        n_prima2 = tree.n;
        s_prima2 = tree.s;
      } else {
        std::cout << std::string(ntabs, '\t')  << "Double to the right from q+:"  << plus_q.t();
        Tree tree  = BuildTree(plus_q, plus_p, logu, v, j-1, epsilon, v_n, W, norms_W, alpha, beta, generator);
        plus_q   = tree.plus_q;
        plus_p   = tree.plus_p;
        q_prima2 = tree.q_prima;
        n_prima2 = tree.n;
        s_prima2 = tree.s;
      }

      // Choose which subtree to propagate a sample up from.
      float prob = static_cast<float>(n_prima2) / std::max((n_prima + n_prima2), 1); // avoids 0/0;
      float rand01 = unif01(generator);
      std::cout << '\n' << std::string(ntabs, '\t')  << "Choose subtree to propagate a sample up from. Probs:" << rand01 << "/" << prob << std::endl;
      if(rand01 < prob){
        q_prima = q_prima2;
        std::cout << std::string(ntabs, '\t')  << "Chose qprima2" << std::endl;
      } else{
      	std::cout << std::string(ntabs, '\t')  << "Chose qprima" << std::endl;
  	  }	

      arma::vec diff_q = (plus_q - minus_q);
      int uturn = (sum(diff_q % minus_p) >= 0) && (sum(diff_q % plus_p) >= 0);
      s_prima  =  s_prima2 * uturn; // stop if large error or U-turn
      n_prima +=  n_prima2; // Update the number of valid points. 
    }
    std::cout << std::string(ntabs, '\t')  << "Right-Left trees built";
    
    tree.minus_q = minus_q;
    tree.minus_p = minus_p;
    tree.plus_q  = plus_q;
    tree.plus_p  = plus_p;
    tree.q_prima = q_prima;
    tree.n = n_prima;
    tree.s = s_prima;
    return tree;
  }
}

arma::mat sample_nuts_cpp(const arma::ivec v_n, const arma::mat& W, arma::vec h_n_current,
                          double alpha = 1, double beta = 1,
                          float epsilon = 0.01,
                          int iter=100){
  
  int K = W.n_cols;
  int F = W.n_rows;
  
  std::default_random_engine generator;
  std::uniform_real_distribution<double> unif01(0.0,1.0);
  std::normal_distribution<double> normal(0,1);


  // Pre-compute column norms
  const arma::rowvec u_(F);
  const arma::rowvec norms_W = u_ * W;
  
  arma::mat h_n_samples(K, iter);   // traces of p
  arma::vec q(K);                   // position
  arma::vec current_q(K);
  arma::vec p0(K);                  // initial momentum
  //arma::vec current_p(K);
  arma::vec minus_q(K);
  arma::vec minus_p(K);
  arma::vec plus_q(K);
  arma::vec plus_p(K);
  arma::vec q_prima(K);
  current_q = h_n_current;
  h_n_samples.col(1) = h_n_current;



  for(int i=2; i<iter; i++){
    std::cout << "************************************************************" << i << std::endl;
    std::cout << "************************************************************" << std::endl;
    std::cout << "Sample: " << i << std::endl;

    // Sample new momentum (K independent standard normal variates)
    for(int k=0; k<K; k++){
      p0[k] = normal(generator);
    }
    std::cout << "Momentum p0: " << p0.t() << std::endl;

    // Joint logprobability of position q and momentum p
    float joint = loglike_cpp(current_q, v_n, W, alpha, beta) - 0.5* sum(p0 % p0);

    // Resample u ~ uniform([0, exp(joint)]). 
    // double limit_sup = exp( loglike_cpp(current_q, v_n, W, alpha, beta) - 0.5* sum(p % p));
    // std::uniform_real_distribution<double> distribution(0.0,limit_sup); Computational issues
    // Equivalent to (log(u) - joint) ~ exponential(1).
    //logu = joint - exprnd(1);
    std::exponential_distribution<double> distribution(1);
    float random = distribution(generator);
    float logu = joint - random;

    // If all fails, the next sample will be the previous on
    // not needed. current_q is, at any moment, the last accepted
    // current_q = samples_q(m-1); 

    // Initialize the tree
    minus_q = current_q;  // position in the backward path
    plus_q = current_q;   // position in the forward path
    minus_p = p0;          // momentum in the backward path
    plus_p = p0;           // momentum in the forward path
        
    int j = 0;            // Initial heigth j = 0
    int n = 1;            // Initially the only valid point is the initial point
    int s = 1;            // Main loop: will keep going until stop criterion s == 0.
    
    
    // While no U-turn
    while(s==1){
      std::cout << "\n~~~~~~~~Depth: " << j << "(will open 2^j trees): " << j << std::endl;

      Tree tree;  
      int n_prima;
      int s_prima;

      // Choose a direction. -1 = backwards, 1 = forwards.
      int v = 2 * (unif01(generator) < 0.5) - 1;
      
      std::cout << "Current values:" << std::endl;
      std::cout << "q+:" << plus_q.t();
      std::cout << "q-:" << minus_q.t();

      std::cout << "New chosen direction:" << v << std::endl;
      
      // Double the size of the tree
      if(v == -1){    // if backwards
        tree = BuildTree(minus_q, minus_p, logu, v, j, epsilon, v_n, W, norms_W, 
                               alpha, beta, generator);
        minus_q = tree.minus_q;
        minus_p = tree.minus_p;
        
      } else {       // if forwards
        tree = BuildTree(plus_q,   plus_p, logu, v, j, epsilon, v_n, W, norms_W, 
                              alpha, beta, generator);
        plus_q  = tree.plus_q;
        plus_p  = tree.plus_p;
      }
      q_prima = tree.q_prima;
      n_prima = tree.n;
      s_prima = tree.s;

      std::cout << "End of trees at level j" << std::endl;
      std::cout << "q+:" << plus_q.t();
      std::cout << "q-:" << minus_q.t();
      std::cout << "n' (valid points in subtrees at level j):" << n_prima << std::endl;
      std::cout << "s' (stop criteria met in subtrees at level j):" << s_prima << std::endl;
      std::cout << "q':" << q_prima.t();
    
      
      if(s_prima == 1){ 
        // Use Metropolis-Hastings to decide whether or not to move to a
        // point from the half-tree we just generated.
        std::cout << "prob q' accepted = n'/n = " << n_prima/n;
        if(unif01(generator) < std::min(1, n_prima/n)){ 
          current_q = q_prima; // Accept proposal (it will be THE new sample when s=0)
          std::cout << "--Accepted:" << std::endl;
        } else {
          std::cout << "--Rejected: " << std::endl;
        }
      }

      // Update number of valid points we've seen.
      n += n_prima;

      // Decide if it's time to stop.
      arma::vec diff_q = (plus_q - minus_q);
      s = s_prima && (sum(diff_q % minus_p) >= 0) && (sum(diff_q % plus_p) >= 0);

      // Increment depth.
      j++; 

    } // end while
    
    h_n_samples.col(i) = current_q;
    
  } // end for
  h_n_samples = h_n_samples.t();
  return(h_n_samples);
  
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


  float epsilon = 0.00001;
  int iter = 3;
  arma::mat samples = sample_nuts_cpp(v_n, W, h_n+10, alpha, beta, epsilon, iter);

  std::cout << "samples of h_n:" << std::endl;
  std::cout << samples << endl;

  std::cout << "Real h_n:" << std::endl;
  std::cout << h_n.t() << endl;
}

