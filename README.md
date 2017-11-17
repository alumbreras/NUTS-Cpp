# No-U-Turn sampler

C++ implementation of the NUTS sampler. 

This is a simplified version of the STAN code, intended to be more comprehensible.
Unlike STAN, there is no Automatic Differentiation. Thus, the logprobability and its gradient are to be explicitly coded in their corresponding functions.
