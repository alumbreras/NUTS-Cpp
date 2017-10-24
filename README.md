# No-U-Turn sampler

C++ implementation of the NUTS sampler. 

The only available implementation is in STAN. However, this is far too complicated compared to the Python or Matlab versions.
The idea is to provide an efficient and simple implementation, with everything in the same file. 
Unlike STAN, there is no Automatic Differentiation. Thus, the logprobability and its gradient are to be explicitly coded in their corresponding functions.
