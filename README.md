# TICC
TICC is a python solver for efficiently segmenting and clustering a multivariate time series. For implementation details, refer to the paper [1]. 

----
The TICC method takes as input a T-by-n data matrix, a regularization parameter "lambda" and smoothness parameter "beta", the window size "w" and the number of clusters "k".  TICC breaks the T timestamps into segments where each segment belongs to one of the "k" clusters. The total number of segments is defined by the smoothness parameter "beta". It does so by running an EM algorithm where TICC alternately assigns points to clusters using a DP algorithm and updates the cluster parameters by solving a Toeplitz Inverse Covariance Estimation problem. The details can be found in the paper.

Download & Setup
======================
Download the source code, by running in the terminal:
```
git clone https://github.com/davidhallac/TICC.git
```
Using TICC
======================

```
TICC_solver.py
```
Solver for the TICC algorithm. This file utilizes several helper functions in the src directory, but the user only needs to interface with TICC_solver.py. The solve function within the file can run an instance of the TICC algorithm. The details of the solve function are as below:

**Parameters**

window_size : the size of the sliding window

number_of_clusters: the number of underlying clusters 'k'

lambda_parameter: sparsity of the MRF for each of the clusters. The sparsity of the inverse covariance matrix of each cluster.

beta: The switching penalty used in the TICC algorithm. Same as the beta parameter described in the paper. 

maxIters : the maximum iterations of the TICC algorithm before covnergence. Default value is 100.

threshold: convergence threshold

write_out_file : Boolean. Flag indicating if the computed inverse covariances for each of the clusters should be saved.

prefix_string: Location of the folder to which you want to save the outputs.

input_file: Location of the Data matrix of size T-by-n.


**Returns**

returns an array of cluster assignments for each time point.

returns a dictionary with keys being the cluster_id (from 0 to k-1) and the values being the cluster MRFs.

----

Example Usage
======================

See example.py for proper usage of TICC.


References
==========
[1] TICC paper : http://stanford.edu/~hallac/TICC.pdf
