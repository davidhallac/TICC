# TICC
TICC is a python solver for efficiently segmenting and clustering a multivariate time series. It takes as input a T-by-n data matrix, a regularization parameter `lambda` and smoothness parameter `beta`, the window size `w` and the number of clusters `k`.  TICC breaks the T timestamps into segments where each segment belongs to one of the `k` clusters. The total number of segments is affected by the smoothness parameter `beta`. It does so by running an EM algorithm where TICC alternately assigns points to clusters using a dynamic programming algorithm and updates the cluster parameters by solving a Toeplitz Inverse Covariance Estimation problem. 

For details about the method and implementation see the paper [1].

## Download & Setup
Download the source code, by running in the terminal:
```
git clone https://github.com/davidhallac/TICC.git
```


## Using TICC
The `TICC`-constructor takes the following parameters:

* `window_size`: the size of the sliding window
* `number_of_clusters`: the number of underlying clusters 'k'
* `lambda_parameter`: sparsity of the Markov Random Field (MRF) for each of the clusters. The sparsity of the inverse covariance matrix of each cluster.
* `beta`: The switching penalty used in the TICC algorithm. Same as the beta parameter described in the paper. 
* `maxIters`: the maximum iterations of the TICC algorithm before convergence. Default value is 100.
* `threshold`: convergence threshold
* `write_out_file`: Boolean. Flag indicating if the computed inverse covariances for each of the clusters should be saved.
* `prefix_string`: Location of the folder to which you want to save the outputs.


The `TICC.fit(input_file)`-function runs the TICC algorithm on a specific dataset to learn the model parameters.

* `input_file`: Location of the data matrix of size T-by-n.

An array of cluster assignments for each time point is returned in the form of a dictionary with keys being the `cluster_id` (from `0` to `k-1`) and the values being the cluster MRFs.


## Example Usage

See `example.py`.


## References
[1] D. Hallac, S. Vare, S. Boyd, and J. Leskovec [Toeplitz Inverse Covariance-Based Clustering of
Multivariate Time Series Data](http://stanford.edu/~hallac/TICC.pdf) Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 215--223
