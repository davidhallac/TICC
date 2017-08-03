# TICC
TICC is a python solver for efficiently segmenting and clustering a multivariate time series. For implementation details refer to the paper. 

----
The TICC method takes as input a T-by-n data matrix, a regularization parameter "lambda" and smoothness parameter "beta", the window size "w" and the number of clusters "k".  TICC breaks the T timestamps into segments where each segment belongs to one of the "k" clusters. The total number of segments is defined by the smoothness parameter "beta". It does so by running an EM algorithm where TICC alternately assigns points to clusters using a DP algorithm and updates the cluster parameters by solving a Toeplitz Inverse Covariance Estimation problem. The details can be found in the paper.

Download & Setup
======================
Download the source code, by running in the terminal:
```
git clone https://github.com/davidhallac/TICC.git
```
Files 
======================
The TICC package has the following important files:
```
TICC.py
```
Runs an instance of TICC algorithm.

**Parameters**

lambda_parameter : the lambda regularization parameter as described in the paper

beta : the beta parameter controlling the smoothness of the output as described in the paper

number_of_cluster: the number of clusters 'k' that the time stamps are clustered into

window_size : the size of the sliding window

prefix_string : the location of the output files

threhsold : used for generating the cross time plots. Not used in the TICC algorithm

input_file : Location of the data file of size T-by-n.

maxIters : maximum iteration of the TICC algorithm


**Returns**

saves a .csv file for each of the cluster inverse covariances

saves a .csv file with list of the assignments for each of the timestamps to the 'k' clusters

prints the binary accuracy, if the correct method for computing the confusion matrix is specified

----

```
car.py
```
Runs an instance of TICC algorithm on the car example (case-study), as described in the paper. The parameters are the same as the TICC example. Note: this file is specifically useful for the car-example, doing some special input data handling for the car dataset. For running an instance of the TICC algorithm please use TICC.py or TICC_solver.py.

**Parameters**

lambda_parameter : the lambda regularization parameter as described in the paper

beta : the beta parameter controlling the smoothness of the output as described in the paper

number_of_cluster: the number of clusters 'k' that the time stamps are clustered into

window_size : the size of the sliding window

prefix_string : the location of the output files

threshold : used for generating the cross time plots. Not used in the TICC algorithm

input_file : Location of the data file of size T-by-n.

maxIters : maximum iteration of the TICC algorithm

**Returns**

saves a .csv file for each of the cluster inverse covariances

saves a .csv file with list of the assignments for each of the timestamps to the 'k' clusters

saves a .csv file with the locations information

saves a .csv file with the color information for each of the time stamps

----

```
network_accuracy.py
```
Runs an instance of TICC algorithm on the T-by-n data matrix as described in the paper. Used for generating the network accuracy table as shown in the paper. The parameters are the same as the TICC example.

**Parameters**

lambda_parameter : the lambda regularization parameter as described in the paper

beta : the beta parameter controlling the smoothness of the output as described in the paper

number_of_cluster: the number of clusters 'k' that the time stamps are clustered into

window_size : the size of the sliding window

prefix_string : the location of the output files

threhsold : used for generating the cross time plots. Not used in the TICC algorithm

input_file : Location of the data file of size T-by-n.

maxIters : maximum iteration of the TICC algorithm

**Returns**

saves a .csv file for each of the cluster inverse covariances

saves a .csv file with list of the assignments for each of the timestamps to the 'k' clusters

prints the network F1 scores for each of the clusters, assuming the "true" networks are stored as specified in the file.

----
```
generate_synthetic_data.py
```
Generates data using the methodology described in the paper. The data is generated from 'k' number of clusters. The 'T' time stamps are broken down into segments, and the segment lengths of the corresponding clusters should be mentioned in the 'break_points' array and 'seg_ids' list, respectively. So length of segment 'i' = break_points[i+1] - break_points[i].

**Parameters**

window_size : the size of the sliding window

number_of_sensors : The dimension 'n' of the output T-by-n data matrix.

sparsity_inv_matrix: sparsity of the MRF for each of the clusters. The sparsity of the inverse covariance matrix of each cluster.

rand_seed : The random seed used for generating random numbers

number_of_cluster: the number of clusters 'k' that the time stamps are generated from

cluster_ids : The corresponding cluster ids from which the segments are generated.

break_points : The end point of the segments. So length of segment 'i' = break_points[i+1] - break_points[i]

save_inverse_covariances : Boolean. Flag indicating if the computed inverse covariances for each of the clusters should be 
saved as "Inverse Covariance cluster = cluster#.csv"

out_file_name : The file name where the .csv data matrix should be stored.

**Returns**

saves a .csv file with data matrix of shape T-by-n

saves a .csv file for each of the inverse covariances of each cluster if the save_inverse_covariances flag is True.

----
```
scalability_test.py
```
Runs an instance of the scalability test. Prints out the time required for each step: E-step (DP algorithm) and M-step (Optimization using Toeplitz Graphical Lasso).

**Parameters**

number_of_cluster: the number of clusters 'k' that the time stamps are clustered into

window_size : the size of the sliding window

input_file : Location of the data file of size T-by-n.

maxIters : maximum iteration of the TICC algorithm

**Output**

prints out the time taken for each of the steps in TICC algorithm. This function was used to generate the scalability plot in the paper.

----
```
TICC_solver.py
```
Solver for the TICC algorithm. Contains all the important functions. The solve function within the file can run an instance of the TICC algorithm. The details of the solve function are as below:

**Parameters**

window_size : the size of the sliding window

maxIters : the maximum iterations of the TICC algorithm before covnergence. Default value is 100.

lambda_parameter: sparsity of the MRF for each of the clusters. The sparsity of the inverse covariance matrix of each cluster.

beta: The switching penalty used in the TICC algorithm. Same as the beta parameter described in the paper. 

number_of_clusters: the number of clusters 'k' that the time stamps are generated from

threshold: the threshold parameter used in visualization. Not a part of the TICC algorithm.

input_file: Location of the Data matrix of size T-by-n.

prefix_string: Location of the folder to which you want to save the outputs.

write_out_file : Boolean. Flag indicating if the computed inverse covariances for each of the clusters should be 
saved as "Inverse Covariance cluster = cluster#.csv"

**Returns**

returns an array of cluster assignments for each time point.

returns a dictionary with keys being the cluster_id (from 0 to k-1) and the values being the cluster MRFs.

----

Example Usage
======================

Generating the data. In case, you already have a data matrix, skip this step. For generating the data as mentioned in the paper, use generate_synthetic_data.py. Change the parameters of break_points and seg_ids, to define the temporal pattern of your time series that you want to generate. Use the sparsity_inv_matrix to define the sparsity of the MRF of each cluster. ALso set window_size, number_of_sensors appropriately according to your application. Then run the following command:

```
python generate_synthetic_data.py
```
Next use the TICC.py file for running an instance of the TICC algorithm on the data matrix. The TICC.py method should be initialized with the following parameters : smoothness parameter 'beta', sparsity regularization 'lambda', window size, maximum Iterations before convergence, number of clusters, location of the input and output file. After updating this in the TICC.py file, run the following:
  
```
python TICC.py
```
For generating the network accuracy plots, use the Network.py file. Add the same parameters as above in the network_accuracy.py file and additionally save the true Inverse covariances as "Inverse Covariance cluster = 'cluster#'.csv" in the same directory as the network_accuracy.py file. Next run:
```
python network_accuracy.py
```
For running a scalability experiment, use the scalability_test.py file. Set the parameters within the file same as the TICC.py file, and run the following command:
```
python scalability_test.py
```

For using the solver, on your data , the usage is as shown below. Enter the parameters as mentioned in the paper. Use the output cluster_assignments and the dictioanry of the cluster_MRFs, as needed in the application.
```
import TICC_solver as TICC
(cluster_assignment, cluster_MRFs) = TICC.solve(window_size = 10,number_of_clusters = 5, lambda_parameter = 11e-2, beta = 400, maxIters = 100, threshold = 2e-5, write_out_file = False, input_file = "data.csv", prefix_string = "output_folder/"):
```


References
==========
[1] TICC paper : http://stanford.edu/~hallac/TICC.pdf
