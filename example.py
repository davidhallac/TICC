import TICC_solver as TICC
import numpy as np
import sys

fname = "example_data.txt"
(cluster_assignment, cluster_MRFs) = TICC.solve(window_size = 1,number_of_clusters = 8, lambda_parameter = 11e-2, beta = 600, maxIters = 100, threshold = 2e-5, write_out_file = False, input_file = fname, prefix_string = "output_folder/", num_proc=1)

print cluster_assignment
np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
