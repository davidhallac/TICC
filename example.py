from TICC_solver import TICC
import numpy as np
import sys

fname = "example_data.txt"
ticc = TICC(window_size=1, number_of_clusters=8, lambda_parameter=11e-2, beta=600, maxIters=100, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1)
(cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

print(cluster_assignment)
np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
