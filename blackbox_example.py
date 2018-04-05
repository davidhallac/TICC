import TICC_solver as TICC
import numpy as np
import sys
from ticc import RunTicc, GetChangePoints
import logging

fname = "synthetic.txt"
log_level = logging.WARNING
results = RunTicc(fname, "Results.txt", input_format='matrix', logging_level=log_level, cluster_number=[2,3,4], process_pool_size=5)

for cluster_assignment, cluster_MRFs, params in results:
    print("-----")
    print(params)
    indices = GetChangePoints(cluster_assignment)
    print(indices)


fname2 = "edge_data.txt"
log_level = logging.INFO
results2 = RunTicc(fname2, "Results.txt", input_format='graph', delimiter='\t', logging_level=log_level, cluster_number=[2,3,4], input_dimensions=10, process_pool_size=)
for cluster_assignment, cluster_MRFs, params in results2:
    print("-----")
    print(params)
    indices = GetChangePoints(cluster_assignment)
    print(indices)
