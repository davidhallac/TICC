import TICC_solver as TICC
import numpy as np
import sys
from ticc import RunTicc, GetChangePoints
import logging

fname = "synthetic.txt"
log_level = logging.WARNING
results = RunTicc(fname, "Results.txt", process_pool_size=5, input_format='matrix', delimiter=',', logging_level=log_level, cluster_number=[2,3,4], beta=[0.01, 0.1, 0.5, 10, 50], BIC_Iters=10)

for cluster_assignment, cluster_MRFs, params in results:
    print "-----"
    print params
    indices = GetChangePoints(cluster_assignment)
    print indices
