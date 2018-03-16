import TICC_solver as TICC
import numpy as np
import sys
from ticc import RunTicc
import logging

fname = "synthetic.txt"
log_level = logging.DEBUG
cluster_assignment, cluster_mrfs,params = RunTicc(fname, "Results.txt", process_pool_size=5, input_format='matrix', delimiter=',', logging_level=logging.DEBUG, cluster_number=[2,3,4,5], beta=[0.01, 0.1, 0.5, 10, 50, 100], BIC_Iters=10)
print params
print cluster_assignment
