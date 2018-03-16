import TICC_solver as TICC
import numpy as np
import sys
from ticc import RunTicc

fname = "edge_data.txt"

cluster_assignment, cluster_mrfs = RunTicc(fname, "Results.txt", process_pool_size=10, thread_pool_size=5, input_format='graph', delimiter='\t')
