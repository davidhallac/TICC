import TICC_solver as TICC
import numpy as np
import sys
from ticc import RunTicc
import logging

fname = "edge_data.txt"

cluster_assignment, cluster_mrfs,params = RunTicc(fname, "Results.txt", process_pool_size=2, input_format='graph', delimiter='\t', logging_level=logging.DEBUG)
