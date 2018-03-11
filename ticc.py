from TICC_solver import solve
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
BIC_ITER_NUMBER = 4 # The max number of iterations to run for BIC
MAX_FEATURES = 30 # WILL SVD otherwise

def RunTicc(input_filename, output_filename, cluster_number=None, process_pool_size=1,
            window_size=1, lambda_param=11e-2, beta=None,
            maxIters=1000, threshold=2e-5, covariance_filename=None,
            input_format='matrix', delimiter=','):
    '''
    Required Parameters:
    -- input_filename: the path to the data file. see input_format below
    -- output_filename: the output file name to write the cluster assignments

    Optional Parameters:
    -- cluster_number: The number of clusters to classify. If not specified, then will
       perform on [3, 5, 10] and use BIC to choose
    -- process_pool_size: the number of processes to spin off for optimization. Default 1
    -- window_size: The size of the window for each cluster. Default 1
    -- lambda_param: sparsity penalty. Default 11e-2
    -- beta: the switching penalty. If not specified, will perform on 
       [50, 100, 200, 400] and then use BIC to choose
    -- maxIters: the maximum number of iterations to allow TICC to run. Default 1000
    -- threshold: the convergence threshold. Default 2e-5
    -- covariance_filename: if not None, write the covariance into this file
    -- file_type is the type of data file. the data file must 
       be a comma separated CSV. the options are:
       -- "matrix": a numpy matrix where each column is a feature and each
          row is a time step
       -- "graph": an adjacency list with each row having the form:
          <start label>, <end label>, value
    -- delimiter is the data file delimiter
    '''
    input_data = None
    if input_format == 'graph':
        input_data = retrieveInputGraphData(input_filename, delim=delimiter)
    elif input_format == "matrix":
        input_data = np.loadtxt(input_filename, delim=delimiter)
    else:
        assert False
    print "data shape is %s, %s" % (np.shape(input_data)[0], np.shape(input_data)[1])
    # perform necessary hyper parameter tuning
    if cluster_number is None:
        tempBeta = beta if beta is not None else 50
        cluster_number = runClusterParams(input_data, process_pool_size, window_size, lambda_param, tempBeta, threshold)
    
    if beta is None:
        beta = runBetaParams(input_data, process_pool_size, window_size, lambda_param, cluster_number, threshold)

    (cluster_assignment, cluster_MRFs) = solve(
        window_size=window_size,number_of_clusters=cluster_number, lambda_parameter=lambda_param,
        beta=beta, maxIters=maxIters, threshold=threshold,
        input_data=input_data, num_proc=process_pool_size)

    np.savetxt(output_filename, cluster_assignment, fmt='%d', delimiter=',')

    return cluster_assignment, cluster_MRFs

def retrieveInputGraphData(input_filename, delim=','):
    mapping = {} # edge to value
    sparse_cols = [] # list of indices that should be 1

    with open(input_filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delim, quotechar='|')
        counter = 0
        curr_timestamp = None
        for row in datareader:
            key = "%s_%s" % (row[0], row[1])
            timestamp = row[2]
            if timestamp != curr_timestamp: # new time
                curr_timestamp = timestamp
                sparse_cols.append(set())
            if key not in mapping: # a new feature
                mapping[key] = counter # assign this key to the current counter value
                counter += 1
            sparse_cols[-1].add(mapping[key]) # assign this feature into the current time step

    lenRow = len(mapping.keys())
    if lenRow <= MAX_FEATURES: # do not need to SVD
        rows = []
        for indices in sparse_cols:
            # indices is a set
            row = [1.0 if i in indices else 0.0 for i in range(lenRow)]
            rows.append(row)
        return np.array(rows)
    else:
        # need to truncated svd
        data = []
        rows = []
        cols = []
        for i, indices in enumerate(sparse_cols): # row
            for j in range(lenRow): # col
                if j in indices:
                    data.append(1)
                    rows.append(i)
                    cols.append(j)
        mat = csr_matrix((data, (rows, cols)), shape=(len(sparse_cols), lenRow))
        solver = TruncatedSVD(n_components=MAX_FEATURES)
        return solver.fit_transform(mat)

def runClusterParams(input_data, process_pool_size, window_size, lambda_param, beta, threshold):
    possibleClusters = [3, 5, 10]
    bestCluster = None
    bestScore = None
    for clusterNum in possibleClusters:
        _, _, score, _ = solve(input_data=input_data, window_size=window_size,number_of_clusters=clusterNum, lambda_parameter=lambda_param,
            beta=beta, maxIters=BIC_ITER_NUMBER, threshold=threshold, num_proc=process_pool_size, compute_BIC=True)
        if bestScore is None or bestScore < score:
            bestCluster = clusterNum
            bestScore = score
    print "performed cluster BIC and solving with cluster = %f" % bestCluster
    return bestCluster

def runBetaParams(input_data, process_pool_size, window_size, lambda_param, clusterNum, threshold):
    possibleBetas = [50,100, 200, 400]
    bestBeta = None
    bestScore = None
    for beta in possibleBetas:
        _, _, _, score = solve(input_data=input_data, window_size=window_size,number_of_clusters=clusterNum, lambda_parameter=lambda_param,
            beta=beta, maxIters=BIC_ITER_NUMBER, threshold=threshold, num_proc=process_pool_size, compute_BIC=True)
        if bestScore is None or bestScore < score:
            bestBeta = beta
            bestScore = score 
    print "performed beta BIC and solving with beta = %f" % bestBeta
    return bestBeta

