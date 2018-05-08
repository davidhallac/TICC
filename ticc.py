from TICC_solver import solve
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
import logging
from collections import namedtuple
import concurrent.futures as cf
import threading
from multiprocessing import Pool
import multiprocessing

# Problem Instance. Contains fields for problem except for the BIC
# changeable ones.
ProblemInstance = namedtuple('ProblemInstance',
                             ['input_data', 'window_size', 'maxIters', 'threshold'])


def RunTicc(input_filename, output_filename, cluster_number=range(2, 11), process_pool_size=10,
            window_size=1, lambda_param=[1e-2], beta=[0.01, 0.1, 0.5, 10, 50, 100, 500],
            maxIters=1000, threshold=2e-5, covariance_filename=None,
            input_format='matrix', delimiter=',', BIC_Iters=15, input_dimensions=50,
            logging_level=logging.INFO):
    '''
    Required Parameters:
    -- input_filename: the path to the data file. see input_format below
    -- output_filename: the output file name to write the cluster assignments

    Optional Parameters: BIC
    For each of these parameters, one can choose to specify:
        - a single number: this value will be used as the parameter
        - a list of numbers: the solver will use grid search on the BIC to choose the parameter
        - not specified: the solver will grid search on a default range (listed) to choose the parameter
    -- cluster_number: The number of clusters to classify. Default: BIC on [2...10]
    -- lambda_param: sparsity penalty. Default: BIC on 11e-2]
    -- beta: the switching penalty. If not specified, BIC on [50, 100, 200, 400]

    Other Optional Parameters:
    -- input_dimensions: if specified, will truncated SVD the matrix to the given number of features
       if the input is a graph, or PCA it if it's a matrix
    -- BIC_iters: if specified, will only run BIC tuning for the given number of iterations
    -- process_pool_size: the number of processes to spin off for optimization. Default 1
    -- window_size: The size of the window for each cluster. Default 1
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
    logging.basicConfig(level=logging_level)
    input_data = None
    if input_format == 'graph':
        input_data = retrieveInputGraphData(
            input_filename, input_dimensions, delim=delimiter)
    elif input_format == "matrix":
        input_data = np.loadtxt(input_filename, delimiter=delimiter)
        if input_dimensions is not None and input_dimensions < np.shape(input_data)[1]:
            pca = PCA(n_components=input_dimensions)
            input_data = pca.fit_transform(input_data)

    else:
        raise ValueError("input_format must either be graph or matrix")

    logging.debug("Data loaded! With shape %s, %s" % (
        np.shape(input_data)[0], np.shape(input_data)[1]))

    # get params via BIC
    cluster_number = cluster_number if isinstance(
        cluster_number, list) else [cluster_number]
    beta = beta if isinstance(beta, list) else [beta]
    lambda_param = lambda_param if isinstance(
        lambda_param, list) else [lambda_param]
    BIC_Iters = maxIters if BIC_Iters is None else BIC_Iters
    problem_instance = ProblemInstance(input_data=input_data, window_size=window_size,
                                       maxIters=BIC_Iters, threshold=threshold)
    clusterResults = runHyperParameterTuning(beta, lambda_param, cluster_number,
                                             process_pool_size, problem_instance)
    final_results = []
    for cluster_number, resultPackage in clusterResults:
        params, results, score = resultPackage
        beta, lambda_param = params
        logging.info("Via BIC with score %s, using params beta: %s, clusterNum %s, lambda %s" % (
            score, beta, cluster_number, lambda_param))
        # perform real run
        cluster_assignments, cluster_MRFs = (None, None)
        if BIC_Iters == maxIters:  # already performed the full run
            cluster_assignments, cluster_MRFs = results
        else:
            (cluster_assignment, cluster_MRFs) = solve(
                window_size=window_size, number_of_clusters=cluster_number, lambda_parameter=lambda_param,
                beta=beta, maxIters=maxIters, threshold=threshold,
                input_data=input_data, num_processes=process_pool_size, logging_level=logging_level)
        outstream = "%s_%s" % (cluster_number, output_filename)
        np.savetxt(outstream, cluster_assignment, fmt='%d', delimiter=',')
        final_results.append(
            (cluster_assignment, cluster_MRFs, (beta, lambda_param, cluster_number)))
    return final_results


def GetChangePoints(cluster_assignment):
    '''
    Pass in the result of RunTicc to split into changepoint indexes
    '''
    currIndex = -1
    index = -1
    currCluster = -1
    results = []
    for cluster in cluster_assignment:
        if currCluster != cluster:
            if currCluster != -1:
                results.append((currIndex, index, currCluster))
            index += 1
            currIndex = index
            currCluster = cluster
        else:
            index += 1
    results.append((currIndex, index, currCluster))
    return results


def retrieveInputGraphData(input_filename, input_dimensions, delim=','):
    mapping = {}  # edge to value
    sparse_cols = []  # list of indices that should be 1

    with open(input_filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=delim, quotechar='|')
        counter = 0
        curr_timestamp = None
        for row in datareader:
            key = "%s_%s" % (row[0], row[1])
            timestamp = row[2]
            if timestamp != curr_timestamp:  # new time
                curr_timestamp = timestamp
                sparse_cols.append(set())
            if key not in mapping:  # a new feature
                # assign this key to the current counter value
                mapping[key] = counter
                counter += 1
            # assign this feature into the current time step
            sparse_cols[-1].add(mapping[key])

    lenRow = len(mapping.keys())
    if input_dimensions is None or lenRow <= input_dimensions:  # do not need to SVD
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
        for i, indices in enumerate(sparse_cols):  # row
            for j in range(lenRow):  # col
                if j in indices:
                    data.append(1)
                    rows.append(i)
                    cols.append(j)
        mat = csr_matrix((data, (rows, cols)),
                         shape=(len(sparse_cols), lenRow))
        solver = TruncatedSVD(n_components=input_dimensions)
        return solver.fit_transform(mat)


def runHyperParameterTuning(beta_vals, lambda_vals, cluster_vals,
                            process_pool_size, problem_instance):
    num_runs = len(beta_vals)*len(lambda_vals)
    pool = Pool(processes=process_pool_size)
    futures = []
    for i, c in enumerate(cluster_vals):
        future_list = []
        for l in lambda_vals:
            for b in beta_vals:
                future_list.append(pool.apply_async(
                    runBIC, (b, c, l, problem_instance,)))
        futures.append(future_list)
    # retrieve results
    # [cluster, (bestParams, bestResults, bestScore)]
    results = []
    for i, c in enumerate(cluster_vals):
        bestParams = (0, 0)  # beta, cluster, lambda
        bestResults = (None, None)
        bestScore = None
        bestConverge = False
        for j in range(num_runs):
            vals = futures[i][j].get()
            clusts, mrfs, score, converged, params = vals
            logging.info("%s,%s,%s" % (cluster_vals[i], params, score))
            if bestScore is None or (converged >= bestConverge and score < bestScore):
                bestScore = score
                bestParams = params
                bestResults = (clusts, mrfs)
                bestConverge = converged
        resultPackage = (bestParams, bestResults, bestScore)
        results.append((cluster_vals[i], resultPackage))
    pool.close()
    pool.join()
    return results


def runBIC(beta, cluster, lambd, pi):
    ''' pi should be a problem instance '''
    clusts, mrfs, score, converged = solve(input_data=pi.input_data, window_size=pi.window_size,
                                           number_of_clusters=cluster, lambda_parameter=lambd,
                                           beta=beta, maxIters=pi.maxIters, threshold=pi.threshold,
                                           compute_BIC=True, num_processes=1)
    return clusts, mrfs, score, converged, (beta, lambd)
