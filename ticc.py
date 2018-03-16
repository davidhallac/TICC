from TICC_solver import solve
import csv
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, PCA
import logging
from collections import namedtuple
import concurrent.futures as cf



# Problem Instance. Contains fields for problem except for the BIC
# changeable ones.
ProblemInstance = namedtuple('ProblemInstance',
                             ['input_data', 'window_size', 'maxIters', 'threshold'])


def RunTicc(input_filename, output_filename, cluster_number=range(2, 11), process_pool_size=1,
            window_size=1, lambda_param=11e-2, beta=[50, 100, 200, 400],
            maxIters=1000, threshold=2e-5, covariance_filename=None,
            input_format='matrix', delimiter=',', BIC_Iters=None, input_dimensions=None,
            logging_level=logging.INFO, logging_filename=None):
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
    logging.basicConfig(filename=logging_filename, level=logging.DEBUG)

    input_data = None
    if input_format == 'graph':
        input_data = retrieveInputGraphData(
            input_filename, input_dimensions, delim=delimiter)
    elif input_format == "matrix":
        input_data = np.loadtxt(input_filename, delim=delimiter)
        if input_dimensions is not None:
            pca = PCA(n_components=input_dimensions)
            input_data = pca.fit_transform(input_data)

    else:
        raise ValueError("input_format must either be graph or matrix")

    print "Data loaded! With shape %s, %s" % (
        np.shape(input_data)[0], np.shape(input_data)[1])

    # get params via BIC
    cluster_number = cluster_number if isinstance(
        cluster_number, list) else [cluster_number]
    beta = beta if isinstance(beta, list) else [beta]
    lambda_param = lambda_param if isinstance(
        lambda_param, list) else [lambda_param]
    BIC_Iters = maxIters if BIC_Iters is None else BIC_Iters
    problem_instance = ProblemInstance(input_data=input_data, window_size=window_size,
                                       maxIters=BIC_Iters, threshold=threshold)
    params, results, score = runHyperParameterTuning(beta, lambda_param, cluster_number,
                                                     process_pool_size, problem_instance)
    beta, cluster_number, lambda_param = params
    print "Via BIC with score %s, using params beta: %s, clusterNum %s, lambda %s" % (
        score, beta, cluster_number, lambda_param)

    # perform real run
    cluster_assignments, cluster_MRFs = (None, None)
    if BIC_Iters == maxIters:  # already performed the full run
        cluster_assignments, cluster_MRFs = results
    else:
        (cluster_assignment, cluster_MRFs) = solve(
            window_size=window_size, number_of_clusters=cluster_number, lambda_parameter=lambda_param,
            beta=beta, maxIters=maxIters, threshold=threshold,
            input_data=input_data, num_proc=process_pool_size)

    np.savetxt(output_filename, cluster_assignment, fmt='%d', delimiter=',')

    return cluster_assignment, cluster_MRFs, params


def retrieveInputGraphData(input_filename, input_dimensions, delim=','):
    mapping = {}  # edge to value
    sparse_cols = []  # list of indices that should be 1

    with open(input_filename, 'rb') as csvfile:
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
        solver = TruncatedSVD(n_components=MAX_FEATURES)
        return solver.fit_transform(mat)


def runHyperParameterTuning(beta_vals, lambda_vals, cluster_vals,
                            process_pool_size, problem_instance):
    num_runs = len(beta_vals)*len(lambda_vals)*len(cluster_vals)
    pool_size = min(process_pool_size, num_runs)

    # use a threadpool since daemons can't have kids. This is a bit slower
    # because of the GIL but ticc solve creates processes that do the real
    # work anyway
    # pool = Pool(processes=pool_size)
    processes_per, extra_processes = dividPoolSizes(
        process_pool_size, num_runs)

    with cf.ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
        futures = [None]*num_runs
        futureIndex = 0
        for l in lambda_vals:
            for c in cluster_vals:
                for b in beta_vals:
                    process_size = processes_per
                    if extra_processes > 0:
                        process_size += 1
                        extra_processes -= 1
                    futures[futureIndex] = executor.submit(runBIC, b, c, l, process_size, problem_instance)
                    futureIndex += 1
        # retrieve results
        bestParams = (0, 0, 0)  # beta, cluster, lambda
        bestResults = (None, None)
        bestScore = None
        for future in cf.as_completed(futures):
            clusts, mrfs, score, params = future.result()
            if bestScore is None or score > bestScore:
                bestScore = score
                bestParams = params
                bestResults = (clusts, mrfs)
        # for i, l in enumerate(lambda_vals):
        #     for j, c in enumerate(cluster_vals):
        #         for k, b in enumerate(beta_vals):
        #             cf.wait(futures[k][j][i])
        #             clusts, mrfs, score = futures[k][j][i].result()
        #             if bestScore is None or score > bestScore:
        #                 bestScore = score
        #                 bestParams = (b, c, l)
        #                 bestResults = (clusts, mrfs)
    return bestParams, bestResults, bestScore


def dividPoolSizes(process_pool_size, num_runs):
    # allocate processes per so that in total we don't go over the process pool size
    # don't count threads in pool (since they'll be waiting on the indiv pool threads
    # anyway)
    # at least one process per job
    processes_per = max(1, process_pool_size/num_runs)
    processes_used = processes_per*num_runs
    # any extra processes left over
    extra_processes = max(0, process_pool_size - processes_used)
    return processes_per, extra_processes


def runBIC(beta, cluster, lambd, process_size, pi):
    ''' pi should be a problem instance '''
    clusts, mrfs, score = solve(input_data=pi.input_data, window_size=pi.window_size,
                                number_of_clusters=cluster, lambda_parameter=lambd,
                                beta=beta, maxIters=pi.maxIters, threshold=pi.threshold,
                                num_proc=process_size, compute_BIC=True)
    return clusts, mrfs, score, (beta,cluster,lambd)
