import numpy as np 
import math, time, collections, os, errno, sys, code, random
import __builtin__ as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats

from sklearn import mixture, covariance
from sklearn.cluster import KMeans
import pandas as pd

from multiprocessing import Pool

from src.TICC_helper import *
from src.admm_solver import ADMMSolver
#######################################################################################################################################################################
pd.set_option('display.max_columns', 500)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.random.seed(102)

#####################################################################################################################################################################################################

def solve(window_size=10, number_of_clusters=5, lambda_parameter=11e-2,
    beta=400, maxIters=1000, threshold=2e-5, write_out_file=False,
    input_file=None, prefix_string="", num_proc=1, compute_BIC=False):
    '''
    Main method for TICC solver.
    Parameters:
        - window_size: size of the sliding window
        - number_of_clusters: number of clusters
        - lambda_parameter: sparsity parameter
        - beta: temporal consistency parameter
        - maxIters: number of iterations
        - threshold: convergence threshold
        - write_out_file: (bool) if true, prefix_string is output file dir
        - prefix_string: output directory if necessary
        - input_file: location of the data file
    '''
    assert maxIters > 0 # must have at least one iteration
    num_blocks = window_size + 1
    num_stacked = window_size
    switch_penalty = beta # smoothness penalty
    lam_sparse = lambda_parameter # sparsity parameter
    num_clusters = number_of_clusters # Number of clusters
    cluster_reassignment = 20 # number of points to reassign to a 0 cluster
    print "lam_sparse", lam_sparse
    print "switch_penalty", switch_penalty
    print "num_cluster", num_clusters
    print "num stacked", num_stacked

    ######### Get Data into proper format
    Data = np.loadtxt(input_file, delimiter= ",") 
    (m,n) = Data.shape # m: num of observations, n: size of observation vector
    print "completed getting the data"

    ############
    ##The basic folder to be created
    str_NULL = prefix_string + "lam_sparse=" + str(lam_sparse) + "maxClusters=" +str(num_clusters+1)+"/"
    if not os.path.exists(os.path.dirname(str_NULL)):
        try:
            os.makedirs(os.path.dirname(str_NULL))
        except OSError as exc: # Guard against race condition of path already existing
            if exc.errno != errno.EEXIST:
                raise

    ###-------INITIALIZATION----------
    # Train test split
    training_indices = getTrainTestSplit(m, num_blocks, num_stacked) #indices of the training samples
    num_train_points = bt.len(training_indices)
    num_test_points = m - num_train_points
    ##Stack the training data
    complete_D_train = np.zeros([num_train_points, num_stacked*n])
    for i in xrange(num_train_points):
        for k in xrange(num_stacked):
            if i+k < num_train_points:
                idx_k = training_indices[i+k]
                complete_D_train[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
    # Initialization
    gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type="full")
    gmm.fit(complete_D_train)
    clustered_points = gmm.predict(complete_D_train) 
    gmm_clustered_pts = clustered_points + 0
    gmm_covariances = gmm.covariances_
    gmm_means = gmm.means_
    # USE K-means
    kmeans = KMeans(n_clusters = num_clusters,random_state = 0).fit(complete_D_train)
    clustered_points_kmeans = kmeans.labels_ #todo, is there a difference between these two?
    kmeans_clustered_pts = kmeans.labels_

    train_cluster_inverse = {}
    log_det_values = {} # log dets of the thetas
    computed_covariance = {} 
    cluster_mean_info = {}
    cluster_mean_stacked_info = {}
    old_clustered_points = None # points from last iteration

    empirical_covariances = {}

    # PERFORM TRAINING ITERATIONS
    pool=Pool(processes=num_proc)
    for iters in xrange(maxIters):
        print "\n\n\nITERATION ###", iters
        ##Get the train and test points
        train_clusters = collections.defaultdict(list) # {cluster: [point indices]}
        for point, cluster in enumerate(clustered_points):
            train_clusters[cluster].append(point)

        len_train_clusters = {k: len(train_clusters[k]) for k in xrange(num_clusters)}

        # train_clusters holds the indices in complete_D_train 
        # for each of the clusters
        optRes = [None for i in xrange(num_clusters)]
        for cluster in xrange(num_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters[cluster]
                D_train = np.zeros([cluster_length,num_stacked*n])
                for i in xrange(cluster_length):
                    point = indices[i]
                    D_train[i,:] = complete_D_train[point,:]
                
                cluster_mean_info[num_clusters,cluster] = np.mean(D_train, axis = 0)[(num_stacked-1)*n:num_stacked*n].reshape([1,n])
                cluster_mean_stacked_info[num_clusters,cluster] = np.mean(D_train,axis=0)
                ##Fit a model - OPTIMIZATION    
                probSize = num_stacked * size_blocks
                lamb = np.zeros((probSize,probSize)) + lam_sparse
                S = np.cov(np.transpose(D_train) )
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, num_stacked, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))


        for cluster in xrange(num_clusters):
            if optRes[cluster] == None:
                continue
            val = optRes[cluster].get()
            print "OPTIMIZATION for Cluster #", cluster,"DONE!!!"
            #THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[num_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[num_clusters,cluster] = cov_out
            train_cluster_inverse[cluster] = X2

        for cluster in xrange(num_clusters):
            print "length of the cluster ", cluster,"------>", len_train_clusters[cluster]

        # update old computed covariance
        old_computed_covariance = computed_covariance
        print "UPDATED THE OLD COVARIANCE"

        inv_cov_dict = {} # cluster to inv_cov
        log_det_dict = {} # cluster to log_det
        for cluster in xrange(num_clusters):
            cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov

        # -----------------------SMOOTHENING
        # For each point compute the LLE 
        print "beginning the smoothening ALGORITHM"

        LLE_all_points_clusters = np.zeros([bt.len(clustered_points),num_clusters])
        for point in xrange(bt.len(clustered_points)):
            if point + num_stacked-1 < complete_D_train.shape[0]:
                for cluster in xrange(num_clusters):
                    cluster_mean = cluster_mean_info[num_clusters,cluster] 
                    cluster_mean_stacked = cluster_mean_stacked_info[num_clusters,cluster] 
                    x = complete_D_train[point,:] - cluster_mean_stacked[0:(num_blocks-1)*n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
                    LLE_all_points_clusters[point,cluster] = lle
        
        ##Update cluster points - using NEW smoothening
        clustered_points = updateClusters(LLE_all_points_clusters,switch_penalty = switch_penalty)

        if iters != 0:
            cluster_norms = [(np.linalg.norm(old_computed_covariance[num_clusters,i]), i) for i in xrange(num_clusters)]
            norms_sorted = sorted(cluster_norms,reverse = True)
            # clusters that are not 0 as sorted by norm
            valid_clusters = [cp[1] for cp in norms_sorted if len_train_clusters[cp[1]] != 0]

            # Add a point to the empty clusters 
            # assuming more non empty clusters than empty ones
            counter = 0
            for cluster in xrange(num_clusters):
                if len_train_clusters[cluster] == 0:
                    cluster_selected = valid_clusters[counter] # a cluster that is not len 0
                    counter = (counter+1) % len(valid_clusters)
                    print "cluster that is zero is:", cluster, "selected cluster instead is:", cluster_selected
                    start_point = np.random.choice(train_clusters[cluster_selected]) # random point number from that cluster
                    for i in range(0, cluster_reassignment):
                        # put cluster_reassignment points from point_num in this cluster
                        point_to_move = start_point + i
                        if point_to_move >= len(clustered_points):
                            break
                        clustered_points[point_to_move] = cluster
                        computed_covariance[num_clusters,cluster] = old_computed_covariance[num_clusters,cluster_selected]
                        cluster_mean_stacked_info[num_clusters,cluster] = complete_D_train[point_to_move,:]
                        cluster_mean_info[num_clusters,cluster] = complete_D_train[point_to_move,:][(num_stacked-1)*n:num_stacked*n]
        

        for cluster in xrange(num_clusters):
            print "length of cluster #", cluster, "-------->", sum([x== cluster for x in clustered_points])

        ##Save a figure of segmentation
        plt.figure()
        plt.plot(training_indices[0:bt.len(clustered_points)],clustered_points,color = "r")#,marker = ".",s =100)
        plt.ylim((-0.5,num_clusters + 0.5))
        if write_out_file: plt.savefig(str_NULL + "TRAINING_EM_lam_sparse="+str(lam_sparse) + "switch_penalty = " + str(switch_penalty) + ".jpg")
        plt.close("all")
        print "Done writing the figure"

        true_confusion_matrix = compute_confusion_matrix(num_clusters,clustered_points,training_indices)

        ####TEST SETS STUFF
        ### LLE + swtiching_penalty
        ##Segment length
        ##Create the F1 score from the graphs from k-means and GMM
        ##Get the train and test points
        train_confusion_matrix_EM = compute_confusion_matrix(num_clusters, clustered_points,training_indices)
        train_confusion_matrix_GMM = compute_confusion_matrix(num_clusters, gmm_clustered_pts,training_indices)
        train_confusion_matrix_kmeans = compute_confusion_matrix(num_clusters, kmeans_clustered_pts,training_indices)
        ###compute the matchings
        matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        matching_GMM = find_matching(train_confusion_matrix_GMM)
        matching_EM = find_matching(train_confusion_matrix_EM)

        correct_EM = 0
        correct_GMM = 0
        correct_KMeans = 0
        for cluster in xrange(num_clusters):
            matched_cluster_EM = matching_EM[cluster]
            matched_cluster_GMM = matching_GMM[cluster]
            matched_cluster_Kmeans = matching_Kmeans[cluster]

            correct_EM += train_confusion_matrix_EM[cluster,matched_cluster_EM]
            correct_GMM += train_confusion_matrix_GMM[cluster,matched_cluster_GMM]
            correct_KMeans += train_confusion_matrix_kmeans[cluster, matched_cluster_Kmeans]
        binary_EM = correct_EM/bt.len(clustered_points)
        binary_GMM = correct_GMM/bt.len(gmm_clustered_pts)
        binary_Kmeans = correct_KMeans/bt.len(kmeans_clustered_pts)

        ##compute the F1 macro scores
        f1_EM_tr = -1#computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1#computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1#computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)

        print "\n\n\n"

        if np.array_equal(old_clustered_points,clustered_points):
            print "\n\n\n\nCONVERGED!!! BREAKING EARLY!!!"
            break
        old_clustered_points = clustered_points
        # end of training

    train_confusion_matrix_EM = compute_confusion_matrix(num_clusters,clustered_points,training_indices)
    train_confusion_matrix_GMM = compute_confusion_matrix(num_clusters,gmm_clustered_pts,training_indices)
    train_confusion_matrix_kmeans = compute_confusion_matrix(num_clusters,clustered_points_kmeans,training_indices)

    f1_EM_tr = -1#computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
    f1_GMM_tr = -1#computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
    f1_kmeans_tr = -1#computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)

    print "\n\n"
    print "TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr

    correct_EM = 0
    correct_GMM = 0
    correct_KMeans = 0
    for cluster in xrange(num_clusters):
        matched_cluster_EM = matching_EM[cluster]
        matched_cluster_GMM = matching_GMM[cluster]
        matched_cluster_Kmeans = matching_Kmeans[cluster]

        correct_EM += train_confusion_matrix_EM[cluster,matched_cluster_EM]
        correct_GMM += train_confusion_matrix_GMM[cluster,matched_cluster_GMM]
        correct_KMeans += train_confusion_matrix_kmeans[cluster, matched_cluster_Kmeans]
        # np.savetxt("computed estimated_matrix cluster =" + str(cluster) + ".csv", train_cluster_inverse[matched_cluster] , delimiter = ",", fmt = "%1.6f")
    binary_EM = correct_EM/num_train_points
    binary_GMM = correct_GMM/num_train_points
    binary_Kmeans = correct_KMeans/num_train_points

    #########################################################
    ##DONE WITH EVERYTHING 
    if compute_BIC:
        bic = computeBIC(num_clusters, m, clustered_points,train_cluster_inverse, empirical_covariances)
        return (clustered_points, train_cluster_inverse, bic)
    return (clustered_points, train_cluster_inverse)

#######################################################################################################################################################################




