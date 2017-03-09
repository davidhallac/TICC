from cvxpy import *
import numpy as np 
import time, collections, os, errno, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Visualization_function import visualize
from solveCrossTime import *
from scipy import stats

from sklearn import mixture
from sklearn import covariance
import sklearn, random
from sklearn.cluster import KMeans

import pandas as pd
pd.set_option('display.max_columns', 500)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.random.seed(102)
##PARAMETERS TO PLAY WITH 
cvxpy_iterations = 15300
num_blocks = 3
maxIters = 1 ##number of Iterations of the smoothening + clustering algo
adherence_penalty =50000
switch_penalty = 200
seg_len = 400
lam_sparse = 1e-2

rep_iterations = 2
maxClusters = 10
threshold = 10.5e-5
write_out_file = False ##Only if True are any files outputted
scaling_time = 3
num_stacked = num_blocks - 1

#num_clusters = 4
hexadecimal_color_list = ["cc0000","0000ff","003300","33ff00","00ffcc","ffff00","ff9900","ff00ff","cccc66","666666","ffccff","660000","00ff00","ffffff","3399ff","006666","330000","ff0000","cc99ff","b0800f","3bd9eb","ef3e1b"]


print "num_cluster", maxClusters - 1
print "num_blocks:", num_blocks


Data = 1000*np.random.rand(1e7,50)
Data_pre = Data
UNNORMALIZED_Data = Data*1000
(m,n) = Data.shape
len_D_total = m
size_blocks = n
# def optimize(emp_cov = No)
print "Length of data:", m
print "completed getting the data"

def upper2Full(a, eps = 0):
    ind = (a<eps)&(a>-eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))             
    return A   


def updateClusters(LLE_node_vals,switch_penalty = 1):
	"""
	Takes in LLE_node_vals matrix and computes the path that minimizes
	the total cost over the path
	Note the LLE's are negative of the true LLE's actually!!!!!

	Note: switch penalty > 0
	"""
	(T,num_clusters) = LLE_node_vals.shape
	future_cost_vals = np.zeros(LLE_node_vals.shape)

	##compute future costs
	for i in xrange(T-2,-1,-1):
		j = i+1
		indicator = np.zeros(num_clusters)
		future_costs = future_cost_vals[j,:]
		lle_vals = LLE_node_vals[j,:]
		for cluster in xrange(num_clusters):
			total_vals = future_costs + lle_vals + switch_penalty
			total_vals[cluster] -= switch_penalty
			future_cost_vals[i,cluster] = np.min(total_vals)

	##compute the best path
	path = np.zeros(T)

	##the first location
	curr_location = np.argmin(future_cost_vals[0,:] + LLE_node_vals[0,:])
	path[0] = curr_location
	DP_start2 = time.time()
	##compute the path
	for i in xrange(T-1):
		j = i+1
		future_costs = future_cost_vals[j,:]
		lle_vals = LLE_node_vals[j,:]
		total_vals = future_costs + lle_vals + switch_penalty
		total_vals[int(path[i])] -= switch_penalty

		path[i+1] = np.argmin(total_vals)

	##return the computed path
	return path

def find_matching(confusion_matrix):
	"""
	returns the perfect matching
	"""
	_,n = confusion_matrix.shape
	path = []
	for i in xrange(n):
		max_val = -1e10
		max_ind = -1
		for j in xrange(n):
			if j in path:
				pass
			else:
				temp = confusion_matrix[i,j]
				if temp > max_val:
					max_val = temp
					max_ind = j
		path.append(max_ind)
	return path

def computeF1Score_delete(num_cluster,matching_algo,actual_clusters,threshold_algo,save_matrix = False):
	"""
	computes the F1 scores and returns a list of values
	"""
	F1_score = np.zeros(num_cluster)
	for cluster in xrange(num_cluster):
		matched_cluster = matching_algo[cluster]
		true_matrix = actual_clusters[cluster]
		estimated_matrix = threshold_algo[matched_cluster]
		TP = 0
		TN = 0
		FP = 0
		FN = 0
		for i in xrange(num_stacked*n):
			for j in xrange(num_stacked*n):
				if estimated_matrix[i,j] == 1 and true_matrix[i,j] != 0:
					TP += 1.0
				elif estimated_matrix[i,j] == 0 and true_matrix[i,j] == 0:
					TN += 1.0
				elif estimated_matrix[i,j] == 1 and true_matrix[i,j] == 0:
					FP += 1.0
				else:
					FN += 1.0
		precision = (TP)/(TP + FP)
		print "cluster #", cluster
		print "TP,TN,FP,FN---------->", (TP,TN,FP,FN)
		recall = TP/(TP + FN)
		f1 = (2*precision*recall)/(precision + recall)
		F1_score[cluster] = f1
	return F1_score

def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
	"""
	computes a confusion matrix and returns it
	"""
	seg_len = 50
	true_confusion_matrix = np.zeros([num_clusters,num_clusters])
	for point in xrange(len(clustered_points_algo)):
		cluster = clustered_points_algo[point]

		#CASE E : ABCABC
		num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
		true_confusion_matrix[num,cluster] += 1

	return true_confusion_matrix

def computeF1_macro(confusion_matrix,matching, num_clusters):
	"""
	computes the macro F1 score
	confusion matrix : requres permutation
	matching according to which matrix must be permuted
	"""
	##Permute the matrix columns
	permuted_confusion_matrix = np.zeros([num_clusters,num_clusters])
	for cluster in xrange(num_clusters):
		matched_cluster = matching[cluster]
 		permuted_confusion_matrix[:,cluster] = confusion_matrix[:,matched_cluster]
 	##Compute the F1 score for every cluster
 	F1_score = 0
 	for cluster in xrange(num_clusters):
 		TP = permuted_confusion_matrix[cluster,cluster]
 		FP = np.sum(permuted_confusion_matrix[:,cluster]) - TP
 		FN = np.sum(permuted_confusion_matrix[cluster,:]) - TP
 		precision = TP/(TP + FP)
 		recall = TP/(TP + FN)
 		f1 = stats.hmean([precision,recall])
 		F1_score += f1
 	F1_score /= num_clusters
 	return F1_score

############
##The basic folder to be created
str_NULL = "VW_data_lam_sparse=" + str(lam_sparse)+"cvxpy_iterations"+str(cvxpy_iterations) + "maxClusters=" +str(maxClusters)+"/"

if not os.path.exists(os.path.dirname(str_NULL)):
    try:
        os.makedirs(os.path.dirname(str_NULL))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def hex_to_rgb(value):
	"""Return (red, green, blue) for the color given as #rrggbb."""
	lv = len(value)
	out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
	out = tuple([x/256.0 for x in out])
	return out
color_list = []
for hex_color in hexadecimal_color_list:
	rgb_color = hex_to_rgb(hex_color)
	color_list.append(rgb_color)
colors = color_list


train_cluster_inverse = {}
log_det_values = {}
computed_covariance = {}
cluster_mean_info = {}
cluster_mean_stacked_info = {}
old_clustered_points = np.zeros(10)
for iters in xrange(maxIters):
	print "\n\n\nITERATION ###", iters
	iter_start =  time.time()
	num_clusters = maxClusters - 1

	if iters == 0:
		## Now splitting up stuff 
		## split1 : Training and Test
		## split2 : Training and Test - different clusters

		time1 = time.time()
		training_percent = 1
		training_idx = np.random.choice(m-num_blocks+1, size=int((m-num_stacked)*training_percent),replace = False )
		##Ensure that the first and the last few points are in
		training_idx = list(training_idx)
		if 0 not in training_idx:
			training_idx.append(0)
		if m - num_stacked  not in training_idx:
			training_idx.append(m-num_stacked)

		training_idx = np.array(training_idx)
		sorted_training_idx = range(len(training_idx))
		# sorted_training_idx = sorted(training_idx)
		##compute the test indices

		##Stack the complete data
		print "stacking training data"
		time_stack_start = time.time()
		complete_Data = np.zeros([m - num_stacked + 1, num_stacked*n])
		len_data = m
		for i in xrange(m - num_stacked + 1):
			idx = i
			for k in xrange(num_stacked):
				if i+k < len_data:
					idx_k = i + k
					complete_Data[i][k*n:(k+1)*n] =  Data[idx_k][0:n]

		##Stack the training data
		complete_D_train = np.zeros([len(training_idx), num_stacked*n])
		len_training = len(training_idx)
		for i in xrange(len(sorted_training_idx)):
			idx = sorted_training_idx[i]
			for k in xrange(num_stacked):
				if i+k < len_training:
					idx_k = sorted_training_idx[i+k]
					complete_D_train[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
		print "completed stacking data"
		time_stack_end = time.time()
		print "It took :", time_stack_end - time_stack_start, "to stack the data"

		actual_start = time.time()
		print "starting to count time from here"
		#####INITIALIZATION!!!
		gmm_start = time.time()
		gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type="full")
		clustered_points = np.random.choice(num_clusters, size = complete_D_train.shape[0], replace = True)#gmm.predict(complete_D_train)
		gmm_clustered_pts = clustered_points + 0

		gmm_end = time.time()
		print "intialization completed"
		print "gmm took:", gmm_end - gmm_start

		##USE K-means
		print "running K means"
		kmeans_start = time.time()

		clustered_points_kmeans = np.random.choice(num_clusters, size = complete_D_train.shape[0], replace = True)#kmeans.labels_
		kmeans_clustered_pts = np.random.choice(num_clusters, size = complete_D_train.shape[0], replace = True)#kmeans.labels_
		kmeans_end = time.time()
		print "k means took:", kmeans_end - kmeans_start
	
	##Get the train and test points
	train_clusters = collections.defaultdict(list)
	test_clusters = collections.defaultdict(list)
	len_train_clusters = collections.defaultdict(int)
	len_test_clusters = collections.defaultdict(int)

	counter = 0
	for point in range(len(clustered_points)):
		cluster = clustered_points[point]
		train_clusters[cluster].append(point)
		len_train_clusters[cluster] += 1
		counter +=1 



	##train_clusters holds the indices in complete_D_train 
	##for each of the clusters
	for cluster in xrange(num_clusters):
		if len_train_clusters[cluster] != 0:
			indices = train_clusters[cluster]

			print "\n\n\nstarting with cluster:", cluster
			cluster_start = time.time()
			S = np.cov(np.transpose(complete_D_train[indices,:]))
			S_mean = np.mean(complete_D_train[indices,:], axis = 0)
			print "it took about", time.time() - cluster_start,"to get all the training data"
			size_blocks = n
			probSize = num_stacked * size_blocks
			lamb = np.zeros((probSize,probSize)) + lam_sparse
			cov_start = time.time()

			#OPTIMIZATION CODE
			print "starting the OPTIMIZATION"
			opt_start = time.time()
			gvx = TGraphVX()
			theta = semidefinite(probSize,name='theta')
			obj = -log_det(theta) + trace(S*theta)
			gvx.AddNode(0, obj)
			gvx.AddNode(1)
			dummy = Variable(1)
			gvx.AddEdge(0,1, Objective = lamb*dummy + num_stacked*dummy + size_blocks*dummy)
			gvx.Solve(Verbose=False, MaxIters=1000, Rho = 1, EpsAbs = 1e-6, EpsRel = 1e-6)

			opt_end = time.time() 
			print "SOLVER took:", opt_end - opt_start

			#THIS IS THE SOLUTION
			val = gvx.GetNodeValue(0,'theta')
			S_est = upper2Full(val, 0)
			X2 = S_est
			u, _ = np.linalg.eig(S_est)
			cov_out = np.linalg.inv(X2)

			inv_matrix = cov_out

			##Store the log-det, covariance, inverse-covariance, cluster means, stacked means
			log_det_values[num_clusters, cluster] = np.log(np.linalg.det(cov_out))
			computed_covariance[num_clusters,cluster] = cov_out
			cluster_mean_info[num_clusters,cluster] = S_mean[(num_stacked-1)*n:num_stacked*n].reshape([1,n])
			cluster_mean_stacked_info[num_clusters,cluster] = S_mean
			train_cluster_inverse[cluster] = X2

	cluster_norms = list(np.zeros(num_clusters))

	##Computing the norms
	if iters != 0:
		for cluster in xrange(num_clusters):
			cluster_norms[cluster] = (np.linalg.norm(old_computed_covariance[num_clusters,cluster]),cluster)
		sorted_cluster_norms = sorted(cluster_norms,reverse = True)

	##Add a point to the empty clusters 
	##Assumption more non empty clusters than empty ones
	counter = 0
	for cluster in xrange(num_clusters):
		if len_train_clusters[cluster] == 0:
			print "doing robustness stuff one cluster has zero points"
			robust_start = time.time()
			##Add a point to the cluster
			while len_train_clusters[sorted_cluster_norms[counter][1]] == 0:
				counter += 1
				counter = counter % num_clusters

			cluster_selected = sorted_cluster_norms[counter][1]
			break_flag = False
			while not break_flag:
				point_num = random.randint(0,len(clustered_points))
				if clustered_points[point_num] == cluster_selected:
					clustered_points[point_num] = cluster
					# print "old covariances shape", old_computed_covariance[cluster_selected].shape
					computed_covariance[num_clusters,cluster] = old_computed_covariance[num_clusters,cluster_selected]
					cluster_mean_stacked_info[num_clusters,cluster] = complete_D_train[point_num,:]
					cluster_mean_info[num_clusters,cluster] = complete_D_train[point,:][(num_stacked-1)*n:num_stacked*n]
					break_flag = True
			print "done adding stuff"
			counter += 1
			print "it took:",  time.time() - robust_start

	old_train_clusters = train_clusters
	old_computed_covariance = computed_covariance
	print "\n\n\n\n"
	cache_start = time.time()
	inv_matrix_cluster ={}
	log_det_cov_cluster = {}
	for cluster in xrange(num_clusters):
		cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
		log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
		inv_matrix_cluster[cluster] = np.linalg.inv(cov_matrix)
		log_det_cov_cluster[cluster] = log_det_cov
 	##Code -----------------------SMOOTHENING
	##For each point compute the LLE 
	print "\n\n"
	LLE_start =  time.time()

	LLE_all_points_clusters = np.zeros([len(clustered_points),num_clusters])
	for point in xrange(len(clustered_points)):
		# print "Point #", point
		if point + num_stacked-1 < complete_D_train.shape[0]:
			for cluster in xrange(num_clusters):
				# print "\nCLuster#", cluster
				cluster_mean = cluster_mean_info[num_clusters,cluster] 
				cluster_mean_stacked = cluster_mean_stacked_info[num_clusters,cluster] 

				x = complete_D_train[point,:] - cluster_mean_stacked[0:(num_blocks-1)*n]
				cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
				inv_cov_matrix = inv_matrix_cluster[cluster]#np.linalg.inv(cov_matrix)
				log_det_cov = log_det_cov_cluster[cluster]#np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
				lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
				LLE_all_points_clusters[point,cluster] = lle
	##Update cluster points - using NEW smoothening
	DP_start = time.time()
	clustered_points = updateClusters(LLE_all_points_clusters,switch_penalty = switch_penalty)
	print "Total time for the DYNAMIC PROGRAMMING ALGORITHM:", time.time() - LLE_start



	##compute the F1 macro scores
	f1_EM_tr = 0#computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
	f1_GMM_tr = 0#computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
	f1_kmeans_tr = 0#computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)

	print "\n\n\n"
	print "\n\n\nThe TOTAL ITERATION TIME (from GMM Initialization):", time.time() - actual_start
	print "\n\n"
	if np.array_equal(old_clustered_points,clustered_points):
		print "\n\n\n\nCONVERGED!!! BREAKING EARLY!!!"
		break
	old_clustered_points = clustered_points





