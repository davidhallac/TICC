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
#####################PARAMETERS TO PLAY WITH 
window_size = 10
maxIters = 1000 ##number of Iterations of the smoothening + clustering algo
beta = 400 ## Beta parameter
lambda_parameter = 11e-2 ## Lambda regularization parameter
number_of_clusters = 11
threshold = 2e-5##Threshold for plots. Not used in TICC algorithm.
write_out_file = False ##Only if True are any files outputted
seg_len = 300##segment-length : used in confusion matrix computation

##INPUT file location
input_file = "Synthetic Data Matrix rand_seed =[0,1] generated2.csv"

##Folder name to store all the OUPUTS
prefix_string = "data_lambda=" + str(lambda_parameter)+"beta = "+str(beta) + "clusters=" +str(number_of_clusters)+"/"




########################################################

##parameters that are automatically set based upoon above
num_blocks = window_size + 1
switch_penalty = beta## smoothness penalty
lam_sparse = lambda_parameter##sparsity parameter
maxClusters = number_of_clusters+1## Number of clusters + 1
write_out_file = False ##Only if True are any files outputted
num_stacked = num_blocks - 1
##colors used in hexadecimal format
hexadecimal_color_list = ["cc0000","0000ff","003300","33ff00","00ffcc","ffff00","ff9900","ff00ff","cccc66","666666","ffccff","660000","00ff00","ffffff","3399ff","006666","330000","ff0000","cc99ff","b0800f","3bd9eb","ef3e1b"]
##The basic folder to be created
str_NULL = prefix_string


print "lam_sparse", lam_sparse
print "switch_penalty", switch_penalty
print "num_cluster", maxClusters - 1
print "num stacked", num_stacked

Data = np.loadtxt(input_file, delimiter= ",")

print "completed getting the data"
Data_pre = Data
UNNORMALIZED_Data = Data*1000
(m,n) = Data.shape
len_D_total = m
size_blocks = n

##Add an optimization function
# def optimize():

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
	Uses the Viterbi path dynamic programming algorithm
	to compute the optimal cluster assigments

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
		if save_matrix: np.savetxt("estimated_matrix_cluster=" + str(cluster)+".csv",estimated_matrix,delimiter = ",", fmt = "%1.4f")
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
		recall = TP/(TP + FN)
		f1 = (2*precision*recall)/(precision + recall)
		F1_score[cluster] = f1
	return F1_score

def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
	"""
	computes a confusion matrix and returns it
	"""
	seg_len = 200
	true_confusion_matrix = np.zeros([num_clusters,num_clusters])
	for point in xrange(len(clustered_points_algo)):
		cluster = int(clustered_points_algo[point])


		##CASE G: ABBACCCA
		# num = (int(sorted_indices_algo[point]/seg_len) )
		# if num in [0,3,7]:
		# 	true_confusion_matrix[0,cluster] += 1
		# elif num in[1,2]:
		# 	true_confusion_matrix[1,cluster] += 1
		# else:
		# 	true_confusion_matrix[2,cluster] += 1

		##CASE F: ABCBA
		# num = (int(sorted_indices_algo[point]/seg_len))
		# num = min(num, 4-num)
		# true_confusion_matrix[num,cluster] += 1

		#CASE E : ABCABC
		num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
		true_confusion_matrix[num,cluster] += 1

		##CASE D : ABABABAB
		# num = (int(sorted_indices_algo[point]/seg_len) %2)
		# true_confusion_matrix[num,cluster] += 1

		##CASE C: 
		# num = (sorted_indices_algo[point]/seg_len)
		# if num < 15:
		# 	true_confusion_matrix[0,cluster] += 1
		# elif num < 20:
		# 	true_confusion_matrix[1,cluster] += 1
		# else:
		# 	true_confusion_matrix[0,cluster] += 1

		##CASE B : 
		# if num > 4:
		# 	num = 9 - num
		# true_confusion_matrix[num,cluster] += 1

		##CASE A : ABA
		# if sorted_indices_algo[point] < seg_len:
		# 	true_confusion_matrix[0,cluster] += 1

		# elif sorted_indices_algo[point] <3*seg_len:
		# 	true_confusion_matrix[1,cluster] += 1
		# else:
		# 	true_confusion_matrix[0,cluster] += 1

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

def computeNetworkAccuracy(matching,train_cluster_inverse, num_clusters):
	"""
	Takes in the matching for the clusters
	takes the computed clusters
	computes the average F1 score over the network
	"""
	threshold = 1e-2
	f1 = 0
	for cluster in xrange(num_clusters):
		true_cluster_cov = np.loadtxt("Inverse Covariance cluster ="+ str(cluster) +".csv", delimiter = ",")
		matched_cluster = matching[cluster]
		matched_cluster_cov = train_cluster_inverse[matched_cluster] 
		(nrow,ncol) = true_cluster_cov.shape

		out_true = np.zeros([nrow,ncol])
		for i in xrange(nrow):
			for j in xrange(ncol):
				if np.abs(true_cluster_cov[i,j]) > threshold:
					out_true[i,j] = 1
		out_matched = np.zeros([nrow,ncol])
		for i in xrange(nrow):
			for j in xrange(ncol):
				if np.abs(matched_cluster_cov[i,j]) > threshold:
					out_matched[i,j] = 1
		np.savetxt("Network_true_cluster=" +str(cluster) + ".csv",true_cluster_cov, delimiter = ",")
		np.savetxt("Network_matched_cluster=" + str(matched_cluster)+".csv",matched_cluster_cov, delimiter = ",")


		##compute the confusion matrix
		confusion_matrix = np.zeros([2,2])
		for i in xrange(nrow):
			for j in xrange(ncol):
				confusion_matrix[out_true[i,j],out_matched[i,j]] += 1
		f1 += computeF1_macro(confusion_matrix, [0,1],2)
	return f1/num_clusters

############

if not os.path.exists(os.path.dirname(str_NULL)):
    try:
        os.makedirs(os.path.dirname(str_NULL))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def hex_to_rgb(value):
	"""
	Return (red, green, blue) values
	Input is hexadecimal color code: #rrggbb.
	"""
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
	num_clusters = maxClusters - 1

	if iters == 0:
		## Now splitting up stuff 
		## split1 : Training and Test
		## split2 : Training and Test - different clusters
		training_percent = 0.90
		training_idx = np.random.choice(m-num_blocks+1, size=int(m*training_percent),replace = False )
		##Ensure that the first and the last few points are in
		training_idx = list(training_idx)
		if 0 not in training_idx:
			training_idx.append(0)
		if m - num_stacked  not in training_idx:
			training_idx.append(m-num_stacked)
		training_idx = np.array(training_idx)

		sorted_training_idx = sorted(training_idx)
		num_test_points = m - len(training_idx)
		test_idx = []
		##compute the test indices
		for point in xrange(m-num_stacked+1):
			if point not in sorted_training_idx:
				test_idx.append(point)
		sorted_test_idx = sorted(test_idx)
		# np.savetxt("sorted_training.csv", sorted_training_idx, delimiter = ",")
		# np.savetxt("sorted_test.csv", sorted_test_idx, delimiter = ",")

		##Stack the complete data
		complete_Data = np.zeros([m - num_stacked + 1, num_stacked*n])
		len_data = m
		for i in xrange(m - num_stacked + 1):
			idx = i
			for k in xrange(num_stacked):
				if i+k < len_data:
					idx_k = i + k
					complete_Data[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
		# np.savetxt("Complete_Data_stacked_rand_seed="+str(0)+".csv", complete_Data,delimiter=",")

		##Stack the training data
		complete_D_train = np.zeros([len(training_idx), num_stacked*n])
		len_training = len(training_idx)
		for i in xrange(len(sorted_training_idx)):
			idx = sorted_training_idx[i]
			for k in xrange(num_stacked):
				if i+k < len_training:
					idx_k = sorted_training_idx[i+k]
					complete_D_train[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
		# np.savetxt("Data_train_rand_seed="+str(0)+".csv", complete_D_train,delimiter=",")
		##Stack the test  
		complete_D_test = np.zeros([len(test_idx), num_stacked*n])
		len_test = len(test_idx)

		for i in xrange(len(sorted_test_idx)):
			idx = sorted_test_idx[i]
			idx_left = idx -1 
			while idx_left not in sorted_training_idx:
				idx_left -= 1
			point_tr = sorted_training_idx.index(idx_left)
			complete_D_test[i] = complete_D_train[point_tr]
			complete_D_test[i][0:n] = Data[idx][0:n]
		# np.savetxt("Data_test_rand_seed="+str(0)+".csv", complete_D_test,delimiter=",")
		#####INITIALIZATION!!!
		gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type="full")
		gmm.fit(complete_D_train)
		clustered_points = gmm.predict(complete_D_train)
		clustered_points_test = gmm.predict(complete_D_test)
		gmm_clustered_pts_test = gmm.predict(complete_D_test)
		gmm_clustered_pts = clustered_points + 0

		gmm_covariances = gmm.covariances_
		gmm_means = gmm.means_

		##USE K-means
		kmeans = KMeans(n_clusters = num_clusters,random_state = 0).fit(complete_D_train)
		clustered_points_kmeans = kmeans.labels_
		clustered_points_test_kmeans = kmeans.predict(complete_D_test)

		true_confusion_matrix_g = compute_confusion_matrix(num_clusters,gmm_clustered_pts,sorted_training_idx)
		true_confusion_matrix_k = compute_confusion_matrix(num_clusters,clustered_points_kmeans,sorted_training_idx)

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

	for point in range(len(clustered_points_test)):
		cluster = clustered_points_test[point]
		test_clusters[cluster].append(point)
		len_test_clusters[cluster] += 1
		counter +=1 



	##train_clusters holds the indices in complete_D_train 
	##for each of the clusters
	for cluster in xrange(num_clusters):
		if len_train_clusters[cluster] != 0:
			indices = train_clusters[cluster]
			indices_test = test_clusters[cluster]



			D_train = np.zeros([len_train_clusters[cluster],num_stacked*n])
			for i in xrange(len_train_clusters[cluster]):
				point = indices[i]
				D_train[i,:] = complete_D_train[point,:]

			D_test = np.zeros([len_test_clusters[cluster], num_stacked*n])
			for i in xrange(len_test_clusters[cluster]):
				point = indices_test[i]
				D_test[i,:] = complete_D_test[point,:]
			print "stacking Cluster #", cluster,"DONE!!!"
			##Fit a model - OPTIMIZATION	
			size_blocks = n
			probSize = num_stacked * size_blocks
			lamb = np.zeros((probSize,probSize)) + lam_sparse
			S = np.cov(np.transpose(D_train) )

			print "starting the OPTIMIZATION"
			#Set up the Toeplitz graphical lasso problem
			gvx = TGraphVX()
			theta = semidefinite(probSize,name='theta')
			obj = -log_det(theta) + trace(S*theta)
			gvx.AddNode(0, obj)
			gvx.AddNode(1)
			dummy = Variable(1)
			gvx.AddEdge(0,1, Objective = lamb*dummy + num_stacked*dummy + size_blocks*dummy)
			
			##solve using customized ADMM solver
			gvx.Solve(Verbose=False, MaxIters=1000, Rho = 1, EpsAbs = 1e-6, EpsRel = 1e-6)


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
			cluster_mean_info[num_clusters,cluster] = np.mean(D_train, axis = 0)[(num_stacked-1)*n:num_stacked*n].reshape([1,n])
			cluster_mean_stacked_info[num_clusters,cluster] = np.mean(D_train,axis=0)
			train_cluster_inverse[cluster] = X2

	cluster_norms = list(np.zeros(num_clusters))

	# for cluster in xrange(num_clusters):
	# 	print "length of the cluster ", cluster,"------>", len_train_clusters[cluster]
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
			##Add a point to the cluster
			while len_train_clusters[sorted_cluster_norms[counter][1]] == 0:
				# print "counter is:", counter
				counter += 1
				counter = counter % num_clusters
				# print "counter is:", counter

			cluster_selected = sorted_cluster_norms[counter][1]
			# print "cluster that is zero is:", cluster, "selected cluster instead is:", cluster_selected
			break_flag = False
			while not break_flag:
				point_num = random.randint(0,len(clustered_points))
				if clustered_points[point_num] == cluster_selected:
					clustered_points[point_num] = cluster
					computed_covariance[num_clusters,cluster] = old_computed_covariance[num_clusters,cluster_selected]
					cluster_mean_stacked_info[num_clusters,cluster] = complete_D_train[point_num,:]
					cluster_mean_info[num_clusters,cluster] = complete_D_train[point,:][(num_stacked-1)*n:num_stacked*n]
					break_flag = True
			counter += 1

	old_train_clusters = train_clusters
	old_computed_covariance = computed_covariance



	##Code -----------------------SMOOTHENING
	##For each point compute the LLE 
	print "beginning with the DP - smoothening ALGORITHM"

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
				inv_cov_matrix = np.linalg.inv(cov_matrix)
				log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
				lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
				LLE_all_points_clusters[point,cluster] = lle
	
	##Update cluster points - using dynamic programming smoothening
	clustered_points = updateClusters(LLE_all_points_clusters,switch_penalty = switch_penalty)
	print "\ncompleted smoothening algorithm"
	print "\n\nprinting the length of points in each cluster"
	for cluster in xrange(num_clusters):
		print "length of cluster #", cluster, "-------->", sum([x== cluster for x in clustered_points])
	true_confusion_matrix = np.zeros([num_clusters,num_clusters])

	##Save a figure of segmentation
	# print "length of clustered_points", len(clustered_points)
	# print "length of sorted training_idx", len(sorted_training_idx)
	plt.figure()
	plt.plot(sorted_training_idx[0:len(clustered_points)],clustered_points,color = "r")#,marker = ".",s =100)
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("TRAINING_EM_lam_sparse="+str(lam_sparse) + "switch_penalty = " + str(switch_penalty) + ".jpg")
	plt.close("all")

	plt.figure()
	plt.plot(sorted_training_idx[0:len(gmm_clustered_pts)],gmm_clustered_pts,color = "r")#,marker=".",s=100)
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("TRAINING_GMM_lam_sparse="+str(lam_sparse) + "switch_penalty =" + str(switch_penalty)+ ".jpg")
	plt.close("all")
	# print "Done writing the figure"

	# print "Done writing the figure"

	true_confusion_matrix = compute_confusion_matrix(num_clusters,clustered_points,sorted_training_idx)
	# print "TRAINING TRUE confusion MATRIX:\n", true_confusion_matrix

	####TEST SETS STUFF
	### The closest point in training set is the cluster
	### LLE + swtiching_penalty
	clustered_test = np.zeros(len(clustered_points_test))
	for point in xrange(len(clustered_points_test)):
		idx = sorted_test_idx[point]
		##Get the 2 closest points from training
		idx1 = idx + 1
		while (idx1 not in sorted_training_idx and idx1 < m):
			idx1 += 1
		idx2 = idx -1 
		while (idx2 not in sorted_training_idx and idx2 > -1):
			idx2 -= 1
		if idx1 == m or idx2 == -1:
			print "idx1 :", idx1 == m
			print "idx2 :", idx2 == -1
			print "point is:", point
			print "idx of the point is:", idx
			print "control should NOT reach here!!!!!!!"
			break
		vals = np.zeros(num_clusters)
		right_clust = clustered_points[sorted_training_idx.index(idx1)]
		left_clust = clustered_points[sorted_training_idx.index(idx2)]
		point_tr = sorted_training_idx.index(idx2)
		data_tt = complete_D_train[point_tr,:]
		data_tt[0:n] = Data[idx,:]#complete_D_test[point,0:num_stacked] 

		for cluster in xrange(num_clusters):
			cluster_mean = cluster_mean_info[num_clusters,cluster] 
			cluster_mean_stacked = cluster_mean_stacked_info[num_clusters,cluster] 

			x = data_tt - cluster_mean_stacked[0:(num_blocks-1)*n]
			cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
			inv_cov_matrix = np.linalg.inv(cov_matrix)
			log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
			lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
			vals[cluster] = lle + switch_penalty*(cluster !=left_clust ) + switch_penalty*(cluster != right_clust )
		out = np.argmin(vals)
		clustered_test[point] = out

	plt.figure()
	plt.plot(sorted_test_idx[0:len(clustered_test)],clustered_test,color = "r")#,marker = ".", s= 100)
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("TEST_EM_lam_sparse="+str(lam_sparse) +"switch_penalty="+str(switch_penalty)+ ".jpg")
	plt.close("all")
	# print "done writing"


	##GMM - TEST PREDICTIONS
	clustered_test_gmm = np.zeros(len(clustered_points_test))
	for point in xrange(len(clustered_points_test)):
		idx = sorted_test_idx[point]
		##Get the 2 closest points from training
		idx1 = idx + 1
		while (idx1 not in sorted_training_idx and idx1 < m):
			idx1 += 1
		idx2 = idx -1 
		while (idx2 not in sorted_training_idx and idx2 > -1):
			idx2 -= 1
		if idx1 == m or idx2 == -1:
			print "idx1 :", idx1 == m
			print "idx2 :", idx2 == -1
			print "point is:", point
			print "idx of the point is:", idx

			print "SOMETHING WRONG!!!!!!!"
			break
		vals = np.zeros(num_clusters)
		right_clust = gmm_clustered_pts[sorted_training_idx.index(idx1)]
		left_clust = gmm_clustered_pts[sorted_training_idx.index(idx2)]
		point_tr = sorted_training_idx.index(idx2)
		data_tt = complete_D_train[point_tr,:]
		data_tt[0:n] = Data[idx,:]#complete_D_test[point,0:num_stacked] 

		for cluster in xrange(num_clusters):
			cluster_mean_stacked = gmm_means[cluster] 

			x = data_tt - cluster_mean_stacked[0:(num_blocks-1)*n]
			cov_matrix = gmm_covariances[cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
			inv_cov_matrix = np.linalg.inv(cov_matrix)
			log_det_cov = np.log(np.linalg.det(cov_matrix))
			lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
			vals[cluster] = lle #+ switch_penalty*(cluster !=left_clust ) + switch_penalty*(cluster != right_clust )
		out = np.argmin(vals)
		clustered_test_gmm[point] = out

	plt.figure()
	plt.plot(sorted_test_idx[0:len(clustered_test_gmm)],clustered_test_gmm,color = "r")#,marker = ".", s= 100)
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("TEST_GMM_lam_sparse="+str(lam_sparse) + ".jpg")
	plt.close("all")
	# print "done writing"

	plt.figure()
	plt.plot(sorted_test_idx[0:len(clustered_points_test_kmeans)],clustered_points_test_kmeans,color = "r")#,marker = ".", s= 100)
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("TEST_Modified_KMEANS_NEW_lam_sparse="+str(lam_sparse) + ".jpg")
	plt.close("all")
	# print "done writing"

	##Segment length
	seg_len = 50
	true_confusion_matrix_EM = compute_confusion_matrix(num_clusters,clustered_test,sorted_test_idx)
	true_confusion_matrix_GMM = compute_confusion_matrix(num_clusters,gmm_clustered_pts_test,sorted_test_idx)
	true_confusion_matrix_kmeans = compute_confusion_matrix(num_clusters,clustered_points_test_kmeans,sorted_test_idx)


	true_answers = np.zeros(len(clustered_points))
	for point in xrange(len(clustered_points)):
		num = int(sorted_training_idx[point]/25.0)
		if num <10 :
			cluster = 0
		elif num < 20:
			cluster = 1
		else:
			cluster = 0
		true_answers[point] = cluster

	# print "length of sorted_test_idx", len(sorted_test_idx)
	# print "length of true answers", len(true_answers)
	# print "new length", len(sorted_training_idx[0:len(clustered_points)])
	plt.figure()
	plt.plot(sorted_training_idx[0:len(clustered_points)],true_answers,color = "k")
	plt.ylim((-0.5,num_clusters + 0.5))
	# plt.savefig("True Output modifed lam_sparse=" + str(lam_sparse)+ ".jpg")
	plt.close("all")
	binary_EM = (true_confusion_matrix_EM[0,0] + true_confusion_matrix_EM[1,1])/len(clustered_points_test)
	binary_EM = np.max([binary_EM,1 -binary_EM])
	# print "EM is -------->",binary_EM
	binary_GMM = (true_confusion_matrix_GMM[0,0] + true_confusion_matrix_GMM[1,1])/len(clustered_points_test)
	binary_GMM = np.max([binary_GMM,1-binary_GMM])

	binary_Kmeans = (true_confusion_matrix_kmeans[0,0] + true_confusion_matrix_kmeans[1,1])/len(clustered_points_test)
	binary_Kmeans = np.max([binary_Kmeans,1-binary_Kmeans])


	# print "TEST EM TRUE confusion MATRIX:\n", true_confusion_matrix_EM
	# print "TEST GMM TRUE confusion MATRIX:\n", true_confusion_matrix_GMM					
	# print "TEST KMEANS TRUE CONFUSION MATRIX:\n", true_confusion_matrix_kmeans

	##Create the F1 score from the graphs from k-means and GMM
	##Get the train and test points
	train_inverse_covariance_kmeans = {}
	train_inverse_covariance_gmm = {}

	counter = 0
	for cluster in xrange(num_clusters):
		##GMM
		out = [(x == cluster) for x in gmm_clustered_pts]
		len_cluster = sum(out)
		D_train = np.zeros([len_cluster,num_stacked*n])
		counter = 0
		for point in xrange(len(gmm_clustered_pts)):
			if gmm_clustered_pts[point] == cluster:
				D_train[counter,:] = complete_D_train[point,:]
				counter += 1

		train_inverse_covariance_gmm[cluster] = np.linalg.inv(gmm_covariances[cluster])

		##Kmeans
		out = [(x == cluster) for x in clustered_points_kmeans]
		len_cluster = sum(out)
		D_train = np.zeros([len_cluster,num_stacked*n])
		counter2 = 0
		for point in xrange(len(clustered_points_kmeans)):
			if clustered_points_kmeans[point] == cluster:
				D_train[counter2,:] = complete_D_train[point,:]
				counter2 += 1

		train_inverse_covariance_kmeans[cluster] = X2

	##Using inverses from kmeans, GMM and EM
	threshold = 1e-5
	##Thresholding
	threshold_kmeans = {}
	threshold_GMM = {}
	threshold_EM = {}
	threshold_actual = {}

	##Change this to add thresholding function
	##Kmeans - thresholding
	for cluster in xrange(num_clusters):
	    out = np.zeros(train_inverse_covariance_kmeans[0].shape, dtype = np.int)
	    A = train_inverse_covariance_kmeans[cluster]
	    for i in xrange(out.shape[0]):
	        for j in xrange(out.shape[1]):
				if (np.abs(A[i,j]) > threshold):
					out[i,j] = 1
		threshold_kmeans[cluster] = out

	##GMM - thresholding
	for cluster in xrange(num_clusters):
		out = np.zeros(train_inverse_covariance_gmm[0].shape, dtype = np.int)
		A = train_inverse_covariance_gmm[cluster]
		for i in xrange(out.shape[0]):
			for j in xrange(out.shape[1]):
				if np.abs(A[i,j]) > threshold:
					out[i,j] = 1
		threshold_GMM[cluster] = out

    ## EM - thresholding
	for cluster in xrange(num_clusters):
		out = np.zeros(train_inverse_covariance_gmm[0].shape, dtype = np.int)
		A = train_cluster_inverse[cluster]
		for i in xrange(out.shape[0]):
			for j in xrange(out.shape[1]):
				if np.abs(A[i,j]) > threshold:
					out[i,j] = 1
		threshold_EM[cluster] = out

    ##compute the matching
    ##Assume its a 2x2 matrix?
	actual_clusters = {}
	for cluster in xrange(num_clusters):
		actual_clusters[cluster] = np.loadtxt("Inverse Covariance cluster =" + str(cluster)+".csv", delimiter = ",")

    ##compute the appropriate matching
	matching_Kmeans = find_matching(true_confusion_matrix_kmeans)
	matching_GMM = find_matching(true_confusion_matrix_GMM)
	matching_EM = find_matching(true_confusion_matrix_EM)

	correct_EM = 0
	correct_GMM = 0
	correct_KMeans = 0
	for cluster in xrange(num_clusters):
		matched_cluster_EM = matching_EM[cluster]
		matched_cluster_GMM = matching_GMM[cluster]
		matched_cluster_Kmeans = matching_Kmeans[cluster]

		correct_EM += true_confusion_matrix_EM[cluster,matched_cluster_EM]
		correct_GMM += true_confusion_matrix_GMM[cluster,matched_cluster_GMM]
		correct_KMeans += true_confusion_matrix_kmeans[cluster, matched_cluster_Kmeans]
		# np.savetxt("computed estimated_matrix cluster =" + str(cluster) + ".csv", train_cluster_inverse[matched_cluster] , delimiter = ",", fmt = "%1.6f")
	binary_EM = correct_EM/len(clustered_points_test)
	binary_GMM = correct_GMM/len(clustered_points_test)
	binary_Kmeans = correct_KMeans/len(clustered_points_test)


	# print "\n\nKMEANS"
	# print "true confusion_matrix", true_confusion_matrix_kmeans
	# print "matching", matching_Kmeans

	# print "\n\nGMM"
	# print "true_confusion_matrix_GMM", true_confusion_matrix_GMM
	# print "matching GMM", matching_GMM

	# print "\n\nEM"
	# print "true confusion_matrix", true_confusion_matrix_EM
	# print "matching ", matching_EM


	print "\n\n\n"
	# print "The F1 scores is:", F1_EM,F1_GMM, F1_Kmeans
	# print "The binary accuracy", binary_EM, binary_GMM, binary_Kmeans

	if np.array_equal(old_clustered_points,clustered_points):
		print "\n\n\n\nCONVERGED!!! BREAKING EARLY!!!"
		break
	old_clustered_points = clustered_points

##Training confusion matrix
train_confusion_matrix_EM = compute_confusion_matrix(num_clusters,clustered_points,sorted_training_idx)
train_confusion_matrix_GMM = compute_confusion_matrix(num_clusters,gmm_clustered_pts,sorted_training_idx)
train_confusion_matrix_kmeans = compute_confusion_matrix(num_clusters,clustered_points_kmeans,sorted_training_idx)
test_confusion_matrix_EM = compute_confusion_matrix(num_clusters,clustered_test,sorted_test_idx)

out = computeNetworkAccuracy(matching_EM, train_cluster_inverse,num_clusters)

print "the network accuracy F1 score is:", out


f1_EM_tr = computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
f1_GMM_tr = computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
f1_kmeans_tr = computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
f1_EM_test = computeF1_macro(test_confusion_matrix_EM,matching_EM,num_clusters)
# print "The TEST binary accuracy", binary_EM, binary_GMM, binary_Kmeans
# print "\n\n"
# print "TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr
# print "TEST F1 score:", f1_EM_test
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
binary_EM = correct_EM/len(training_idx)
binary_GMM = correct_GMM/len(training_idx)
binary_Kmeans = correct_KMeans/len(training_idx)

print "\n\n"
print "\n\n\n"
# print "The TRAINING binary accuracy", binary_EM, binary_GMM, binary_Kmeans
# print "lam_sparse", lam_sparse
# print "switch_penalty", switch_penalty
# print "num_cluster", maxClusters - 1
# print "num stacked", num_stacked








