from cvxpy import *
import numpy as np 
import time, collections, os, errno, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Visualization_function import visualize
import solveCrossTime as SCT
from sklearn import mixture
from sklearn import covariance
import sklearn, random
import pandas as pd
pd.set_option('display.max_columns', 500)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.random.seed(2)

#####################PARAMETERS TO PLAY WITH 
window_size = 10
maxIters = 2 ##number of Iterations of the smoothening + clustering algo
beta = 400 ## Beta parameter
lambda_parameter = 5e-3 ## Lambda regularization parameter
number_of_clusters = 11
threshold = 2e-5##
write_out_file = False ##Only if True are any files outputted
scaling_time = 10
prefix_string = "data_lambda=" + str(lambda_parameter)+"beta = "+str(beta) + "clusters=" +str(number_of_clusters)+"/"

##Enter the location of data file
data_pre = pd.read_csv('data.tsv', sep ='\t',low_memory = False)
print "\n\ncompleted getting the data"
###########################################################

##set other parameter values
maxClusters = number_of_clusters + 1
seg_len = 100
num_blocks = window_size
num_stacked = num_blocks - 1
str_NULL = prefix_string
switch_penalty = beta
lam_sparse = lambda_parameter
print "\n\n\nthe set parameters are:"
print "THRESHOLD IS:", threshold
print "lam_sparse", lam_sparse
print "switch_penalty", switch_penalty
print "num_cluster", maxClusters-1


######### Get Date into proper format

##111,...1505,75 ---> Break pedal, Y -accn, sw - angle, x - accn, Vel, RPM,gas pedal
out1 = data_pre.iloc[:,[111,1433,1498,1432,1471,1505,75,1613,1614]]
out2 = out1.fillna(method = 'ffill')
UNNORMALIZED_Data = np.array(out2.iloc[165:,:])
out2 = (out2 - out2.mean())/(out2.max() - out2.min()) 
Data_pre = UNNORMALIZED_Data
Data = np.array(out2)[165:,:7]#Data_pre[:,:6]
(m,n) = Data.shape
len_D_total = m


##color list used here
hexadecimal_color_list = ["000099","ff00ff","00ff00","663300","996633","66ffff","3333cc","660066","66ccff","cc0000","0000ff","003300","33ff00","00ffcc","ffff00","ff9900","ff00ff","cccc66","666666","ffccff","660000","00ff00","ffffff","3399ff","006666","330000","ff0000","cc99ff","b0800f","3bd9eb","ef3e1b"]


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

def computeF1Score(num_cluster,matching_algo,actual_clusters,threshold_algo,save_matrix = False):
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
		recall = TP/(TP + FN)
		f1 = (2*precision*recall)/(precision + recall)
		F1_score[cluster] = f1
	return F1_score

def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
	"""
	computes a confusion matrix and returns it
	"""
	true_confusion_matrix = np.zeros([num_clusters,num_clusters])
	for point in xrange(len(clustered_points_algo)):
		cluster = clustered_points_algo[point]
		if sorted_indices_algo[point] < seg_len:
			true_confusion_matrix[0,cluster] += 1

		elif sorted_indices_algo[point] <3*seg_len:
			true_confusion_matrix[1,cluster] += 1
		else:
			true_confusion_matrix[0,cluster] += 1

	return true_confusion_matrix

def compute_BIC_score(actual_clusters, emprical_cov, LLE_all_points_clusters,clustered_points, sorted_training_idx):
	"""
	compute BIC score for the clusters 
	"""
	(T, num_clusters) = LLE_all_points_clusters.shape
	point_lle = 0
	mod_lle = 0
	##Compute the total LLE
	for idx in xrange(len(clustered_points)):
		point = sorted_training_idx[idx]
		if point < T:
			point_lle -= LLE_all_points_clusters[point,clustered_points[point]]
	for cluster in xrange(num_clusters):
		mod_lle += np.log(np.linalg.det(actual_clusters[cluster])) - np.trace(np.dot(empirical_cov[cluster], actual_clusters[cluster]))
	tot_lle = mod_lle/num_clusters#point_lle + mod_lle
	threshold = 2e-5
	nonzero_params = 0
	for cluster in xrange(num_clusters):
		nonzero_params += np.sum(np.sum(np.abs(actual_clusters[cluster] > threshold)))
	BIC = nonzero_params*np.log(T) - 2*tot_lle

	return BIC



############

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
for iters in xrange(maxIters):
	print "\n\n\nITERATION ###", iters
	num_clusters = maxClusters - 1

	if iters == 0:
		## Now splitting up stuff 
		## split1 : Training and Test
		## split2 : Training and Test - different clusters
		training_percent = 0.99
		training_idx = np.random.choice(m-num_blocks+1, size=int(m*training_percent),replace = False )
		sorted_training_idx = sorted(training_idx)
		num_test_points = m - len(training_idx)
		test_idx = []
		##compute the test indices
		for point in xrange(m):
			if point not in sorted_training_idx:
				test_idx.append(point)
		sorted_test_idx = sorted(test_idx)

		##Stack the complete data
		complete_Data = np.zeros([m - num_stacked + 1, num_stacked*n])
		len_data = m
		for i in xrange(m - num_stacked + 1):
			idx = i
			for k in xrange(num_stacked):
				if i+k < len_data:
					idx_k = i + k
					complete_Data[i][k*n:(k+1)*n] =  Data[idx_k][0:n]

		##Stack the training data
		complete_D_train = np.zeros([len(training_idx) - num_stacked + 1, num_stacked*n])
		len_training = len(training_idx)
		for i in xrange(len(sorted_training_idx) - num_stacked + 1):
			idx = sorted_training_idx[i]
			for k in xrange(num_stacked):
				if i+k < len_training:
					idx_k = sorted_training_idx[i+k]
					complete_D_train[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
		##Stack the test data
		complete_D_test = np.zeros([len(test_idx) - num_stacked + 1, num_stacked*n])
		len_test = len(test_idx)

		for i in xrange(len(sorted_test_idx) - num_stacked + 1):
			idx = sorted_test_idx[i]
			for k in xrange(num_stacked):
				if i+k < len_test:
					idx_k = sorted_test_idx[i+k]
					complete_D_test[i][k*n:(k+1)*n] =  Data[idx_k][0:n]


		#####INITIALIZATION!!!
		gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type="full")
		gmm.fit(complete_D_train)
		clustered_points = gmm.predict(complete_D_train)
		clustered_points_test = gmm.predict(complete_D_test)
		gmm_clustered_pts = clustered_points + 0

		gmm_covariances = gmm.covariances_


		true_confusion_matrix_gmm = compute_confusion_matrix(num_clusters,gmm_clustered_pts,sorted_training_idx)
		
		##Output a color file
		file_name = str_NULL + "color_file_gmm.txt"
		color_list_file = open(file_name,"w")
		str0 = ""
		counter = 0
		for i in xrange(len(gmm_clustered_pts)):
			if i % scaling_time == 0:
				cluster = gmm_clustered_pts[i]
				color_c = hexadecimal_color_list[cluster]
				str0 += '"' + str(color_c) + '"'
				if i == len(gmm_clustered_pts) - 1:
					pass
				else:
					str0 += ","
					counter += 1
		# color_list_file.write(str0)
		color_list_file.close()

		##Output a location file
		for i in xrange(m):
			if i % scaling_time == 0:
				location = (Data_pre[i,7],Data_pre[i,8])
				str0 += "new GLatLng" + str(location)
				if i == m-num_stacked:
					pass
				else:
					str0 += ","
					counter += 1
		location_file = open((str_NULL + "location_info_lam_sparse="+ str(lam_sparse)+"threshold="+str(threshold)+"maxClusters="+str(maxClusters)+".txt"),"w")
		# location_file.write(str0)
		location_file.close()

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

	empirical_cov = {}

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

			print "starting OPTIMIZATION for cluster#", cluster
			##Fit a model - OPTIMIZATION	
#			solver_model = sklearn.covariance.GraphLasso(alpha = lam_sparse)
#			solver_model.fit(D_train)
#			cov_out = solver_model.covariance_
#			X2 = np.linalg.inv(cov_out)
			size_blocks = n
			probSize = num_stacked * size_blocks
			lamb = np.zeros((probSize,probSize)) + lam_sparse
			for block_i in xrange(num_stacked):
				for block_j in xrange(num_stacked):
					scale = np.abs(block_i - block_j)
					lambda_block = np.zeros([size_blocks,size_blocks]) + scale*lam_sparse
					lamb[block_i*size_blocks:(block_i+1)*size_blocks, (block_j*size_blocks):(block_j+1)*size_blocks] += lambda_block
			S = np.cov(np.transpose(D_train) )
			#COPY THIS CODE
			gvx = SCT.TGraphVX()
			theta = SCT.semidefinite(probSize,name='theta')
			obj = -SCT.log_det(theta) + SCT.trace(S*theta)
			gvx.AddNode(0, obj)
			gvx.AddNode(1)
			dummy = SCT.Variable(1)
			gvx.AddEdge(0,1, Objective = lamb*dummy + num_stacked*dummy + size_blocks*dummy)
			gvx.Solve(Verbose=False, MaxIters=1000, Rho = 1, EpsAbs = 1e-6, EpsRel = 1e-6)


			#THIS IS THE SOLUTION
			empirical_cov[cluster] = S
			val = gvx.GetNodeValue(0,'theta')
			S_est = upper2Full(val, 0)
			u0,_ = np.linalg.eig(S_est)
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
				counter += 1
				counter = counter % num_clusters

			cluster_selected = sorted_cluster_norms[counter][1]
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


	##Caching stuff
	inv_matrix_cluster ={}
	log_det_cov_cluster = {}
	for cluster in xrange(num_clusters):
		cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
		log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
		inv_matrix_cluster[cluster] = np.linalg.inv(cov_matrix)
		log_det_cov_cluster[cluster] = log_det_cov



	##Code -----------------------SMOOTHENING
	##For each point compute the LLE 
	print "beginning with the smoothening ALGORITHM"
	LLE_all_points_clusters = np.zeros([len(clustered_points),num_clusters])
	for point in xrange(len(clustered_points)):
		# print "Point #", point
		if point + num_stacked-1 < complete_D_train.shape[0]:
			for cluster in xrange(num_clusters):
				# print "\nCLuster#", cluster
				cluster_mean = cluster_mean_info[num_clusters,cluster] 
				cluster_mean_stacked = cluster_mean_stacked_info[num_clusters,cluster] 

				x = complete_D_train[point,:] - cluster_mean_stacked#[0:(num_blocks-1)*n]
				cov_matrix = computed_covariance[num_clusters,cluster][0:(num_blocks-1)*n,0:(num_blocks-1)*n]
				inv_cov_matrix = inv_matrix_cluster[cluster]#np.linalg.inv(cov_matrix)
				log_det_cov = log_det_cov_cluster[cluster]#np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
				lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
				LLE_all_points_clusters[point,cluster] = lle
		
	##Update cluster points - using NEW smoothening
	clustered_points = updateClusters(LLE_all_points_clusters,switch_penalty = switch_penalty)

print "\n\ncompleted running the TICC algorithm. Saving results"

##Write the location and the appropriate color file
##Output a color file
file_name = str_NULL + "color_file_EM.txt"
color_list_file = open(file_name,"w")
str0 = ""
counter = 0
for i in xrange(len(gmm_clustered_pts)):
	if i % scaling_time == 0 and (i <40000 or i<5000):
		cluster = clustered_points[i]
		color_c = hexadecimal_color_list[int(cluster)]
		str0 += '"' + str(color_c) + '"'
		if i == len(gmm_clustered_pts) - 1:
			pass
		else:
			str0 += ","
			counter += 1
color_list_file.write(str0)
color_list_file.close()

##Output a location file
str0 = ""
counter = 0
for i in xrange(len(gmm_clustered_pts)):
	if i % scaling_time == 0 and (i<40000 or i <5000) :
		idx = sorted_training_idx[i]
		location = (Data_pre[idx,7],Data_pre[idx,8])
		str0 += "new GLatLng" + str(location)
		if idx == m-num_stacked:
			pass
		else:
			str0 += ","
			counter += 1
location_file = open((str_NULL + "EM location file lambda="+ str(lam_sparse)+"beta = " + str(switch_penalty) +"clusters="+str(number_of_clusters)+".txt"),"w")
location_file.write(str0)
location_file.close()


##With the inverses do some sort of thresholding
cluster = 0
for cluster in xrange(num_clusters):
	out = np.zeros(train_cluster_inverse[0].shape, dtype = np.int)
	A = train_cluster_inverse[cluster]
	for i in xrange(out.shape[0]):
		for j in xrange(out.shape[1]):
			if np.abs(A[i,j]) > threshold:
				out[i,j] = 1
	file_name = str_NULL+ "Cross time graphs"+str(cluster)+"maxClusters="+str(maxClusters-1)+"lam_sparse="+str(lam_sparse)+".jpg"
	out2 = out[(num_stacked-1)*n:num_stacked*n,]
	names = ["brake pedal amt","y-accn","sw angle","x-accn","velocity","RPM","gas pedal"]
	#names = ["DM stocks", "EM stocks", "Real estate", "Oil", "Gold", "HY bonds", "EM HY bonds", "GVT bonds", "CORP bonds", "IFL bonds."]
	if write_out_file:
		visualize(out2,-1,num_stacked,names,file_name)

plt.close("all")
cluster_means_raw = {}
str0 = ""
for cluster in xrange(num_clusters):
	counter = 0
	cluster_mean_raw = np.zeros(n)
	for i in xrange(len(clustered_points)):
		if clustered_points[i] == cluster:
			counter += 1
			cluster_mean_raw += np.array(Data_pre[i,:7])
	cluster_means_raw[cluster] = cluster_mean_raw/counter
	str0 += str(cluster_mean_raw/counter) + "\n"
means_file = open((str_NULL + "EM cluster means lam_sparse="+ str(lam_sparse)+"switch_penalty = " + str(switch_penalty) +"maxClusters="+str(maxClusters)+".txt"),"w")
# means_file.write(str0)
means_file.close()

print "\n\n\nBIC SCORE"
bic = compute_BIC_score(train_cluster_inverse, empirical_cov, LLE_all_points_clusters, clustered_points, sorted_training_idx)
print "BIC:", bic
























