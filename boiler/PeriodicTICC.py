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
##PARAMETERS TO PLAY WITH 
cvxpy_iterations = 15300
num_blocks = 10
maxIters = 1 ##number of Iterations of the smoothening + clustering algo
adherence_penalty = 50000

switch_penalty = 40
seg_len = 100
lam_sparse = 0.75e-2

rep_iterations = 2
maxClusters = 4
threshold = 2e-5
write_out_file = False ##Only if True are any files outputted
scaling_time = 3
num_stacked = num_blocks - 1

#num_clusters = 4
hexadecimal_color_list = ["ff0000","0000ff","000099","ff00ff","00ff00","663300","996633","66ffff","3333cc","660066","66ccff","cc0000","0000ff","003300","33ff00","00ffcc","ffff00","ff9900","ff00ff","cccc66","666666","ffccff","660000","00ff00","ffffff","3399ff","006666","330000","ff0000","cc99ff","b0800f","3bd9eb","ef3e1b"]

##Sensor names
# out = []
# with open ("relevantSensors.csv", "r") as myfile:
# 	lis = [line.split() for line in myfile]
# 	lis = [line.split() for line in myfile]
# with open("relevantSensors.csv", "rb") as f:
#     reader = csv.reader(f, delimiter=",")
#     for i, line in enumerate(reader):
#         print 'line[{}] = {}'.format(i, line)
#         out.append(line)

# Number_of_burner_starts
# Operating_status:_Central_heating_active	
# Operating_status:_Hot_water_active	
# Operating_status:_Flame	
# Relay_status:_Gasvalve	
# Relay_status:_Fan	
# Relay_status:_Ignition	
# Relay_status:_CH_pump	
# Relay_status:_internal_3-way-valve
# Relay_status:_HW_circulation_pump	
# Supply_temperature_(primary_flow_temperature)_setpoint	
# Supply_temperature_(primary_flow_temperature)	
# CH_pump_modulation	
# Maximum_supply_(primary_flow)_temperature	
# Hot_water_temperature_setpoint	
# Hot_water_outlet_temperature	
# Actual_flow_rate_turbine	
# Fan_speed
# 2029717969629609985
######### Get Date into proper format
# data_pretest = np.loadtxt("2029717969629609985.csv")
# print data_pretest.shape
# data_pre = pd.read_csv('2029717969629609985.csv',sep = " ",low_memory = False)
# data_pre = pd.read_table("2029717969629609985.csv")
# print "completed getting the data"
# print data_pre.shape

##111,...1505,75 ---> Break pedal, Y -accn, sw - angle, x - accn, Vel, RPM,gas pedal
# out1 = data_pre.iloc[:,:]#[:,[111,1433,1498,1432,1471,1505,75,1613,1614]]
# out2 = out1.fillna(method = 'ffill')
# col_names = ['Operating_status:_Flame','Number_of_burner_starts','Operating_status:_Central_heating_active','Operating_status:_Hot_water_active'\
# ,'Supply_temperature_(primary_flow_temperature)_setpoint'\
# ,'Supply_temperature_(primary_flow_temperature)','CH_pump_modulation'\
# ,'Hot_water_outlet_temperature','Hot_water_temperature_setpoint'\
# ]#, 'Actual_flow_rate_turbine','Fan_speed']
# out2 = out2[col_names]

###Downsample the data
##Take one reading every minute into the dataset
col_names = ['Operating_status:_Flame','Number_of_burner_starts','Operating_status:_Central_heating_active','Operating_status:_Hot_water_active'\
,'Supply_temperature_(primary_flow_temperature)_setpoint'\
,'Supply_temperature_(primary_flow_temperature)','CH_pump_modulation'\
,'Hot_water_outlet_temperature','Hot_water_temperature_setpoint'\
]#, 'Actual_flow_rate_turbine','Fan_speed']




# ###PROCESSING FOR boiler DATA
# dfOrig = pd.read_csv('2029717969629609985.csv', parse_dates=[['Date','Time']], delimiter='\t')#,nrows=99999)
# dfOrig2 = dfOrig[col_names]
# dfOrig2['Date_Time'] = pd.to_datetime(dfOrig2['Date_Time'],format='%d.%m.%Y %H:%M:%S',  coerce = True)
# dfOrig2 = dfOrig2.fillna(method = 'ffill')
# dfOrig2 = dfOrig2.iloc[100:,:]
# startTime = dfOrig2.iloc[10,:]['Date_Time']
# newDf = pd.DataFrame()
# # while (count < 9):
# #    print 'The count is:', count
# #    count = count + 1
# counter = 0
# start_t = time.time()
# Data = np.zeros((100000,dfOrig2.shape[1]-1))
# while (1==1):
#    prevTime = startTime - pd.Timedelta('30 minutes') ##hopefully once every 7 days
#    dfPos = dfOrig2[(dfOrig2['Date_Time'] >= prevTime) & (dfOrig2['Date_Time'] < startTime)]
#    if counter%100 == 0:
#       print counter
#    if counter > 100000 or dfPos.shape[0] == 0:
#       print "it took", time.time() - start_t
#       break
#    Data[counter,:] = np.array(dfPos.iloc[-1,1:])
#    # newDf = newDf.append(dfPos.iloc[-1,:])#, ignore_index = True) # ignoring index is optional
#    startTime += pd.Timedelta('1 minute')
#    counter += 1


# Data = np.array(newDf)
max_lim = 200000
Data = np.loadtxt("cleaned_boiler_data.csv", delimiter = ",")
Data = Data[:max_lim,:]
file = open("time_Stamps2.txt","r")
out = file.read()
file.close()
out = out.split(",")
time_stamps = pd.to_datetime(out,format='%Y-%m-%d %H:%M:%S',  errors = "coerce")
time_stamps = time_stamps[:max_lim]
print "completed getting data"
print "data shape", Data.shape
print "time stamps ", time_stamps[1]
print "\n\n"

# UNNORMALIZED_Data = np.array(out2.iloc[65:,:])
# out2 = out2.iloc[30:,:10]
# out2 = (out2 - out2.mean())/(out2.max() - out2.min()) 
# Data_pre = UNNORMALIZED_Data
# Data = np.array(out2)[65:,:]#Data_pre[:,:6]
# Data = Data[:,:]
# Data = np.loadtxt("Synthetic Data Matrix rand_seed =[0,1] generated2.csv", delimiter= ",")
# Data_pre = Data
# UNNORMALIZED_Data = Data*1000
(m,n) = Data.shape
len_D_total = m
Nan_locations = np.argwhere(np.isnan(Data))
if len(Nan_locations) > 0:
	print "nan exist in data"
	print Nan_locations
	quit()



# def optimize(emp_cov = No)
#	""" Function for optimizing based upon the  

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
	max_iterations = 1000
	period_a = 60*24 ##daily period
	period_b = 60*24*7##weekly period
	for i in xrange(T-1):
		j = i+1
		future_costs = future_cost_vals[j,:]
		lle_vals = LLE_node_vals[j,:]
		total_vals = future_costs + lle_vals + switch_penalty
		total_vals[int(path[i])] -= switch_penalty

		path[i+1] = np.argmin(total_vals)
	####################################################
	start_period = time.time()
	##add the periodic constraints to the smoothened path
	## max_iterations, period_a , period_b
	for iteration in xrange(max_iterations):
		old_path = path + 0
		start_iter = time.time()
		for point in xrange(len(path)):
			points = []
			points.append(point + period_a)
			points.append(point - period_a)
			points.append(point + period_b)
			points.append(point - period_b)
			points.append(point + 1)
			points.append(point - 1)

			##take all the valid "neighbor" points
			final_idx = points

			if point < max(period_a, period_b) or point > len(path) - max(period_b,period_a):
				final_idx = []
				for idx in points:
					if idx < len(path) and idx >= 0:
						final_idx.append(idx)

			##get the clusters of these points
			clusters = [path[idx] for idx in final_idx]

			##base lle if the point was in that idx
			lle_base = LLE_node_vals[point,:]
			for cluster in clusters:
				lle_base[int(cluster)] -= switch_penalty

			#get the cluster of the max lle val
			cluster_out = np.argmin(lle_base)
			path[point] = cluster_out
		if np.array_equal(path, old_path):
			print "Converged !! BReaking Early of periodic viterbi"
			print "Number of iterations:", iteration
			break
		print "Iteration took:", time.time() - start_iter
	print "time took is:", time.time() - start_period
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
		# print "cluster #", cluster
		# print "TP,TN,FP,FN---------->", (TP,TN,FP,FN)
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


############
##The basic folder to be created
str_NULL = "VW_data_lam_sparse=" + str(lam_sparse)+"switch_penalty = "+str(switch_penalty) + "maxClusters=" +str(maxClusters)+"/"

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


# cluster_mean_num_cluster = {}
# cluster_mean_stacked_num_cluster = {}
train_cluster_inverse = {}
log_det_values = {}
computed_covariance = {}
cluster_mean_info = {}
cluster_mean_stacked_info = {}
for iters in xrange(maxIters):
	print "\n\n\nITERATION ###", iters
	start_time_iter = time.time()
	num_clusters = maxClusters - 1

	if iters == 0:
		## Now splitting up stuff 
		## split1 : Training and Test
		## split2 : Training and Test - different clusters
		training_percent = 1
		training_idx = np.random.choice(m-num_stacked+1, size=m-num_stacked+1,replace = False )
		sorted_training_idx = sorted(training_idx)
		test_idx = []
		# print "total length is:", m, "length of training is:", len(training_idx)
		##compute the test indices
		##Stack the complete data
		complete_Data = np.zeros([m - num_stacked + 1, num_stacked*n])
		len_data = m
		for i in xrange(m - num_stacked + 1):
			idx = i
			for k in xrange(num_stacked):
				if i+k < len_data:
					idx_k = i + k
					complete_Data[i][k*n:(k+1)*n] =  Data[idx_k][0:n]

		# ##Stack the training data
		# complete_D_train = np.zeros([len(training_idx) - num_stacked + 1, num_stacked*n])
		# len_training = len(training_idx)
		# for i in xrange(len(sorted_training_idx) - num_stacked + 1):
		# 	idx = sorted_training_idx[i]
		# 	for k in xrange(num_stacked):
		# 		if i+k < len_training:
		# 			idx_k = sorted_training_idx[i+k]
		# 			complete_D_train[i][k*n:(k+1)*n] =  Data[idx_k][0:n]
		complete_D_train = complete_Data
		# print "complete D_train shape:", complete_D_train.shape
		##Stack the test data


		#####INITIALIZATION!!!
		gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type="full")
		gmm.fit(complete_D_train)
		clustered_points = gmm.predict(complete_D_train)
		# print "clustered_points length", len(clustered_points)
		gmm_clustered_pts = clustered_points + 0

		gmm_covariances = gmm.covariances_


		true_confusion_matrix_gmm = np.zeros((num_clusters,num_clusters)) + 1#compute_confusion_matrix(num_clusters,gmm_clustered_pts,sorted_training_idx)
		
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
		color_list_file.write(str0)
		color_list_file.close()

		##Output a location file
		for i in xrange(m):
			if i % scaling_time == 0:
				location = (0,0)#(Data_pre[i,7],Data_pre[i,8])
				str0 += "new GLatLng" + str(location)
				if i == m-num_stacked:
					pass
				else:
					str0 += ","
					counter += 1
		# print "FINALLY CLUSTERED #points",i, "counter is:", counter
		location_file = open((str_NULL + "location_info_lam_sparse="+ str(lam_sparse)+"threshold="+str(threshold)+"maxClusters="+str(maxClusters)+".txt"),"w")
		# location_file.write(str0)
		location_file.close()

	##Get the train and test points
	train_clusters = collections.defaultdict(list)
	len_train_clusters = collections.defaultdict(int)

	counter = 0
	# print "length of training idx", len(training_idx)
	for point in range(len(clustered_points)):
		cluster = clustered_points[point]
		train_clusters[cluster].append(point)
		len_train_clusters[cluster] += 1
		counter +=1 

	# for point in range(len(clustered_points_test)):
	# 	cluster = clustered_points_test[point]
	# 	test_clusters[cluster].append(point)
	# 	len_test_clusters[cluster] += 1
	# 	counter +=1 



	##train_clusters holds the indices in complete_D_train 
	##for each of the clusters
	for cluster in xrange(num_clusters):
		if len_train_clusters[cluster] != 0:
			indices = train_clusters[cluster]



			D_train = np.zeros([len_train_clusters[cluster],num_stacked*n])
			for i in xrange(len_train_clusters[cluster]):
				point = indices[i]
				D_train[i,:] = complete_D_train[point,:]

			# print "DONE!!!"
			##Fit a model - OPTIMIZATION	
			# print "shape of the Data is:", D_train.shape
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
			# print lamb
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
			val = gvx.GetNodeValue(0,'theta')
			S_est = upper2Full(val, 0)
			u0,_ = np.linalg.eig(S_est)
			X2 = S_est
			u, _ = np.linalg.eig(S_est)
			cov_out = np.linalg.inv(X2)

			inv_matrix = cov_out
			# print "percent norm of the difference between the two is ------------------->", np.linalg.norm(cov_out - gmm_covariances[cluster])/np.linalg.norm(gmm_covariances[cluster])
			# print "percent norm of the difference with the inverse ------------------->", np.linalg.norm(S - gmm_covariances[cluster])/np.linalg.norm(gmm_covariances[cluster])

			# print "norm of the actual covariance matrix is -------------------->", np.linalg.norm(cov_out)
			# print "lod det value is -------------------> :", np.log(np.linalg.det(cov_out))

			##Store the log-det, covariance, inverse-covariance, cluster means, stacked means
			log_det_values[num_clusters, cluster] = np.log(np.linalg.det(cov_out))
			computed_covariance[num_clusters,cluster] = cov_out
			cluster_mean_info[num_clusters,cluster] = np.mean(D_train, axis = 0)[(num_stacked-1)*n:num_stacked*n].reshape([1,n])
			cluster_mean_stacked_info[num_clusters,cluster] = np.mean(D_train,axis=0)
			train_cluster_inverse[cluster] = X2

	cluster_norms = list(np.zeros(num_clusters))

	for cluster in xrange(num_clusters):
		print "length of the cluster ", cluster,"------>", len_train_clusters[cluster]
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
				print "counter is:", counter
				counter += 1
				counter = counter % num_clusters
				print "counter is:", counter

			cluster_selected = sorted_cluster_norms[counter][1]
			print "cluster that is zero is:", cluster, "selected cluster instead is:", cluster_selected
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
	# print "UPDATED THE OLD COVARIANCE"

	inv_cov_dict = {}
	log_det_dict = {}
	for cluster in xrange(num_clusters):
		cov_matrix  = computed_covariance[num_clusters, cluster]
		inv_cov_matrix = np.linalg.inv(cov_matrix)
		log_det_cov = np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
		inv_cov_dict[cluster] = inv_cov_matrix
		log_det_dict[cluster] = log_det_cov

	print "\n\n\nM STEP TOOK ---------->", time.time() - start_time_iter
	E_time = time.time()
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
				inv_cov_matrix = inv_cov_dict[cluster]#np.linalg.inv(cov_matrix)
				log_det_cov = log_det_dict[cluster]#np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
				lle = np.dot(   x.reshape([1,(num_blocks-1)*n]), np.dot(inv_cov_matrix,x.reshape([n*(num_blocks-1),1]))  ) + log_det_cov
				LLE_all_points_clusters[point,cluster] = lle
	print "E step took:", time.time() - E_time
	##Update cluster points - using NEW smoothening
	clustered_points = updateClusters(LLE_all_points_clusters,switch_penalty = switch_penalty)
	# print "number of 0    ",np.sum([(x == 1) for x in clustered_points])
	# print "number of 1    ",np.sum([(x == 0) for x in clustered_points])

# print "length of the clustered points is:", len(clustered_points)
# print "length of sorted_training_idx", len(sorted_training_idx)
plt.figure()
plt.scatter(sorted_training_idx[0:len(clustered_points)],clustered_points,color = "r",marker = ".",s =2)
plt.ylim((-0.5,num_clusters + 0.5))
# plt.savefig(str_NULL + "TRAINING_EM_NEW_lam_sparse="+str(lam_sparse) + ".jpg")
plt.close("all")

##Write the location and the appropriate color file
##Output a color file
file_name = str_NULL + "color_file_EM.txt"
color_list_file = open(file_name,"w")
str0 = ""
counter = 0
# print "length of gmm_clustered_pts", len(gmm_clustered_pts)
# print "length of clustered_points", len(clustered_points)
for i in xrange(len(gmm_clustered_pts)):
	if i % scaling_time == 0 and (i <40000 or i<5000) and clustered_points[i] == 4:
		cluster = clustered_points[i]
		color_c = hexadecimal_color_list[int(cluster)]
		str0 += '"' + str(color_c) + '"'
		if i == len(gmm_clustered_pts) - 1:
			pass
		else:
			str0 += ","
			counter += 1
# print "NUMBER OF POINTS IS---------->:", counter
# print "length of clustered points", len(gmm_clustered_pts)
color_list_file.write(str0)
color_list_file.close()

##Output a location file
str0 = ""
counter = 0
for i in xrange(len(gmm_clustered_pts)):
	if i % scaling_time == 0 and (i<40000 or i <5000) and clustered_points[i] == 4 :
		idx = sorted_training_idx[i]
		location = (0,0)#(Data_pre[idx,7],Data_pre[idx,8])
		str0 += "new GLatLng" + str(location)
		if idx == m-num_stacked:
			pass
		else:
			str0 += ","
			counter += 1
# print "NUMBER OF POINTS IS---------->:", counter
# print "length of sorted training data", len(sorted_training_idx)
location_file = open((str_NULL + "EM location file lam_sparse="+ str(lam_sparse)+"switch_penalty = " + str(switch_penalty) +"maxClusters="+str(maxClusters)+".txt"),"w")
# location_file.write(str0)
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
	names = col_names#["brake pedal amt","y-accn","sw angle","x-accn","velocity","RPM","gas pedal"]
	#names = ["DM stocks", "EM stocks", "Real estate", "Oil", "Gold", "HY bonds", "EM HY bonds", "GVT bonds", "CORP bonds", "IFL bonds."]
	visualize(out2,-1,num_stacked,names,file_name)

plt.close("all")


###################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################
print "\n\nCOMPUTING GRAPHS\n\n"
##Get the color list
points_colored = []
for cluster in clustered_points:
	points_colored.append(color_list[int(cluster)])

##Working with the computed cluster assignments 
# time_stamps = list(np.loadtxt('cleaned_boiler_timestamps.csv', delimiter = ","))
startTime = time_stamps[0]
graph1 = time.time()
len_plot = min(len(time_stamps), len(clustered_points))
plt.figure()
plt.scatter(time_stamps[:len_plot], clustered_points[:len_plot],color = points_colored[:len_plot])
plt.savefig(str_NULL + "complete_time_series_plot.png")
plt.close("all")
print "It took graph1:", time.time() - graph1
print "done with graph1"
# ##weekly stuff
# weekly_time_stamps = []
# weekly_time_stamps.append(startTime)
# i = 1
# weekly_time = 60*24*7
# graph2 = time.time()
# while i < len(time_stamps):
# 	nextTime = time_stamps[i]
# 	while nextTime - startTime > pd.Timedelta('7 days'):
# 		nextTime -= pd.Timedelta('7 days')
# 	weekly_time_stamps.append(nextTime)
# 	i += 1
# 	if i%1000 == 0:
# 		print 'i is:', i
# plt.figure()
# plt.scatter(weekly_time_stamps[0:len_plot], clustered_points[:len_plot], color = points_colored[:len_plot])
# plt.savefig(str_NULL + "weekly plot.png")
# plt.close("all")
# print "It took graph2:", time.time() - graph2
# print "done with graph2"

# ##daily stuff
# daily_time_stamps = []
# daily_time_stamps.append(startTime)
# graph3 = time.time()
# for i in xrange(1,len(time_stamps)):
# 	nextTime = time_stamps[i]
# 	while nextTime - startTime > pd.Timedelta('1 day'):
# 		nextTime -= pd.Timedelta('1 day')
# 	daily_time_stamps.append(nextTime)
# plt.figure()
# plt.scatter(daily_time_stamps[0:len_plot], clustered_points[:len_plot], color = points_colored[:len_plot])
# plt.savefig(str_NULL + "daily plot.png")
# plt.close("all")
# print "It took graph2:", time.time() - graph3
# print "done with graph3"

## plot of cluster vs time on a daily basis
num_daily = 24*60
daily_clusters = np.zeros((num_clusters,num_daily))
daily_time = []
startTime = time_stamps[0]
graph4 = time.time()
for i in xrange(num_daily):
	daily_time.append(startTime) #+ pd.Timedelta('1 minute')
	startTime += pd.Timedelta('1 minute')
	counter = i
	clusters_point = []
	while counter < len(clustered_points):
		clusters_point.append(clustered_points[counter])
		counter += num_daily
	for cluster in xrange(num_clusters):
		daily_clusters[cluster,i] = clusters_point.count(cluster)
daily_clusters = daily_clusters/np.sum(daily_clusters, axis = 0)
print "done with graph4"
print "it took graph4:", time.time() - graph4

graph5 = time.time()
for cluster in xrange(num_clusters):
	plt.figure()
	plt.plot(daily_time,daily_clusters[cluster,:])
	plt.savefig(str_NULL + "daily probabilities cluster= " + str(cluster) + ".png")
	plt.close("all")
print "done with graph5"
print "it took graph5:", time.time() - graph5


## plot of cluster vs time on a weekly basis
num_weekly = 24*60*7
weekly_clusters = np.zeros((num_clusters,num_weekly))
weekly_time = []
startTime = time_stamps[0]
for i in xrange(num_weekly):
	weekly_time.append(startTime) #+ pd.Timedelta('1 minute')
	startTime += pd.Timedelta('1 minute')
	counter = i
	clusters_point = []
	while counter < len(clustered_points):
		clusters_point.append(clustered_points[counter])
		counter += num_weekly
	for cluster in xrange(num_clusters):
		weekly_clusters[cluster,i] = clusters_point.count(cluster)
weekly_clusters = weekly_clusters/np.sum(weekly_clusters, axis = 0)
print "done with graph6"

for cluster in xrange(num_clusters):
	plt.figure()
	plt.plot(weekly_time,weekly_clusters[cluster,:])
	plt.savefig(str_NULL + "weekly probabilities cluster= " + str(cluster) + ".png")
	plt.close("all")
print "done with graph7"





##Save stuff for analyzing
np.savetxt(str_NULL + "clustered_points_lam_sparse = " + str(lam_sparse) + " switch_penalty=" + str(switch_penalty) + " clusters=" + str(num_clusters) +".csv", clustered_points,delimiter = ",")


print "\n\n\nTHRESHOLD IS:", threshold

print "lam_sparse", lam_sparse
print "switch_penalty", switch_penalty
print "num_cluster", maxClusters
print "NORMALIZED"





















