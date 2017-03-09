##New visualization function
import matplotlib.pyplot as plt

def visualize(Matrix_v, stock_number, time_steps, y_ticks,file_name):
	"""
	Takes in a np.matrix of size m x n
	and Visualizes the cross time correlation structure 
	stores the plot using file_name passed in

	@params
	m : number of sensors/stocks in each time steps
	n : number of time_steps*sensors_per_time_step behind that we are going
	stock_number : The particular stock whose connections you want to checkout
	Note the matrix should be sparse

	Visualizes the matrix
	"""
	(m,n) = Matrix_v.shape
	plt.figure()
	time_separation = 2
	row_separation = 2
	marker_size = 300
	time_per_step = n/time_steps

	##Add x ticks
	my_xticks = []
	x = []
	for i in xrange(1,time_steps+1):
		my_xticks.append(("t="+str(i)))
		x.append(i*time_separation)
	plt.xticks(x,my_xticks)
	ax = plt.axes()
	##Add y ticks
	y = []
	for i in xrange(1,m+1):
		y.append(i*row_separation)
	plt.yticks(y,y_ticks) 

	##Create the points that we want!
	for t in xrange(1,time_steps+1):
		for row in xrange(1,m+1):
			plt.scatter(t*time_separation,row*row_separation, s = marker_size, marker ='o', facecolors = 'none')

	##Make the cross time edges
	if stock_number != -1:
		for i in xrange(n):
			if Matrix_v[stock_number,i] != 0:
				stock_num_edge = (i%time_per_step)*row_separation
				time_stock_edge = (int(i/time_per_step) + 1)*time_separation
				##Draw an edge from the time_step to the 
				ax.arrow([time_stock_edge, time_steps*time_separation],[stock_num_edge, stock_number*row_separation], head_width = 0.05, head_length = 0.1, fc = 'k', ec='k')#color ='black', linestyle = '-', linewidth = 0.5)
	else:
		for stock_number in xrange(m):
			for i in xrange(n - time_per_step):
				if Matrix_v[stock_number,i] != 0:
					stock_num_edge = (i%time_per_step + 1)*row_separation
					time_stock_edge = (int(i/time_per_step) + 1)*time_separation
					##Draw an edge from the time_step to the 
					plt.plot([time_stock_edge, (time_steps)*time_separation],[stock_num_edge, (stock_number+1)*row_separation], color ='black', linestyle = '-', linewidth = 0.5)

	plt.savefig(file_name)
	plt.show()