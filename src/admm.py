
## ADMM Global Variables and Functions ##

# By default, the objective function is Minimize().
__default_m_func = Minimize
m_func = __default_m_func

# By default, rho is 1.0. Default rho update is identity function and does not
# depend on primal or dual residuals or thresholds.
__default_rho = 1.0
__default_rho_update_func = lambda rho, res_p, thr_p, res_d, thr_d: rho
rho = __default_rho
# Rho update function takes 5 parameters
# - Old value of rho
# - Primal residual and threshold
# - Dual residual and threshold
rho_update_func = __default_rho_update_func

def SetRho(Rho=None):
    global rho
    rho = Rho if Rho else __default_rho

# Rho update function should take one parameter: old_rho
# Returns new_rho
# This function will be called at the end of every iteration
def SetRhoUpdateFunc(Func=None):
    global rho_update_func
    rho_update_func = Func if Func else __default_rho_update_func

# Tuple of indices to identify the information package for each node. Actual
# length of specific package (list) may vary depending on node degree.
# X_NID: Node ID
# X_OBJ: CVXPY Objective
# X_VARS: CVXPY Variables (entry from node_variables structure)
# X_CON: CVXPY Constraints
# X_IND: Starting index into shared node_vals Array
# X_LEN: Total length (sum of dimensions) of all variables
# X_DEG: Number of neighbors
# X_NEIGHBORS: Placeholder for information about each neighbors
#   Information for each neighbor is two entries, appended in order.
#   Starting index of the corresponding z-value in edge_z_vals. Then for u.
(X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS) = range(8)

# Tuple of indices to identify the information package for each edge.
# Z_EID: Edge ID / tuple
# Z_OBJ: CVXPY Objective
# Z_CON: CVXPY Constraints
# Z_[IJ]VARS: CVXPY Variables for Node [ij] (entry from node_variables)
# Z_[IJ]LEN: Total length (sum of dimensions) of all variables for Node [ij]
# Z_X[IJ]IND: Starting index into shared node_vals Array for Node [ij]
# Z_Z[IJ|JI]IND: Starting index into shared edge_z_vals Array for edge [ij|ji]
# Z_U[IJ|JI]IND: Starting index into shared edge_u_vals Array for edge [ij|ji]
(Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
    Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)

# Contain all x, z, and u values for each node and/or edge in ADMM. Use the
# given starting index and length with getValue() to get individual node values
node_vals = None
edge_z_vals = None
edge_u_vals = None

# Extract a numpy array value from a shared Array.
# Give shared array, starting index, and total length.
def getValue(arr, index, length):
    return numpy.array(arr[index:(index + length)])

# Write value of numpy array nparr (with given length) to a shared Array at
# the given starting index.
def writeValue(sharedarr, index, nparr, length):
    if length == 1:
        nparr = [nparr]
    sharedarr[index:(index + length)] = nparr

# Write the values for all of the Variables involved in a given Objective to
# the given shared Array.
# variables should be an entry from the node_values structure.
def writeObjective(sharedarr, index, objective, variables):
    for v in objective.variables():
        vID = v.id
        value = v.value
        # Find the tuple in variables with the same ID. Take the offset.
        # If no tuple exists, then silently skip.
        for (varID, varName, var, offset) in variables:
            if varID == vID:
                writeValue(sharedarr, index + offset, value, var.size[0])
                break
# Proximal operators
def Prox_logdet(S, A, eta):
    global rho
    d, q = numpy.linalg.eigh(eta*A-S)
    q = numpy.matrix(q)
    X_var = ( 1/(2*float(eta)) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
    x_var = X_var[numpy.triu_indices(S.shape[1])] # extract upper triangular part as update variable      

    return numpy.matrix(x_var).T

def upper2Full(a):
    n = int((-1  + numpy.sqrt(1+ 8*a.shape[0]))/2)  
    A = numpy.zeros([n,n])
    A[numpy.triu_indices(n)] = a 
    temp = A.diagonal()
    A = (A + A.T) - numpy.diag(temp)             
    return A   

def ij2symmetric(i,j,size):
    return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i
    
# x-update for ADMM for one node
def ADMM_x(entry):
    global rho
    variables = entry[X_VARS]
    
    #-----------------------Proximal operator ---------------------------
    x_update = [] # proximal update for the variable x
    if(__builtin__.len(entry[1].args) > 1 ):
        # print 'we are in logdet + trace node'
        cvxpyMat = entry[1].args[1].args[0].args[0]
        numpymat = cvxpyMat.value

        mat_shape = ( int( numpymat.shape[1] *  ( numpymat.shape[1]+1 )/2.0 ) ,)
        a = numpy.zeros(mat_shape) 

        for i in xrange(entry[X_DEG]):  
            z_index = X_NEIGHBORS + (2 * i)
            u_index = z_index + 1
            zi = entry[z_index]
            ui = entry[u_index]
            
            for (varID, varName, var, offset) in variables:
                 
                z = getValue(edge_z_vals, zi + offset, var.size[0])
                u = getValue(edge_u_vals, ui + offset, var.size[0])
                a += (z-u) 
        A = upper2Full(a)
        A =  A/entry[X_DEG]
        eta = 1/float(rho)

        x_update = Prox_logdet(numpymat, A, eta)
        solution = numpy.array(x_update).T.reshape(-1)

        writeValue(node_vals, entry[X_IND] + variables[0][3], solution, variables[0][2].size[0]) 
    else:
        x_update = [] # no variable to update for dummy node
    return None

# z-update for ADMM for one edge
def ADMM_z(entry, index_penalty = 1):
    global rho
    
    rho = float(rho)
    #-----------------------Proximal operator ---------------------------
    a_ij = [] # 
    flag = 0
    variables_i = entry[Z_IVARS]
    for (varID, varName, var, offset) in variables_i:
        x_i = getValue(node_vals, entry[Z_XIIND] + offset, var.size[0])
        u_ij = getValue(edge_u_vals, entry[Z_UIJIND] + offset, var.size[0])
        if flag == 0:
            a_ij = (x_i + u_ij)
            flag = 1
        else:
            a_ij += (x_i + u_ij) 

    lamb = entry[1].args[0].args[0].value
    numBlocks = entry[1].args[1].args[0].value
    sizeBlocks = entry[1].args[2].args[0].value
    probSize = numBlocks*sizeBlocks
    z_ij = numpy.zeros(probSize*(probSize+1)/2)
    for i in range(numBlocks):
        if (i == 0):
            #In the A^{(0)} block (the blocks on the diagonal)
            for j in range(sizeBlocks):
                for k in range(j, sizeBlocks):
                    elems = numBlocks
                    lamSum = 0
                    points = numpy.zeros((elems))
                    locList = []
                    for l in range(elems):
                        (loc1, loc2) = (l*sizeBlocks + j, l*sizeBlocks+k)
                        locList.append((loc1,loc2))
                        index = ij2symmetric(loc1, loc2, probSize)
                        points[l] = a_ij[index]
                        lamSum = lamSum + lamb[loc1,loc2]
                    #Calculate soft threshold
                    #If answer is positive
                    ansPos = max((rho*numpy.sum(points) - lamSum)/(rho*elems),0)

                    #If answer is negative
                    ansNeg = min((rho*numpy.sum(points) + lamSum)/(rho*elems),0)

                    if (rho*numpy.sum(points) > lamSum):
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = ansPos
                    elif(rho*numpy.sum(points) < -1*lamSum):
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = ansNeg
                    else:
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = 0

        else:
            #Off-diagonal blocks
            for j in range(sizeBlocks):
                for k in range(sizeBlocks):

                    elems = (2*numBlocks - 2*i)/2
                    lamSum = 0
                    points = numpy.zeros((elems))
                    locList = []
                    for l in range(elems):
                        (loc1, loc2) = ((l+i)*sizeBlocks + j, l*sizeBlocks+k)
                        locList.append((loc2,loc1))
                        index = ij2symmetric(loc2, loc1, probSize)
                        points[l] = a_ij[index]
                        lamSum = lamSum + lamb[loc2,loc1]


                    #Calculate soft threshold
                    #If answer is positive
                    ansPos = max((rho*numpy.sum(points) - lamSum)/(rho*elems),0)

                    #If answer is negative
                    ansNeg = min((rho*numpy.sum(points) + lamSum)/(rho*elems),0)

                    if (rho*numpy.sum(points) > lamSum):
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = ansPos
                    elif(rho*numpy.sum(points) < -1*lamSum):
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = ansNeg
                    else:
                        for locs in locList:
                            index = ij2symmetric(locs[0], locs[1], probSize)
                            z_ij[index] = 0


    writeValue(edge_z_vals, entry[Z_ZIJIND] + variables_i[0][3], z_ij, variables_i[0][2].size[0])

    return None

# u-update for ADMM for one edge
def ADMM_u(entry):
    global rho
    size_i = entry[Z_ILEN]
    uij = getValue(edge_u_vals, entry[Z_UIJIND], size_i) +\
          getValue(node_vals, entry[Z_XIIND], size_i) -\
          getValue(edge_z_vals, entry[Z_ZIJIND], size_i)
    writeValue(edge_u_vals, entry[Z_UIJIND], uij, size_i)

    size_j = entry[Z_JLEN]
    uji = getValue(edge_u_vals, entry[Z_UJIIND], size_j) +\
          getValue(node_vals, entry[Z_XJIND], size_j) -\
          getValue(edge_z_vals, entry[Z_ZJIIND], size_j)
    writeValue(edge_u_vals, entry[Z_UJIIND], uji, size_j)
    return entry
