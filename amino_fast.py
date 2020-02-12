import numpy as np
from sklearn.neighbors import KernelDensity
import copy
import dask
import dask.multiprocessing
from dask.diagnostics import ProgressBar
dask.config.set(scheduler='processes')

# Order Parameter (OP) class - stores OP name and trajectory
class OrderParameter:

    # name should be unique to the Order Parameter being defined
    def __init__(self, name, traj):
        self.name = name
        self.traj = np.array(traj).reshape([-1,1])/np.std(traj)

    def __eq__(self, other):
        return self.name == other.name

    # This is needed for the sets construction used in clustering
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)

# Memoizes distance computation between OP's to prevent re-calculations
class Memoizer:

    def __init__(self, bins, bandwidth, kernel):
        self.memo = {}
        self.bins = bins
        self.bandwidth = bandwidth
        self.kernel = kernel

    # Binning two OP's in 2D space
    def d2_bin(self, x, y):
        KD = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel)
        KD.fit(np.column_stack((x,y)))
        grid1 = np.linspace(np.min(x),np.max(x),self.bins)
        grid2 = np.linspace(np.min(y),np.max(y),self.bins)
        mesh = np.meshgrid(grid1,grid2)
        data = np.column_stack((mesh[0].reshape(-1,1),mesh[1].reshape(-1,1)))
        samp = KD.score_samples(data)
        samp = samp.reshape(self.bins,self.bins)
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p

    # Checks if distance has been computed before, otherwise computes distance
    def distance(self, OP1, OP2):

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1, False) or self.memo.get(index2, False)
        if memo_val:
            return memo_val, False

        x = OP1.traj
        y = OP2.traj
        
        output = dask.delayed(self.dist_calc)(x,y)
        return output, index1

    def dist_calc(self, x, y):
        p_xy = self.d2_bin(x, y)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
        info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y)))
        entropy = np.sum(-1 * p_xy * np.ma.log(p_xy))

        output = max(0.0, (1 - (info / entropy)))
        return output
                
    def dist_matrix(self, group1, group2):
        tmps = []
        for i in group2:
            tmps.append([])
            for j in group1:
                mi, label = self.distance(i, j)
                tmps[-1].append(mi)
        return tmps
        

# Dissimilarity Matrix (DM) construction
class DissimilarityMatrix:

    # Initializes DM based on size
    def __init__(self, size, mut):
        self.size = size
        self.matrix = [[] for i in range(size)]
        self.mut = mut
        self.OPs = []

    # Checks a new OP against OP's in DM to see if it should be added to DM
    def add_OP(self, OP):
        if len(self.OPs) == self.size:
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mi, label = self.mut.distance(self.OPs[i], OP)
                mut_info.append(mi)
                product = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        product = product * self.matrix[i][j]
                existing.append(product)
            #mut_info = list(dask.compute(*mut_info,scheduler='single-threaded'))
            update = False
            difference = None
            for i in range(len(self.OPs)):
                candidate_info = 1
                for j in range(len(self.OPs)):
                    if not i == j:
                        candidate_info = candidate_info * mut_info[j]
                if candidate_info > existing[i]:
                    update = True
                    if difference == None:
                        difference = candidate_info - existing[i]
                        old_OP = i
                    else:
                        if (candidate_info - existing[i]) > difference:
                            difference = candidate_info - existing[i]
                            old_OP = i
            if update == True:
                mi, label = self.mut.distance(OP, OP)
                #mi = dask.compute(mi,scheduler='single-threaded')
                mut_info[old_OP] = mi
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else:
            distances = []
            for i in range(len(self.OPs)):
                mi,label = self.mut.distance(OP, self.OPs[i])
                distances.append(mi)
            #distances = list(dask.compute(*distances,scheduler='single-threaded'))
            for i in range(len(self.OPs)):
                mut_info = distances[i]
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            mi, label = self.mut.distance(OP, OP)
            #mi = dask.compute(mi)
            self.matrix[len(self.OPs)].append(mi)
            self.OPs.append(OP)
           
            
# Computes distortion using selected `centers` given full set of `ops`
def distortion(centers, ops, mut):
    tmps = mut.dist_matrix(centers, ops)
    min_vals = np.min(tmps,axis=1)
    dis = np.sum(min_vals**2)
    return 1 + (dis ** (0.5))

# Groups `ops` around `centers`
def grouping(centers, ops, mut):
    groups = [[] for i in range(len(centers))]
    tmps = mut.dist_matrix(centers, ops)  
    assignment = np.argmin(tmps,axis=1)
     
    for i in range(len(assignment)):
        groups[assignment[i]].append(ops[i])
    return groups

# Returns the "center-most" OP in the set `ops`
def group_evaluation(ops, mut):

    index_mat = np.ones((len(ops),len(ops)))
    np.fill_diagonal(index_mat,0)
    pairs = np.argwhere(np.triu(index_mat)==1)
    dist_mat = np.zeros((len(ops),len(ops)))
    distances = []

    for pair in pairs:
        mi, label = mut.distance(ops[pair[0]], ops[pair[1]])
        distances.append(mi)

    #distances = dask.compute(*distances, scheduler='single-threaded')

    for i in range(len(pairs)):
        pair = pairs[i]
        dist_mat[pair[0], pair[1]] = distances[i]


    dist_mat = dist_mat + dist_mat.T
    distortions = 1 + np.sum(dist_mat**2, axis=0)**(.5)
    center = ops[np.argmin(distortions)]
    
    return center

def full_matrix(ops, mut):
    index_mat = np.ones((len(ops),len(ops)))
    pairs = np.argwhere(np.triu(index_mat)==1)
    dist_mat = np.zeros((len(ops),len(ops)))
    distances = []
    labels = []

    for pair in pairs:
        mi, label = mut.distance(ops[pair[0]], ops[pair[1]])
        distances.append(mi)
        labels.append(label)
    with ProgressBar():
        distances = dask.compute(*distances)

    for i in range(len(labels)):
        mut.memo[labels[i]] = distances[i]
        
            

# Running clustering on `ops` starting with `seeds`
def cluster(ops, seeds, mut):

    old_centers = []
    centers = copy.deepcopy(seeds)

    while (set(centers) != set(old_centers)):
        old_centers = copy.deepcopy(centers)
        centers = []
        groups = grouping(old_centers, ops, mut)
        for i in range(len(groups)):
            result = group_evaluation(groups[i], mut)
            centers.append(result)

    return centers

def set_bandwidth(ops, kernel):
        if kernel == 'parabolic':
            kernel = 'epanechnikov'
        if kernel == 'epanechnikov':
            bw_constant = 2.2
        else:
            bw_constant = 1
        
        n = np.shape(ops[0].traj)[0]
        bandwidth = bw_constant*n**(-1/6)
        
        print('Selected bandwidth: ' + str(bandwidth)+ '\n')
        
        return bandwidth
        
def starting_centroids(old_ops, max_outputs, mut):
    matrix = DissimilarityMatrix(max_outputs, mut)
    for i in old_ops:
        matrix.add_OP(i)
    for i in old_ops[::-1]:
        matrix.add_OP(i)
        
    return matrix

def k_clusters(old_ops, max_outputs, mut):
            
        # DM construction
        matrix = starting_centroids(old_ops, max_outputs, mut)
        
        
        # Clustering
        seed = []
        for i in matrix.OPs:
            seed.append(i)
        centroids = cluster(old_ops, seed, mut)
        disto = distortion(centroids, old_ops, mut)
        
        return centroids, disto

def num_clust(distortion_array, num_array, jump_filename):
    num_ops = 0
    all_jumps = []

    for dim in range(1,11):
        neg_expo = np.array(distortion_array) ** (-0.5 * dim)
        jumps = []
        for i in range(len(neg_expo) - 1):
            jumps.append(neg_expo[i] - neg_expo[i + 1])
        all_jumps.append(jumps)

        min_index = 0
        for i in range(len(jumps)):
            if jumps[i] > jumps[min_index]:
                min_index = i
        if num_array[min_index] > num_ops:
            num_ops = num_array[min_index]
        
        return num_ops

# This is the general workflow for AMINO
def find_ops(old_ops, max_outputs=20, bins=None, bandwidth=None, kernel='gaussian', jump_filename=None):
    if bandwidth == None:
        bandwidth = set_bandwidth(old_ops, kernel)

    mut = Memoizer(bins, bandwidth, kernel)
    distortion_array = []
    num_array = []
    op_dict = {}

    if bins == None:
        bins = np.ceil(np.sqrt(len(old_ops[0].traj)))
        
    print('Calculating all pairwise distances...')
    full_matrix(old_ops, mut)

    # This loops through each number of clusters
    while (max_outputs > 0):

        print("Checking " + str(max_outputs) + " order parameters...")
        
        tmp_ops, disto = k_clusters(old_ops, max_outputs, mut)

        num_array.append(max_outputs)
        op_dict[max_outputs] = tmp_ops
        distortion_array.append(disto)
        
        max_outputs = max_outputs - 1

    
    if not jump_filename == None:
        np.save(jump_filename, distortion_array[::-1])
        
    # Determining number of clusters
    num_ops = num_clust(distortion_array, num_array, jump_filename)

    return op_dict[num_ops]