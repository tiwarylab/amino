"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith.

This is the parallelized kernel density estimation version.

Read and cite the following when using this method:
https://pubs.rsc.org/--/content/articlehtml/2020/me/c9me00115h
"""

import numpy as np
from sklearn.neighbors import KernelDensity
import copy
import dask
import dask.multiprocessing
from dask.diagnostics import ProgressBar
dask.config.set(scheduler='processes')

class OrderParameter:
    """Order Parameter (OP) class - stores OP name and trajectory
    
    Attributes
    ----------
    name : str
        Name of OP.
        
    traj : list of floats or np.array
        Trajectory of OP values. This will be normalized to have std = 1.
        
    """

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


class Memoizer:
    """Memoizes distance computation between OP's to prevent re-calculations.
    
    Attributes
    ----------
    bins : int
        Number of values along each axis for the joint probability. 
        The probability will be a bins x bins grid.
        
    bandwidth : float
        Bandwidth parameter for kernel denensity estimation.
        
    kernel : str
        Kernel name for kernel density estimation.
        
    """

    def __init__(self, bins, bandwidth, kernel):
        self.memo = {}
        self.bins = bins
        self.bandwidth = bandwidth
        self.kernel = kernel

    # Binning two OP's in 2D space
    def d2_bin(self, x, y):
        """ Calculate a joint probability distribution for two trajectories.
        
        Parameters
        ----------
        x : np.array
            Trajectory of first OP.
            
        y : np.array
            Trajcetory of second OP.
            
        Returns
        -------
        p : np.array
            self.bins by self.bins array of joint probabilities from KDE.
            
        """
        
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

    def distance(self, OP1, OP2):
        """Returns the mutual information distance between two OPs.
        Calls distance calculation in parallel or returns saved values.
        
        Parameters
        ----------
        OP1 : OrderParameter
            The first order parameter for distance calculation.
            
        OP2 : OrderParameter
            The second order parameter for distance calculation.
            
        Returns
        -------
        output : float
            The mutual information distance.
            
        label (index1 or False) : str or bool
            The label for saving the distance in the memoizer.
            The name of the OP pair if not memoized or False if memoized.
            
        """

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1)
        if memo_val == None: 
            memo_val = self.memo.get(index2)
        if memo_val != None:
            return memo_val, False

        x = OP1.traj
        y = OP2.traj
        
        output = dask.delayed(self.dist_calc)(x,y)

        return output, index1

    def dist_calc(self, x, y):
        """Calculates the distance between two trajectories.
        This is called when distances are not memoized.
        
        Parameters
        ----------
        x : np.array
            First trajectory.
            
        y : np.array
            Second trajectory.
            
        Returns
        -------
        output : float
            Calculated mutual information distance.
        
        """
        p_xy = self.d2_bin(x, y)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
        info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y)))
        entropy = np.sum(-1 * p_xy * np.ma.log(p_xy))

        output = max(0.0, (1 - (info / entropy)))
        return output
                
    def dist_matrix(self, group1, group2):
        """Calculates all distances between two groups of OPs.
        
        Parameters
        ----------
        group1 : list of OrderParameters
            First group of OPs.
            
        group2 : list of OrderParameters
            Second group of OPs.
            
        Returns
        -------
        tmps : list of lists of floats
            Matrix containing distances between the groups.
            
        """
        tmps = []
        for i in group2:
            tmps.append([])
            for j in group1:
                mi, label = self.distance(i, j)
                tmps[-1].append(mi)
        return tmps
        

# Dissimilarity Matrix (DM) construction
class DissimilarityMatrix:
    """Matrix containing distances for initial centroid determination.
    
    Attributes
    ----------
    size : int
        Maximum number of OPs contained.
    
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    """

    def __init__(self, size, mut):
        self.size = size
        self.matrix = [[] for i in range(size)]
        self.mut = mut
        self.OPs = []
        
    def add_OP(self, OP):
        """Adds OPs to the matrix if they should be added.
        OPs should be added when there are fewer OPs than self.size
        or if they can increase the geometric mean of distances
        by being swapped with a different OP.
        
        Parameters
        ----------
        OP : OrderParameter
            OP to potentially be added.
            
        Returns
        -------
        None
            self.matrix is updated directly without a return.
        """
        if len(self.OPs) == self.size: # matrix is full, check for swaps
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
            if update == True: # swapping out an OP
                mi, label = self.mut.distance(OP, OP)
                mut_info[old_OP] = mi
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else: # adding an OP when there are fewer than self.size
            distances = []
            for i in range(len(self.OPs)):
                mi,label = self.mut.distance(OP, self.OPs[i])
                distances.append(mi)
            for i in range(len(self.OPs)):
                mut_info = distances[i]
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            mi, label = self.mut.distance(OP, OP)
            #mi = dask.compute(mi)
            self.matrix[len(self.OPs)].append(mi)
            self.OPs.append(OP)
           
            
def distortion(centers, ops, mut):
    """Computes the distortion between a set of centeroids and OPs.
    When multiple centoids are used, the minimum distortion grouping
    will be used to calculate the total distortion.
    
    Parameters
    ----------
    centers : list of OrderParameters
        Cluster centroids.
        
    ops : list of OrderParameters
        All OPs belonging to the centroid's clusters.
        
    Returns
    -------
    float
        Minimum total distortion given the centroids and OPs.
    
    """
    tmps = mut.dist_matrix(centers, ops)
    min_vals = np.min(tmps,axis=1)
    dis = np.sum(min_vals**2)
    return 1 + (dis ** (0.5))

def grouping(centers, ops, mut):
    """Assigns OPs to minimum distortion clusters.
    
    Parameters
    ----------
    centers : list of OrderParameters
        Cluster centroids.
        
    ops : list of OrderParameters
        All OPs to be assigned to clusters.
        
    Returns
    -------
    groups : list of lists of OrderParameters
        One list for each centroid containing the associated OPs.
    
    """
    groups = [[] for i in range(len(centers))]
    tmps = mut.dist_matrix(centers, ops)  
    assignment = np.argmin(tmps,axis=1)
     
    for i in range(len(assignment)):
        groups[assignment[i]].append(ops[i])
    return groups

def group_evaluation(ops, mut):
    """Calculates the centroid minimizing distortion for a set of OPs.
    
    Parameters
    ----------
    ops : list of OrderParameters
        Set of OPs for centroid calculation.
    
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    center : OrderParameter
        The OP that is the minimum distortion centroid.
        
    """

    center = ops[0]
    min_distortion = distortion([ops[0]], ops, mut)
    for i in ops:
        tmp = distortion([i], ops, mut)
        if tmp < min_distortion:
            center = i
            min_distortion = tmp
    return center


def full_matrix(ops, mut):
    """Calculates all the OP distances in parallel.
    Used before any of the clustering to maximize the
    number of distances calculated at once.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    None
        Stores all values in mut.memo and does not return a value.
    """
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
        
            

def cluster(ops, seeds, mut):
    """Clusters OPs startng with centroids from a DissimilarityMatrix.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    seeds : list of OrderParameters
        Starting centroids from a DissimilarityMatrix.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    centers : list of OPs
        Final centroids after clustering.
        
    """

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
    """Calculates the bandwidth consistent with Scott and Silverman's
    rules of thumb for bandwidth selection.
    
    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.
        
    kernel : str
        Kernel name for kernel density estimation.
        
    Returns
    -------
    bandwidth : float
        Bandwidth from the rules of thumb (they're the same for 2D KDE).
        
    """
    
    if kernel == 'epanechnikov':
        bw_constant = 2.2
    else:
        bw_constant = 1

    n = np.shape(ops[0].traj)[0]
    bandwidth = bw_constant*n**(-1/6)

    print('Selected bandwidth: ' + str(bandwidth)+ '\n')

    return bandwidth
        
def starting_centroids(old_ops, max_outputs, mut):
    """"Makes a DissimilarityMatrix and scans all OPs forward and backward
    adding OPs to have maximum separation between starting centroids.
    
    Parameters
    ----------
    old_ops : list of OrderParameters
        All OPs.
        
    max_outputs : int
        Number of starting centroids.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    matrix : DissimilarityMatrix
        The DissimilarityMatrix with the starting centroids.
        
    """"
    matrix = DissimilarityMatrix(max_outputs, mut)
    for i in old_ops:
        matrix.add_OP(i)
    for i in old_ops[::-1]:
        matrix.add_OP(i)
        
    return matrix

def k_clusters(old_ops, max_outputs, mut):
    """Calculates starting centroids, clusters, and calculates
    total dissimilarity for a set number of clusters.
    
    old_ops : list of OrderParameters
        All OPs.
        
    max_outputs : int
        The number of clusters to be calculated.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    centroids : list of OrderParameters
        Centroids resulting from clustering.
        
    disto : float
        The total distortion from all centroids.
        
    """
    
    # DM construction
    matrix = starting_centroids(old_ops, max_outputs, mut)


    # Clustering
    seed = []
    for i in matrix.OPs:
        seed.append(i)
    centroids = cluster(old_ops, seed, mut)
    disto = distortion(centroids, old_ops, mut)

    return centroids, disto

def num_clust(distortion_array, num_array):
    """Calculates the optimal number of clusters given
    the distortion for each k-clustering.
    
    Parameters
    ----------
    distortion_array : list of floats
        The total distortion for each k-clustering.
        
    num_array : list of ints
        The number of clusters associated with the distortions.
        
    Returns
    -------
    num_ops : int
        The optimal number of clusters/centroids.
        
    """
    
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
def find_ops(old_ops, max_outputs=20, bins=20, bandwidth=None, kernel='epanechnikov', jump_filename=None):
    """Main function performing clustering and finding the optimal number of OPs.
    
    Parameters
    ----------
    old_ops : list of OrderParameters
        All OPs for clustering.
        
    max_outputs : int
        The maximum number of clusters/centroids.
        
    bins : int or None
        Number of values along each axis for the joint probability. 
        The probability will be a bins x bins grid.
        If None this is set with a rule of thumb.
        
    bandwidth : float or None
        Bandwidth parameter for kernel denensity estimation.
        If None this is set with a rule of thumb.
        
    kernel : str
        Kernel name for kernel density estimation.
        It is recommended to use either epanechnikov (parabolic) or gaussian.
        These are currently the only two implemented in bandwidth rule of thumb.
        
    jump_filename : str or None
        The filename to save distortion jumps.
        
    Returns
    -------
    list of OPs
        The centrioids for the optimal clustering.
    """
    
    if kernel == 'parabolic':
        kernel = 'epanechnikov'
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
    num_ops = num_clust(distortion_array, num_array)

    return op_dict[num_ops]