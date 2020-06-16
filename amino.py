"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith.

This is the serial kernel density estimation version.

Read and cite the following when using this method:
https://pubs.rsc.org/--/content/articlehtml/2020/me/c9me00115h
"""

import numpy as np
from sklearn.neighbors import KernelDensity
import copy

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
        self.traj = traj

    def __eq__(self, other):
        return self.name == other.name

    # This is needed for the sets construction used in clustering
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)

# Memoizes distance computation between OP's to prevent re-calculations
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
    
    def __init__(self, bins, bandwidth, kernel, weights=None):
        self.memo = {}
        self.bins = bins
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights

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
        x,y = np.array(x),np.array(y) # FIX THE NORMALIZATION TO BE AT OP INITIALIZATION
        x = x.reshape([-1,1])/np.std(x)
        y = y.reshape([-1,1])/np.std(y)

        KD = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel)
        KD.fit(np.column_stack((x,y)), sample_weight=self.weights)
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
        """Returns the mutual information distance between two OPs.
        
        Parameters
        ----------
        OP1 : OrderParameter
            The first order parameter for distance calculation.
            
        OP2 : OrderParameter
            The second order parameter for distance calculation.
            
        Returns
        -------
        float
            The mutual information distance.
            
        """

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1, False) or self.memo.get(index2, False)
        if memo_val:
            return memo_val

        x = OP1.traj
        y = OP2.traj
        p_xy = self.d2_bin(x, y)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
        info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y)))
        entropy = np.sum(-1 * p_xy * np.ma.log(p_xy))

        output = max(0.0, (1 - (info / entropy)))
        self.memo[index1] = output
        return output

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

    # Initializes DM based on size
    def __init__(self, size, mut):
        self.size = size
        self.matrix = [[] for i in range(size)]
        self.mut = mut
        self.OPs = []

    # Checks a new OP against OP's in DM to see if it should be added to DM
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
        
        if len(self.OPs) == self.size:
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mut_info.append(self.mut.distance(self.OPs[i], OP))
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
            if update == True:
                mut_info[old_OP] = self.mut.distance(OP, OP)
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else:
            for i in range(len(self.OPs)):
                mut_info = self.mut.distance(OP, self.OPs[i])
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            self.matrix[len(self.OPs)].append(self.mut.distance(OP, OP))
            self.OPs.append(OP)

# Computes distortion using selected `centers` given full set of `ops`
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
    dis = 0.0
    for i in ops:
        min_val = np.inf
        for j in centers:
            tmp = mut.distance(i, j)
            if tmp < min_val:
                min_val = tmp
        dis = dis + (min_val * min_val)
    return 1 + (dis ** (0.5))

# Groups `ops` around `centers`
def grouping(centers, ops, mut):
    """Assigns OPs to minimum distortion clusters.
    
    Parameters
    ----------
    centers : list of OrderParameters
        Cluster centroids.
        
    ops : list of OrderParameters
        All OPs to be assigned to clusters.
        
    mut : Memoizer
        The Memoizer used for distance calculation and storage.
        
    Returns
    -------
    groups : list of lists of OrderParameters
        One list for each centroid containing the associated OPs.
    
    """
    
    groups = [[] for i in range(len(centers))]
    for OP in ops:
        group = 0
        for i in range(len(centers)):
            tmp = mut.distance(OP, centers[i])
            if tmp < mut.distance(OP, centers[group]):
                group = i
        groups[group].append(OP)
    return groups

# Returns the "center-most" OP in the set `ops`
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

# Running clustering on `ops` starting with `seeds`
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

# This is the general workflow for AMINO
def find_ops(old_ops, max_outputs=20, bins=20, bandwidth=None, kernel='epanechnikov', jump_filename=None, weights = None):
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
    
    if bandwidth == None:
        if kernel == 'parabolic':
            kernel = 'epanechnikov'
        if kernel == 'epanechnikov':
            bw_constant = 2.2
        else:
            bw_constant = 1
        
        if type(weights) == type(None):
            n = np.shape(old_ops[0].traj)[0]
        else:
            weights = np.array(weights)
            n = np.sum(weights)**2 / np.sum(weights**2)
        bandwidth = bw_constant*n**(-1/6)
        print('Selected bandwidth: ' + str(bandwidth)+ '\n')

    mut = Memoizer(bins, bandwidth, kernel, weights)
    distortion_array = []
    num_array = []
    op_dict = {}

    if bins == None:
        bins = np.ceil(np.sqrt(len(old_ops[0].traj)))

    # This loops through each number of clusters
    while (max_outputs > 0):

        print("Checking " + str(max_outputs) + " order parameters...")

        # DM construction
        matrix = DissimilarityMatrix(max_outputs, mut)
        for i in old_ops:
            matrix.add_OP(i)
        for i in old_ops[::-1]:
            matrix.add_OP(i)

        # Clustering
        num_array.append(len(matrix.OPs))
        seed = []
        for i in matrix.OPs:
            seed.append(i)
        tmp_ops = cluster(old_ops, seed, mut)
        op_dict[len(seed)] = tmp_ops
        distortion_array.append(distortion(tmp_ops, old_ops, mut))
        max_outputs = max_outputs - 1

    # Determining number of clusters
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

    if not jump_filename == None:
        np.save(jump_filename, distortion_array[::-1])

    return op_dict[num_ops]