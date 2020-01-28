import numpy as np
import copy

# Order Parameter (OP) class - stores OP name and trajectory
class OrderParameter:

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

    def __init__(self, bins):
        self.memo = {}
        self.bins = bins

    # Binning two OP's in 2D space
    def d2_bin(self, x, y, bins = 50):
        p_xy, _, _ = np.histogram2d(x, y, bins=bins)
        return p_xy / np.sum(p_xy)

    # Checks if distance has been computed before, otherwise computes distance
    def distance(self, OP1, OP2):

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1, False) or self.memo.get(index2, False)
        if memo_val:
            return memo_val

        x = OP1.traj
        y = OP2.traj
        p_xy = self.d2_bin(x, y, self.bins)
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
def find_ops(old_ops, max_outputs=20, bins=None, jump_filename=None):

    mut = Memoizer(bins)
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
