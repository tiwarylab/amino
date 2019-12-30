import numpy as np
import copy

class OrderParameter:

    # name should be unique to the Order Parameter being defined
    # In other words for every possible pair of OP's x and y, (x.name != y.name) must be true
    def __init__(self, name, traj):
        self.name = name
        self.traj = traj

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)

class Memoizer:

    def __init__(self, bins):
        self.memo = {}
        self.bins = bins

    def d1_bin(self, x, bins = 80):
        p_x, _ = np.histogram(x, bins=bins)
        return p_x / np.sum(p_x)

    def d2_bin(self, x, y, bins = 50):
        p_xy, _, _ = np.histogram2d(x, y, bins=bins)
        return p_xy / np.sum(p_xy)

    def iqr(self, OP1, OP2):

        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)

        memo_val = self.memo.get(index1, False) or self.memo.get(index2, False)
        if memo_val:
            return memo_val

        x = OP1.traj
        y = OP2.traj
        p_x = self.d1_bin(x, self.bins)
        p_y = self.d1_bin(y, self.bins)
        p_xy = self.d2_bin(x, y, self.bins)

        p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
        info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y)))
        entropy = np.sum(-1 * p_xy * np.ma.log(p_xy))

        output = max(0.0, (1 - (info / entropy)))
        self.memo[index1] = output
        return output

class DissimilarityMatrix:

    def __init__(self, max_OPs, mut):
        self.max_OPs = max_OPs
        self.matrix = [[] for i in range(max_OPs)]
        self.mut = mut
        self.OPs = []

    def add_OP(self, OP):
        if len(self.OPs) == self.max_OPs:
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mut_info.append(self.mut.iqr(self.OPs[i], OP))
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
                mut_info[old_OP] = self.mut.iqr(OP, OP)
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else:
            for i in range(len(self.OPs)):
                mut_info = self.mut.iqr(OP, self.OPs[i])
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            self.matrix[len(self.OPs)].append(self.mut.iqr(OP, OP))
            self.OPs.append(OP)

def distortion(centers, ops, mut):
    dis = 0.0
    for i in ops:
        min_val = np.inf
        for j in centers:
            tmp = mut.iqr(i, j)
            if tmp < min_val:
                min_val = tmp
        dis = dis + (min_val * min_val)
    return 1 + (dis ** (0.5))

def grouping(new_OPs, all_OPs, mut):
    groups = [[] for i in range(len(new_OPs))]
    for OP in all_OPs:
        group = 0
        for i in range(len(new_OPs)):
            tmp = mut.iqr(OP, new_OPs[i])
            if tmp < mut.iqr(OP, new_OPs[group]):
                group = i
        groups[group].append(OP)
    return groups

def group_evaluation(OPs, mut):

    center = OPs[0]
    min_distortion = distortion([OPs[0]], OPs, mut)

    for i in OPs:
        tmp = distortion([i], OPs, mut)
        if tmp < min_distortion:
            center = i
            min_distortion = tmp

    return center

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

def find_ops(old_ops, max_outputs, bins):

    mut = Memoizer(bins)

    distortion_array = []
    num_array = []
    op_dict = {}

    while (max_outputs > 0):

        print("Checking " + str(max_outputs) + " order parameters...")

        matrix = DissimilarityMatrix(max_outputs, mut)
        for i in old_ops:
            matrix.add_OP(i)
        for i in old_ops[::-1]:
            matrix.add_OP(i)

        num_array.append(len(matrix.OPs))
        seed = []
        for i in matrix.OPs:
            seed.append(i)
        tmp_ops = cluster(old_ops, seed, mut)
        op_dict[len(seed)] = tmp_ops
        distortion_array.append(distortion(tmp_ops, old_ops, mut))
        max_outputs = max_outputs - 1

    num_ops = 0

    for dim in range(1,11):
        neg_expo = np.array(distortion_array) ** (-0.5 * dim)
        jumps = []
        for i in range(len(neg_expo) - 1):
            jumps.append(neg_expo[i] - neg_expo[i + 1])

        min_index = 0
        for i in range(len(jumps)):
            if jumps[i] > jumps[min_index]:
                min_index = i
        if num_array[min_index] > num_ops:
            num_ops = num_array[min_index]

    return op_dict[num_ops]
