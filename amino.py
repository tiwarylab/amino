import numpy as np
import random
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

def arithmetic_mean(centers, ops, mut):
    mean = 0.0
    for i in ops:
        min_val = np.inf
        for j in centers:
            tmp = mut.iqr(i, j)
            if tmp < min_val:
                min_val = tmp
        mean = mean + min_val
    return (mean / (len(ops)))

def geometric_mean(centers, ops, mut):
    mean = 1.0
    for i in ops:
        min_val = np.inf
        for j in centers:
            tmp = mut.iqr(i, j)
            if tmp < min_val:
                min_val = tmp
        mean = mean * (min_val ** (1.0 / len(ops)))
    return mean

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

    def reduce(self):
        min_val = 10
        index = -1
        for i in range(len(self.matrix)):
            product = 1
            for j in range(len(self.matrix[i])):
                if not i == j:
                    product = product * self.matrix[i][j]
            if product < min_val:
                index = i
                min_val = product
        self.matrix.pop(index)
        for i in range(len(self.matrix)):
            self.matrix[i].pop(index)
        self.OPs.pop(index)

    def min_product(self):
        min_val = 10
        for i in range(len(self.matrix)):
            product = 1
            for j in range(len(self.matrix[i])):
                if not i == j:
                    product = product * self.matrix[i][j]
            if product < min_val:
                min_val = product
        return min_val

    def get_OPs(self):
        return self.OPs

    def __str__(self):
        output = ""
        output = output + "OPs:\n"
        for i in self.OPs:
            output = output + str(i) + "\n"
        output = output + "\nMatrix:\n"
        for i in self.matrix:
            for j in i:
                output = output + str(j) + " "
            output = output + "\n"
        return output

def d1_bin(x, bins = 80):

    p_x, _ = np.histogram(x, bins=bins)
    return p_x / np.sum(p_x)

def d2_bin(x, y, bins = 50):

    if len(x) != len(y):
        raise Exception("Order parameter lists are of different size.")

    p_xy, _, _ = np.histogram2d(x, y, bins=bins)
    return p_xy / np.sum(p_xy)

class Memoizer:

    def __init__(self, bins):
        self.memo = {}
        self.bins = bins

    def iqr(self, OP1, OP2):
        index1 = str(OP1.name) + " " + str(OP2.name)
        index2 = str(OP2.name) + " " + str(OP1.name)
        if index1 in self.memo:
            return self.memo[index1]
        elif index2 in self.memo:
            return self.memo[index2]
        else:
            x = OP1.traj
            y = OP2.traj
            p_x = d1_bin(x, self.bins)
            p_y = d1_bin(y, self.bins)
            p_xy = d2_bin(x, y, self.bins)

            info = 0
            entropy = 0

            entropy = np.sum(-1 * p_xy * np.ma.log(p_xy))
            p_x_times_p_y = np.tensordot(p_x, p_y, axes = 0)
            info = np.sum(p_xy * np.ma.log(np.ma.divide(p_xy, p_x_times_p_y)))

            if ((1 - (info / entropy)) < 0):
                output = 0.0
            else:
                output = (1 - (info / entropy))

            self.memo[index1] = output
            return output

    def __str__(self):
        print(len(self.memo))

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

def find_ops(old_ops, max_outputs, bins, jump_filename=None):

    mut = Memoizer(bins)
    matrix = DissimilarityMatrix(max_outputs, mut)

    print("Building dissimilarity matrix...")
    for i in old_ops:
        matrix.add_OP(i)

    for i in old_ops[::-1]:
        matrix.add_OP(i)
    print("Done dissimilarity matrix construction...")

    tmp = copy.deepcopy(matrix)
    distortion_array = []
    num_array = []
    str_num_array = []

    while (len(tmp.OPs) > 0):
        print("Checking " + str(len(tmp.OPs)) + " order parameters...")
        num_array.append(len(tmp.OPs))
        str_num_array.append(str(len(tmp.OPs)))
        seed = []
        for i in tmp.OPs:
            seed.append(i)
        tmp_ops = cluster(old_ops, seed, mut)
        distortion_array.append(distortion(tmp_ops, old_ops, mut))
        tmp.reduce()

    jumps = []

    for dim in range(1,11):
        neg_expo = np.array(distortion_array) ** (-0.5 * dim)
        local = []
        for j in range(len(neg_expo) - 1):
            local.append(neg_expo[j] - neg_expo[j + 1])
        jumps.append(local)

    jump_num_array = []
    for i in range(len(str_num_array) - 1):
        jump_num_array.append(str_num_array[i])

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

    if not jump_filename == None:
        np.save(jump_filename, all_jumps)

    while (len(matrix.OPs) > num_ops):
        matrix.reduce()

    return cluster(old_ops, matrix.OPs, mut)
