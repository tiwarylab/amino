import subprocess
import numpy as np

gro_file = "npt.gro"
traj_file = "md1.xtc"
plumed_file = "input.plu"
tmp_colvar = ".colvar"
bashCommand = "plumed driver --plumed " + plumed_file + " --mf_xtc " + traj_file
rmCommand = "rm bck.*." + tmp_colvar

# Parse through .gro file and returns array of alpha carbons
def find_alpha_carbons(gro_file):

    f = open(gro_file, "r")
    alpha_carbons = []

    # Reads title string and number of atoms and ignores them
    f.readline()
    f.readline()

    # Iterates through each line corresponding to an atom
    for i in f:

        comps = i.split()

        # Checks if atom is an alpha carbon
        if comps[1] == "CA":
            alpha_carbons.append(int(comps[2]))

    f.close()
    return alpha_carbons

# Binning values of one order parameter to create a 1 dimensional probability space
def d1_bin(x, bins = 80):

    min_val = np.amin(x)
    max_val = np.amax(x)
    span = max_val - min_val

    p_x = [0.0 for i in range(bins)]

    for i in x:
        bin_num = (int) (bins * (i - min_val) / span)
        if bin_num == bins:
            bin_num -= 1
        p_x[bin_num] += 1.0 / len(x)

    return p_x

# Binning valus of two order parameters to create a 2 dimensional probability space
def d2_bin(x, y, bins = 80):

    if len(x) != len(y):
        raise Exception("Order parameter lists are of different size.")

    min_x = np.amin(x)
    max_x = np.amax(x)
    span_x = max_x - min_x

    min_y = np.amin(y)
    max_y = np.amax(y)
    span_y = max_y - min_y

    p_xy = [[0.0 for i in range(bins)] for j in range(bins)]

    for i in range(len(x)):
        bin_x = (int) (bins * (x[i] - min_x) / span_x)
        bin_y = (int) (bins * (y[i] - min_y) / span_y)
        if bin_x == bins:
            bin_x -= 1
        if bin_y == bins:
            bin_y -= 1
        p_xy[bin_x][bin_y] += 1.0 / len(x)

    return p_xy

# Calculates normalized mutual information of two order parameters
def mutual_info(x, y, bins = 100):

    p_x = d1_bin(x, bins)
    p_y = d1_bin(y, bins)
    p_xy = d2_bin(x, y, bins)

    info = 0
    entropy = 0

    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i][j] != 0:
                entropy -= p_xy[i][j] * np.log(p_xy[i][j])
                info += p_xy[i][j] * np.log(p_xy[i][j] / (p_x[i] * p_y[j]))

    return info / entropy

# Finds the number of local maxima in a noisy probability distribution
def find_wells(prob):

    energy = []
    for i in (range(len(prob))):
        if prob[i] == 0:
            energy.append(np.inf)
        else:
            energy.append(-1 * np.log(prob[i]))

    wells = 0
    max = np.inf
    min = np.inf
    d = 1
    i = 0
    for x in energy:
        if x > max:
            max = x
            if (max - min > 1):
                min = x
                d = 1
        elif x < min:
            min = x
            if (max - min > 1):
                if d == 1:
                    wells = wells + 1
                max = x
                d = -1
        i = i + 1

    return wells

class OrderParameter:

    def __init__(self, i, j, traj):
        self.i = i
        self.j = j
        self.traj = traj

class SimilarityMatrix:

    def __init__(self, max_OPs):
        self.max_OPs = max_OPs
        self.matrix = [[] for i in range(max_OPs)]
        self.OPs = []

    def add_OP(self, OP):
        if len(self.OPs) == self.max_OPs:
            mut_info = []
            existing = []
            for i in range(len(self.OPs)):
                mut_info.append(mutual_info(self.OPs[i].traj, OP.traj))
                existing.append(sum(self.matrix[i]) - self.matrix[i][i])
            candidate_info = sum(mut_info)
            update = False
            difference = 0
            for i in range(len(self.OPs)):
                candidate_info = sum(mut_info) - mut_info[i]
                if candidate_info < existing[i]:
                    update = True
                    if difference == 0:
                        difference = existing[i] - candidate_info
                        old_OP = i
                    else:
                        if (existing[i] - candidate_info) < difference:
                            difference = existing[i] - candidate_info
                            old_OP = i
            if update == True:
                mut_info[old_OP] = mutual_info(OP.traj, OP.traj)
                self.matrix[old_OP] = mut_info
                self.OPs[old_OP] = OP
                for i in range(len(self.OPs)):
                    self.matrix[i][old_OP] = mut_info[i]
        else:
            for i in range(len(self.OPs)):
                mut_info = mutual_info(OP.traj, self.OPs[i].traj)
                self.matrix[i].append(mut_info)
                self.matrix[len(self.OPs)].append(mut_info)
            self.matrix[len(self.OPs)].append(mutual_info(OP.traj, OP.traj))
            self.OPs.append(OP)

alpha_carbons = find_alpha_carbons(gro_file)
matrix = SimilarityMatrix(12)
f = open(plumed_file, "w+")
print_line = "PRINT ARG="

for i in range(len(alpha_carbons)):
    print(i)
    for j in range(i + 1, len(alpha_carbons)):
        f.write("d" + str(alpha_carbons[i]) + "_" + str(alpha_carbons[j]) + ": DISTANCE ATOMS=" + str(alpha_carbons[i]) + "," + str(alpha_carbons[j]) + "\n")
        print_line = print_line + "d" + str(alpha_carbons[i]) + "_" + str(alpha_carbons[j]) + ","

print_line = print_line[:-1]
print_line = print_line + " FILE=" + tmp_colvar + " STRIDE=1"
f.write(print_line)
f.close()
process = subprocess.call(bashCommand.split())

'''
    f = open(tmp_colvar, "r")
    f.readline()
    paths = [[] for counter in range(len(alpha_carbons) - i - 1)]
    for line in f:
        timestep = [float(x) for x in line.split()]
        for a in range(1, len(timestep)):
            paths[a - 1].append(timestep[a])

    counter = i + 1

    for path in paths:
        prob = d1_bin(path)
        if find_wells(prob) > 1:
            OP = OrderParameter(alpha_carbons[i], alpha_carbons[counter], path)
            matrix.add_OP(OP)
        counter = counter + 1

    process = subprocess.call(rmCommand.split())

print("ORDER PARAMETERS")
for i in matrix.OPs:
    print(str(i.i) + " " + str(i.j))

print("\nSIMILARITY MATRIX")
print(matrix.matrix)
'''
