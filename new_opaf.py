import subprocess
import numpy as np
import sgoop
import scipy.optimize as opt
import time
import random

gro_file = "npt.gro"
traj_file = "md1.xtc"
plumed_file = "input.plu"
tmp_colvar = ".colvar"
shell_file = "plumed.log"
bashCommand = "plumed driver --plumed " + plumed_file + " --mf_xtc " + traj_file
rmCommand = "rm bck.*." + tmp_colvar

class OrderParameter:

    def __init__(self, i, j, traj):
        self.i = i
        self.j = j
        self.traj = traj

    def __str__(self):
        return str(self.i) + " " + str(self.j)

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

    return (1 - (info / entropy))

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

def opti_func(rc):
    global nfev, data_array
    nfev +=1
    return -sgoop.rc_eval(rc, data_array)

def print_fun(x, f, accepted):
    global now,last,nfev,lastf
    now=time.time()
    #print(x,end=' ')
    #if accepted == 1:
    #    print("with spectral gap %.4f accepted after %3i runs (%.3f)" % (-f, nfev-lastf, now-last))
    #else:
    #    print("with spectral gap %.4f declined after %3i runs (%.3f)" % (-f, nfev-lastf, now-last))
    last=now
    lastf=nfev

def initial_guess(wells, num_ops, data_array):

    x = [-1 for i in range(num_ops)]
    finished = [1 for i in range(num_ops)]
    rc = x
    max_val = 0
    sgoop.wells = wells

    while not x == finished:

        tmp = sgoop.rc_eval(x, data_array)
        if tmp > max_val:
            max_val = tmp
            rc = x.copy()

        for i in range(num_ops):
            if x[num_ops - 1 - i] == -1:
                x[num_ops - 1 - i] = 1
                break;
            else:
                x[num_ops - 1 - i] = -1

    tmp = sgoop.rc_eval(x, data_array)
    if tmp > max_val:
        max_val = tmp
        rc = x.copy()

    return rc

def find_ops(old_ops, max_outputs=-1, min_val=-1):
    if max_outputs == -1 and min_val == -1:
        raise Exception("You\'re using find_ops wrong. No reduction will be done.'")
    elif max_outputs < 2:
        max_outputs = len(old_ops)
    elif min_val == -1:
        min_val = 0
    most = 0
    vals = [0, 0]
    for i in range(len(old_ops)):
        for j in range(i + 1, len(old_ops)):
            tmp = mutual_info(old_ops[i].traj, old_ops[j].traj)
            if tmp > most:
                most = tmp
                vals = [i, j]

    new_ops = [old_ops[vals[0]], old_ops[vals[1]]]

    info_check = True

    while ((len(new_ops) < max_outputs) and (info_check)):

        print("ITERATION:")
        for i in new_ops:
            print(i)
        print("FIN")
        print()
        global data_array
        data_array = []
        for i in range(len(new_ops[0].traj)):
            x = [j.traj[i] for j in new_ops]
            data_array.append(x.copy())
        consistent = True
        candidate_wells = 2

        while (consistent == True):

            print("Testing " + str(candidate_wells) + " wells")

            global nfev, lastf, last

            guess = initial_guess(candidate_wells, len(new_ops), data_array)

            sgoop.wells = candidate_wells
            start = time.time()
            last = start
            lastf = nfev = 0
            minimizer_kwargs = {"options": {"maxiter":10}}
            ret = opt.basinhopping(opti_func,guess,niter=100,T=.01,stepsize=.1, minimizer_kwargs=minimizer_kwargs, callback=print_fun)
            end = time.time()
            prob_space = sgoop.md_prob(ret.x, data_array)

            if (find_wells(prob_space) >= candidate_wells):
                print(str(candidate_wells) + " well runs accepted. Testing " + str(candidate_wells + 1) + " well RC.")
                rc = ret.x.copy()
                candidate_wells = candidate_wells + 1
            else:
                print(str(candidate_wells) + " well runs failed. Reverting to previous RC.")
                if candidate_wells == 2:
                    rc = ret.x.copy()
                consistent = False

        proj = []
        for v in data_array:
            proj.append(np.dot(np.array(v),rc))
        most = 0
        newest_op = None

        for i in old_ops:
            exists = False
            for j in new_ops:
                if i.i == j.i and i.j == j.j:
                    exists = True
                    break;
            if exists == True:
                continue;
            tmp = mutual_info(i.traj, proj)
            if tmp > most:
                most = tmp
                newest_op = i
        if most > min_val:
            new_ops.append(newest_op)

    return new_ops

plumed_log = open(shell_file, "w+")
alpha_carbons = find_alpha_carbons(gro_file)
all_ops = []

for i in range(len(alpha_carbons)):
    print(i)
    f = open(plumed_file, "w+")
    for j in range(i + 1, len(alpha_carbons)):
        f.write("d" + str(j) + ": DISTANCE ATOMS=" + str(alpha_carbons[i]) + "," + str(alpha_carbons[j]) + "\n")
    f.write("PRINT ARG=")
    for j in range(i + 1, len(alpha_carbons)):
        f.write("d" + str(j))
        if j != len(alpha_carbons) - 1:
            f.write(",")
    f.write(" FILE=" + tmp_colvar + " STRIDE=1")
    f.close()
    process = subprocess.call(bashCommand.split(), stdout = plumed_log, shell = True)

    f = open(tmp_colvar, "r")
    f.readline()
    paths = [[] for counter in range(len(alpha_carbons) - i - 1)]
    for line in f:
        timestep = [float(x) for x in line.split()]
        for a in range(1, len(timestep)):
            paths[a - 1].append(timestep[a])

    counter = i + 1

    for path in paths:
        prob = d1_bin(path, bins=20)
        if find_wells(prob) > 1:
            OP = OrderParameter(alpha_carbons[i], alpha_carbons[counter], path)
            all_ops.append(OP)
        counter = counter + 1

    process = subprocess.call(rmCommand.split(), shell = True)

new_ops = find_ops(all_ops, max_outputs=5)
print("HERE ARE THE SELECTED OPS:")
for i in new_ops:
    print(i)