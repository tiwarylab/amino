import os.path
import argparse
import numpy as np
from math import log, exp, ceil

# Default Arguments
gamma = 15                    # Biasfactor in well-tempered metadynamics.
kT = 2.5                      # Temperature times Boltzmann constant.
fesfilename = "fes_" # FES file name start.
col_fe = 1                    # Column of free energy.
datafile = "../b0-03/BIASED_COLVAR"           # COLVAR file name.
col_rewt = [2]          # COLVAR columns corresponding to RC variables.
col_bias = [3]                # COLVAR bias column.
ngrid = 30                    # Number of grid bins.

def ebc(num):
    global colvar,ebetac
    # File Inputs
    tmp = 2500000 * (num + 1) + 1
    colvar = np.loadtxt(datafile, max_rows=tmp)

    # Calculating c(t):
    # calculates ebetac = exp(beta c(t)), using eq. 12 in eq. 3 in the JPCB paper
    #
    ebetac = []

    for i in range(num + 1):
        # set appropriate format for FES file names, NB: i starts from 0
        fname = '%s%d.dat' % (fesfilename,i)

        data = np.loadtxt(fname)
        s1, s2 = 0., 0.
        for p in data:
            exponent = -p[col_fe]/kT
            s1 += exp(exponent)
            s2 += exp(exponent/gamma)
        ebetac += s1 / s2,

def weights(num):
    # compute reweighting weights from FES data and bias column
    global kT, col_bias, weight
    numcolv = np.shape(colvar)[0]
    weight = np.zeros(numcolv)

    # go through the CV(t) trajectory
    i = 0
    for row in colvar:
        i += 1
        indx = int(ceil(float(i)/numcolv*(num + 1)))-1
        bias = sum([row[j] for j in col_bias])
        ebias = exp(bias/kT)/ebetac[indx]
        weight[i-1] = ebias

def load(num):
    # Loads given files. Runs on import to prevent redundant loading, note this increases import time.
    ebc(num)
    weights(num)

def reweight(sparse=False,size=30,data=None):
    # Reweighting biased MD trajectory to unbiased probabilty along a given RC.
    # By default (sparse=False) bins on the edge of the range with probabilities lower
    # than 1/N where N is number of data points will be removed.
    global kT, fesfilename, col_fe, datafile, col_rewt, numrewt, col_bias, ngrid, s_min, s_max

    rc = [1]

    if data != None:
        datafile = data
        load()

    rc_space = np.dot(colvar[:,col_rewt],rc)
    s_max = np.max(rc_space)
    s_min = np.min(rc_space)



    # initialize square array numrewt-dimensional


    hist = np.histogram(rc_space,size,weights=weight)[0]
    pnorm = hist/np.sum(hist)



    # Trimming off probability values less than one data point could provide
    if not sparse:
        cutoff = 1/np.shape(colvar)[0]
        trim = np.nonzero(pnorm >= cutoff)
        trimmed = pnorm[np.min(trim):np.max(trim)+1]
        if np.min(trimmed) < cutoff:
            cutoff = np.min(trimmed)
            trim = np.nonzero(pnorm >= cutoff)
            trimmed = pnorm[np.min(trim):np.max(trim)+1]
        return trimmed
    return pnorm

vals= []
for i in range(1000):
    load(i)
    plz = reweight()
    vals.append(plz)

np.save("RESULTS", vals)
