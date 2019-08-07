""" Script that evaluates reaction coordinates using the SGOOP method. 
Probabilites are calculated using MD trajectories. Transition rates are
found using the maximum caliber approach.  
For unbiased simulations use rc_eval().
For biased simulations calculate unbiased probabilities and analyze then with sgoop().

The original method was published by Tiwary and Berne, PNAS 2016, 113, 2839.

Author: Zachary Smith                   zsmith7@terpmail.umd.edu
Original Algorithm: Pratyush Tiwary     ptiwary@umd.edu 
Contributor: Pablo Bravo Collado        ptbravo@uc.cl"""

import numpy as np
import scipy.optimize as opt
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



"""User Defined Variables"""
in_file = 'data/RC' # Input file
rc_bin = 20         # Bins over RC
wells = 2           # Expected number of wells with barriers > kT
d = 1               # Distance between indexes for transition
prob_cutoff = 1e-5  # Minimum nonzero probability

"""Auxiliary Variables"""
SG = []             # List of Spectral Gaps
RC = []             # List of Reaction Coordinates
P = []              # List of probabilites on RC
SEE = []            # SGOOP Eigen exp
SEV = []            # SGOOP Eigen values
SEVE = []           # SGOOP Eigen vectors

"""Load MD File"""
data_array = np.loadtxt(in_file)



def rei():
    # Reinitializes arrays for new runs
    global SG,RC,P,SEE,SEV,SEVE
    SG = []
    RC = []         
    P = []              
    SEE = []            
    SEV = []            
    SEVE = []       



def normalize_rc(rc):
    # Normalizes input RC
    squares=0
    for i in rc:
        squares+=i**2
    denom=np.sqrt(squares)
    return np.array(rc)/denom



def generate_rc(i):
    # Generates a unit vector with angle pi*i
    x=np.cos(np.pi*i)
    y=np.sin(np.pi*i)
    return (x,y)



def md_prob(rc,rc_bin=20,show_binned=False):
    # Calculates probability along a given RC
    proj=np.dot(data_array,rc)
    hist,bins=np.histogram(proj,rc_bin)
    hist=hist/np.sum(hist)
    
    if show_binned:
        binned=np.digitize(proj,bins[:-1])-1 
        return hist,binned
    else:
        return hist


    
def set_bins(rc,bins,rc_min,rc_max):  
    # Sets bins from an external source
    global binned, rc_bin
    rc_bin = bins
    proj = np.dot(data_array,rc)
    binned=(proj-rc_min)/(rc_max-rc_min)*(rc_bin-1)
    binned=np.array(binned).astype(int)



def eigeneval(matrix):
    # Returns eigenvalues, eigenvectors, and negative exponents of eigenvalues
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    idx = eigenValues.argsort()     # Sorting by eigenvalues
    eigenValues = eigenValues[idx]  # Order eigenvalues
    eigenVectors = eigenVectors[:,idx]  # Order eigenvectors
    eigenExp = np.exp(-eigenValues)     # Calculate exponentials
    return eigenValues, eigenExp, eigenVectors



def mu_factor(binned,p):
    # Calculates the prefactor on SGOOP for a given RC
    # Returns the mu factor associated with the RC
    # NOTE: mu factor depends on the choice of RC!
    # <N>, number of neighbouring transitions on each RC
    D=0
    N_mean = np.sum(np.abs(binned[:-1]-binned[1:]) <= d)/len(binned)

    for i in np.array(range(d))+1:
        D += np.sum(np.sqrt(p[:-i]*p[i:]))*2

    MU = N_mean/D

    return MU



def transmat(MU,p):
    # Generates transition matrix
    S = np.zeros([rc_bin, rc_bin])
    for step in np.array(range(d))+1: # terms <= d from diagonal
        i=np.linspace(0,rc_bin-step-1,rc_bin-step).astype('int')
        j=i+step
        S[i,j] = -MU * np.sqrt(np.ma.divide(p[i],p[j]))
        S[j,i] = -MU * np.sqrt(np.ma.divide(p[j],p[i]))

    S[np.diag_indices(rc_bin)] = -np.sum(S,axis=0) # Diagonal terms
     
    return S



def spectral():
    # Calculates spectral gap for appropriate number of wells
    SEE_pos=SEE[-1][SEV[-1]>-1e-10] # Removing negative eigenvalues
    SEE_pos=SEE_pos[SEE_pos>0] # Removing negative exponents
    gaps=SEE_pos[:-1]-SEE_pos[1:]
    if np.shape(gaps)[0]>=wells:
        return gaps[wells-1]
    else: 
        return 0



def sgoop(rc,p,binned):
    # SGOOP for a given probability density on a given RC
    # Start here when using probability from an external source
    MU = mu_factor(binned,p) # Calculated with MaxCal approach

    S = transmat(MU,p)       # Generating the transition matrix
    
    sev, see, seve = eigeneval(S) # Calculating eigenvalues and vectors for the transition matrix
    SEV.append(sev)               # Recording values for later analysis
    SEE.append(see)
    SEVE.append(seve)
    
    sg = spectral() # Calculating the spectral gap
    SG.append(sg)
    
    return sg



def biased_prob(rc,bias_rcs,rc_bin=20,show_binned=False):
    # calculates the probability along a given RC conditional on the probability along a given set of RCs
    bias_rcs=np.array(bias_rcs)

    if np.shape(np.shape(bias_rcs))[0]==1:
        bias_rcs = np.array(bias_rcs).reshape(1,np.shape(bias_rcs)[0])

    dim = np.shape(bias_rcs)[0]

    bias_prob=np.zeros((dim,rc_bin))
    bias_binned=np.zeros((dim,np.shape(data_array)[0]))


    for i in range(dim):
        bias_prob[i,:],bias_binned[i,:]=md_prob(bias_rcs[i,:],show_binned=True)

    point_probs=np.zeros(np.shape(bias_binned))

    for i in range(np.shape(point_probs)[0]):
        point_probs[i,:]=bias_prob[i,:][bias_binned[i,:].astype('int')]
    point_bias=1/np.product(point_probs,axis=0)

    proj=np.dot(data_array,rc)
    hist,bins=np.histogram(proj,rc_bin,weights=point_bias)
    hist=hist/np.sum(hist)
    
    if show_binned:
        binned=np.digitize(proj,bins[:-1])-1 
        return hist,binned
    else:
        return hist



def rc_eval(rc):
    # Unbiased SGOOP on a given RC
    # Input type: array of weights
    
    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob,binned=md_prob(rc,show_binned=True)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(rc,prob,binned)
    
    return sg



def biased_eval(rc,bias_rcs):
    # Biased SGOOP on a given RC with bias along a second RC
    # Input type: array of weights, probability from original RC
    
    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob,binned=biased_prob(rc,bias_rcs,show_binned=True)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(rc,prob,binned)
    
    return sg

