import numpy as np

file =  open("BIASED_COLVAR")
file.readline()

t = 0.0
for line in file:
    split = line.split()
    bias = float(split[len(split) - 1])
    t = t + 2 * (np.exp(bias/2.5) / 1e6)

print(t)
