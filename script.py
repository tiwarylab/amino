import numpy as np

file =  open("BIASED_COLVAR")
file.readline()

t = 0.0
for line in file:
    split = line.split()
    bias = float(split[len(split) - 1])
    t = t + 0.002 * ((np.e ** ((bias/2.5)/1e9)))
    print(np.e ** ((bias/2.5)/1e9))

print(t)
