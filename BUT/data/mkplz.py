file = open("20COLVAR_OUTPUT")
file.readline()
outfile = open("SIGMA", "w+")
weights = [1.749, 1.096, -0.916, 1.591, 0.824, -0.529, -1.364, -1.453]
for line in file:
    s = line.split()
    colvar = 0.0
    for i in range(len(s) - 1):
        colvar = colvar + weights[i] * float(s[i+1])
    outfile.write(str(colvar) + "\n")
outfile.close()
file.close()

