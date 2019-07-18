import sys
filename = sys.argv[1]
inf = open(filename + "o")
out = open(filename, "w+")
inf.readline()
for line in inf:
    sp = line.split()
    for i in range(1,len(sp)):
        out.write(sp[i] + " ")
    out.write("\n")
