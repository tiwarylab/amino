import sys
atoms = [2, 5, 6, 7, 9, 11, 15, 16, 17, 19]
distances = ""
file = open(sys.argv[1], "w+")
for i in range(len(atoms)):
    for j in range(i + 1, len(atoms)):
        x = str(atoms[i])
        y = str(atoms[j])
        distance = "d" + x + "_" + y
        distances += distance + ","
        file.write(distance + ": DISTANCE ATOMS=" + x + "," + y + "\n")

file.write("PRINT ARG=")
file.write(distances[:(len(distances)-1)])
file.write(" FILE=COLVAR")
