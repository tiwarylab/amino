prp = ["1664", "1665", "1666", "1667"]
distances = ""

file = open("em.gro")
plumed = open("plumed.dat", "w+")
counter = 0

for line in file:
    if len(line.split()) == 6 and line.split()[1] == "CA":
        alpha = line.split()[2]
        for i in prp:
            plumed.write("d" + alpha + "_" + i + ": DISTANCE ATOMS=" + alpha + "," + i + "\n")
            distances = distances + "d" + alpha + "_" + i + ","
        counter = counter + 1
    if len(line.split()) > 2 and line.split()[2] == "1663":
        break;

distances = distances[0:(len(distances) - 1)]
plumed.write("PRINT ARG=" + distances + " FILE=COLVAR")

print(counter)
