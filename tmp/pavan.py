f = open("COL")
f.readline()
bound = True
reassoc = 0
for line in f:
    s = line.split()
    if (float(s[1]) > 1.6) and bound == True:
        print("UNBINDING " + s[0])
        bound = False
    elif (float(s[1]) < 1.6) and bound == False:
        print("BINDING " + s[0])
        reassoc = reassoc + 1
        bound = True
print(reassoc)
