colvar =  open("RECOLVAR")

for line in colvar:
    split = line.split()
    if (len(split) == 2):
        if (float(split[1]) > 2):
            print(line)
            break;
