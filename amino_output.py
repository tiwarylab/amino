import amino
import sys

filename = sys.argv[1]

colvar = open(filename)
split = colvar.readline().split()
names = []
trajs = {}

if '-n' in sys.argv:
    num = int(sys.argv[sys.argv.index('-n') + 1])
else:
    num = len(split) - 3

bins = 50

if '-b' in sys.argv:
    bins = int(sys.argv[sys.argv.index('-b') + 1])

if num > 20 and '--override' not in sys.argv:
    num = 20

for i in range(3, len(split)):
    names.append(split[i])
    trajs[split[i]] = []

for line in colvar:
    timestep = line.split()
    for i in range(len(timestep) - 1):
        trajs[names[i]].append(float(timestep[i + 1]))

colvar.close()

ops = [amino.OrderParameter(i, trajs[i]) for i in names]

final_ops = amino.find_ops(ops, num, bins)

print("\nAMINO Order Parameters:")
for i in final_ops:
    print(i)
