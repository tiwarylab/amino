file = open("SIGMA")
sigma = 0.0
count = 0
for line in file:
    sigma = sigma + float(line)
    count = count + 1
mean = sigma / count
file.close()
file = open("SIGMA")
sigma = 0.0
for line in file:
    sigma = sigma + ((float(line) - mean) * (float(line) - mean))
sigma = sigma / count
sigma = sigma ** (0.5)
file.close()
print(sigma)
