file = open("RC")
file.readline()
output = open("tmp", "w+")

for line in file:
    s = line.split()
    s = s[1:len(s)]
    new_line = ""
    for num in s:
        new_line = new_line + str(num) + " "
    output.write(new_line + "\n")
