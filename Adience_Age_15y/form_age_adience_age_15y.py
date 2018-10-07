import csv
# import re
import math

with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/adience_age_15y_data.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)


    ages = []
    for line in reader:
        newline = line
        age = newline[3]

        if age[0] == "N":
            age = "0"
        elif age[0] == "(":
            age = age[1:-1]
            age = age.split()
            age = [int(i) for i in age]
            age = math.floor(sum(age) / 2)
        age = int(age)

        if age <= 0:
            age = ""
        elif age <= 15:
            age = "1-15"
        elif age <= 30:
            age = "16-30"
        elif age <= 45:
            age = "31-45"
        elif age <= 60:
            age = "46-60"
        else:
            age = "60+"

        newline[3] = age
        print(newline)

        if newline[3] not in ages:
            ages.append(newline[3])
        # print(ages)

        writer.writerow(newline)


f.close()
newf.close()

print("End of file.")