import csv
import math

with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/adience_age_10y_data.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    binned_ages = []
    for line in reader:
        newline = list(line)
        f.close()

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
            binned_age = ""
        elif age <= 10:
            binned_age = "1-10"
        elif age <= 20:
            binned_age = "11-20"
        elif age <= 30:
            binned_age = "21-30"
        elif age <= 40:
            binned_age = "31-40"
        elif age <= 50:
            binned_age = "41-50"
        elif age <= 60:
            binned_age = "51-60"
        else:
            binned_age = "60+"

        newline[3] = age
        newline.append(binned_age)

        if newline[3] not in binned_ages:
            binned_ages.append(newline[3])

        writer.writerow(newline)
newf.close()
print("End of file.")