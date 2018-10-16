import csv
import math

with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/age_5y_data.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    binned_ages = []
    for line in reader:
        newline = list(line)
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
        elif age <= 5:
            binned_age = "1-5"
        elif age <= 10:
            binned_age = "6-10"
        elif age <= 15:
            binned_age = "11-15"
        elif age <= 20:
            binned_age = "16-20"
        elif age <= 25:
            binned_age = "21-25"
        elif age <= 30:
            binned_age = "26-30"
        elif age <= 35:
            binned_age = "31-35"
        elif age <= 40:
            binned_age = "36-40"
        elif age <= 45:
            binned_age = "41-45"
        elif age <= 50:
            binned_age = "46-50"
        elif age <= 55:
            binned_age = "51-55"
        elif age <= 60:
            binned_age = "56-60"
        else:
            binned_age = "60+"

        newline[3] = age
        newline.append(binned_age)

        if newline[3] not in binned_ages:
            binned_ages.append(newline[3])

        writer.writerow(newline)
f.close()
newf.close()
print("End of file.")
