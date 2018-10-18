import csv
import math

with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/adience_age_15y_data.csv",'w') as newf:
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
        elif age <= 15:
            binned_age = "1-15"
        elif age <= 30:
            binned_age = "16-30"
        elif age <= 45:
            binned_age = "31-45"
        elif age <= 60:
            binned_age = "46-60"
        else:
            binned_age = "60+"

        newline[3] = age
        newline.append(binned_age)

        writer.writerow(newline)
newf.close()
print("End of file.")
