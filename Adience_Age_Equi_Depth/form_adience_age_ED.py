import csv
import math

with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/adience_age_ED_data.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    ages = []
    for line in reader:
        newline = line
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

        if age > 0 and age is not None:
            newline[3] = age
            print(newline)

        if newline[3] not in ages:
            ages.append(newline[3])

        writer.writerow(newline)
newf.close()
print("End of file.")
# Then removed rows with age -60100 and None manually and overwrote file
