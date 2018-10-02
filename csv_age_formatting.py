import csv
import re


with open("/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv",'r') as f, open("/Users/maggieliuzzi/NeuralNetworks/Adience/age_data.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)


    ages = []
    for line in reader:
        age = line[3]
        age.replace("(48 53)", "51")
        age.replace("(4 6)", "5")
        age.replace("(15 20)", "18")
        age.replace("(8 12)", "10")
        age.replace("(38 43)", "41")
        age.replace("(25 32)", "29")
        age.replace("(0 2)", "1")
        age.replace("(38 42)", "40")
        age.replace("(8 23)", "16")
        age.replace("(27 32)", "30")
        age.replace("(38 48)", "43")

        if age not in ages and age > str(0) and age is not None and age is not "None": # remove the None rows
            ages.append(age)
        if age > str(0) and age is not None and age is not "None":
            writer.writerow(line)
        print(sorted(ages))

        '''
        if age not in ages and "(" not in age and age > str(0):
            ages.append(age)
            writer.writerow(line)
        print(ages)
        '''