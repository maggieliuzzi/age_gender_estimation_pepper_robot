import csv
import argparse
import math


def edit_csv(path, name):
    with open(path+name, 'r') as raw_file, open(path+'ready_data.csv', 'w') as ready_file:
        reader = csv.reader(raw_file)
        writer = csv.writer(ready_file)

        for line in reader:
            ready_line = list(line)
            raw_age = ready_line[3]

            if "-" in raw_age or raw_age == None:
                pass
            elif " " in raw_age:
                age = raw_age[1:-1]
                age = age.split()
                age = [int(i) for i in age]
                age = math.floor(sum(age) / 2)
                ready_line[3] = age
            else:
                ready_line = line
                writer.writerow(ready_line)

    print("End of file.")
    return ready_file



parser = argparse.ArgumentParser(description='Edit CSV file')
parser.add_argument('--path', metavar='PATH', help='Path to folder that contains CSV file', required=True)
parser.add_argument('--name', help='Name of CSV file to be edited', required=True)
args = parser.parse_args()
path = args.path
name = args.name

edit_csv(path, name)

# Eg. --path /Users/maggieliuzzi/machine-learning/datasets/Adience/ --name data.csv