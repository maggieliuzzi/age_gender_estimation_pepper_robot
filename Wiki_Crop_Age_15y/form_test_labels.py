import csv
import math

with open("/Users/maggieliuzzi/agerecognition/wiki_dataset/test/test_labels.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/wiki_dataset/test/form_test_labels.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    for line in reader:
        newline = list(line)

        filepath = newline[4]
        gender = newline[6]
        age = newline[7]

        filepath = filepath.split("/",1)[1]
        filepath = filepath.replace("'","")
        print(filepath)

        gender = float(gender[10:])
        gender = int(round(gender,0))
        # print(gender)

        age = age[7:]
        age = int(age)
        # print(age)

        newline[4] = filepath
        newline[6] = gender
        newline[7] = age

        writer.writerow(newline)
f.close()
newf.close()
print("End of file.")
