# The purpose of this file is to process the dataset in preparation for training
# It will download and extract the dataset if necessary.
# Ensure that age_data.csv is in the root of the dataset directory.

# from scipy.io import loadmat # Delete
# from clint.textui import progress
import argparse
import math
import os
# import requests
import shutil
# import tarfile # Delete
import csv
# import numpy as np

parser = argparse.ArgumentParser(
    description="Converts the Adience datasets into a format required for training of neural networks",
    epilog="Created by Maria Liuzzi & Mitchell Clarke")
parser.add_argument('--path', default=None,
                    help="path to the dataset root directory; if omitted, it will download the dataset into the project folder.")
parser.add_argument('--score', default=1.5, help="minimum face score needed to include image; defaults to 1.5.")
args = parser.parse_args()

try:
    n = float(args.score)
    if n < 0:
        print("ERROR: Minimum face score must be greater than or equal to 0.")
        exit(1)
    print("Minimum acceptable face score:", n)
except ValueError:
    print("ERROR: Face score must be a valid float.")
    exit(1)

home_path = "/Users/maggieliuzzi/agerecognition/"
csv_path = "/Users/maggieliuzzi/NeuralNetworks/Adience/age_data.csv"
zip_path = "/Users/maggieliuzzi/NeuralNetworks/adience.zip"
source_path = '/Users/maggieliuzzi/NeuralNetworks/Adience/'
dataset_path = os.path.join(home_path, "dataset_adience_age")

train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
test_path = os.path.join(dataset_path, "test", "test")

train_path_0 = os.path.join(train_path, "2")
train_path_1 = os.path.join(train_path, "3")
train_path_2 = os.path.join(train_path, "13")
train_path_3 = os.path.join(train_path, "22")
train_path_4 = os.path.join(train_path, "23")
train_path_5 = os.path.join(train_path, "29")
train_path_6 = os.path.join(train_path, "32")
train_path_7 = os.path.join(train_path, "34")
train_path_8 = os.path.join(train_path, "35")
train_path_9 = os.path.join(train_path, "36")
train_path_10 = os.path.join(train_path, "42")
train_path_11 = os.path.join(train_path, "45")
train_path_12 = os.path.join(train_path, "46")
train_path_13 = os.path.join(train_path, "55")
train_path_14 = os.path.join(train_path, "56")
train_path_15 = os.path.join(train_path, "57")
train_path_16 = os.path.join(train_path, "58")

validate_path_0 = os.path.join(validate_path, "2")
validate_path_1 = os.path.join(validate_path, "3")
validate_path_2 = os.path.join(validate_path, "13")
validate_path_3 = os.path.join(validate_path, "22")
validate_path_4 = os.path.join(validate_path, "23")
validate_path_5 = os.path.join(validate_path, "29")
validate_path_6 = os.path.join(validate_path, "32")
validate_path_7 = os.path.join(validate_path, "34")
validate_path_8 = os.path.join(validate_path, "35")
validate_path_9 = os.path.join(validate_path, "36")
validate_path_10 = os.path.join(validate_path, "42")
validate_path_11 = os.path.join(validate_path, "45")
validate_path_12 = os.path.join(validate_path, "46")
validate_path_13 = os.path.join(validate_path, "55")
validate_path_14 = os.path.join(validate_path, "56")
validate_path_15 = os.path.join(validate_path, "57")
validate_path_16 = os.path.join(validate_path, "58")

test_path_t = os.path.join(test_path, "test")

processed_paths = [train_path_0, train_path_1, train_path_2, train_path_3, train_path_4, train_path_5, train_path_6,
                    train_path_7, train_path_8, train_path_9, train_path_10, train_path_11, train_path_12, train_path_13,
                   train_path_14, train_path_15, train_path_16,
                   validate_path_0, validate_path_1, validate_path_2, validate_path_3, validate_path_4, validate_path_5,
                   validate_path_6, validate_path_7, validate_path_8, validate_path_9, validate_path_10, validate_path_11,
                    validate_path_12, validate_path_13, validate_path_14, validate_path_15, validate_path_16,
                   test_path_t]


with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    total_images = 0
    usable_age = []
    usable_gender = []
    for row in reader:
        image_folder = row[0]
        original_image = row[1]
        face_id = row[2]
        image_name = "coarse_tilt_aligned_face."+face_id+"."+original_image
        age = row[3]
        if age is not None:
            usable_age.append(row)
        gender = row[4]
        if gender is not None:
            print(row)
            usable_gender.append(row)
        role = None
        total_images += 1

    print("Loaded data from " + csv_path)



    for path in processed_paths:
        if not os.path.isdir(path):
            os.makedirs(path)
            print("Created directory: " + path)
        else:
            print("Directory already exists: " + path)



    good_images = len(usable_age)  # number
    # problematic - fix!
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(good_images * train_cutoff))
    validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
    test_images = good_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []

    current_point = 0
    end_point = len(usable_age)
    file = open("dataset_adience_age/labels.csv", "w")  # Instead of .txt

    for i in range(0,train_images):
        print(i)        # if count <= train_images:
        train_images_data.append(usable_age[i])
        role = "train"
        print(role)
        image_folder = usable_age[i][0]
        original_image = usable_age[i][1]
        face_id = usable_age[i][2]
        image_name = "coarse_tilt_aligned_face."+face_id+"."+original_image
        age = usable_age[i][3]
        print(age)
        filepath = 'faces/' + image_folder + '/' + image_name
        print(filepath)
        source = os.path.join(source_path, filepath)
        print(source)
        destination = os.path.join(home_path, "dataset_adience_age", "train", age)
        print(destination)
        shutil.copy(source, destination)
        file.write(str(usable_age[i]) + '\n')
        current_point += 1
        print(current_point)  # bar.show(current_point)
    for i in range(train_images,train_images + validate_images):
    # elif count <= (train_images + validate_images):
        validate_images_data.append(usable_age[i])
        role = "validate"
        print(role)
        image_folder = usable_age[i][0]
        original_image = usable_age[i][1]
        face_id = usable_age[i][2]
        image_name = "coarse_tilt_aligned_face."+face_id+"."+original_image
        age = usable_age[i][3]
        print(age)
        filepath = 'faces/' + image_folder + '/' + image_name
        print(filepath)
        source = os.path.join(source_path, filepath)
        print(source)
        destination = os.path.join(home_path, "dataset_adience_age", "validate", age)
        print(destination)
        shutil.copy(source, destination)
        file.write(str(usable_age[i]) + '\n')
        current_point += 1
        print(current_point) # bar.show(current_point)
    for i in range(train_images + validate_images,good_images):
        test_images_data.append(usable_age[i])
        role = "test"
        print(role)
        image_folder = usable_age[i][0]
        original_image = usable_age[i][1]
        face_id = usable_age[i][2]
        image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
        age = usable_age[i][3]
        print(age)
        filepath = 'faces/' + image_folder + '/' + image_name
        print(filepath)
        source = os.path.join(source_path, filepath)
        print(source)
        destination = os.path.join(home_path, "dataset_adience_age", "test", "test")
        print(destination)
        shutil.copy(source, destination)
        file.write(str(usable_age[i]) + '\n')
        current_point += 1
        print(current_point)  # bar.show(current_point)

    print("Total images: " + str(total_images))
    print("Usable images: " + str(good_images))
    print("Training images: " + str(train_images))
    print("Validation images: " + str(validate_images))
    print("Testing images: " + str(test_images))

    '''
    print("Train:")
    print(train_images_data)
    print("Validate:")
    print(validate_images_data)
    print("Test:")
    print(test_images_data)
    '''

    file.close()

csvfile.close()
print("Copied.")