import argparse
import math
import os
# import requests
import shutil
# import tarfile # Delete
import csv
# import numpy as np

'''
Parse command line arguments
'''
parser = argparse.ArgumentParser(
    description="Converts the Adience datasets into a format required for training of neural networks",
    epilog="Created by Maria Liuzzi & Mitchell Clarke")
parser.add_argument('--path', default=None,
                    help="path to the dataset root directory; if omitted, it will download the dataset into the project folder.")
parser.add_argument('--score', default=1.5, help="minimum face score needed to include image; defaults to 1.5.")
args = parser.parse_args()


home_path = "/Users/maggieliuzzi/agerecognition/"
csv_path = "/Users/maggieliuzzi/NeuralNetworks/Adience/data.csv"
zip_path = "/Users/maggieliuzzi/NeuralNetworks/adience.zip"
source_path = '/Users/maggieliuzzi/NeuralNetworks/Adience/'

'''
Define the processed dataset's file structure
'''
dataset_path = os.path.join(home_path, "dataset_adience")
train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
train_path_0 = os.path.join(train_path, "0")
train_path_1 = os.path.join(train_path, "1")
validate_path_0 = os.path.join(validate_path, "0")
validate_path_1 = os.path.join(validate_path, "1")
test_path = os.path.join(dataset_path, "test", "test")
dataset_path_tree = [train_path_0, train_path_1, validate_path_0, validate_path_1, test_path]


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
        filepath = 'faces/' + image_folder + '/' + image_name
        print(filepath)

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

        print(total_images)
        print(type(usable_gender))
        print(usable_gender)

        exit(0)
        good_images_age = len(usable_age)
        good_images_gender = len(usable_gender)

        train_cutoff = 0.64
        validate_cutoff = 0.8

        '''
        # Repeat with Age:
        train_images = int(math.floor(good_images * train_cutoff))
        validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
        test_images = good_images - (train_images + validate_images)
        '''

        print(good_images_gender)
        print(train_cutoff)
        print(validate_cutoff)
        # Gender:
        train_images = int(math.floor(good_images_gender * train_cutoff))
        validate_images = int(math.floor(good_images_gender * (validate_cutoff - train_cutoff)))
        test_images = good_images_gender - (train_images + validate_images)

        print(train_images)
        print(validate_images)
        print(test_images)

        for i in range(0, train_images):
            role = "train"
        for i in range(train_images, train_images + validate_images):
            role = "validate"
        for i in range(train_images + validate_images, good_images_gender):
            role = "test"


        print("Total images: " + str(total_images))
        print("Usable images: " + str(good_images_gender))
        print("Training images: " + str(train_images))
        print("Validation images: " + str(validate_images))
        print("Testing images: " + str(test_images))

        '''
        Create dataset directory, the 'train', 'validate' and 'test' subfolders
        '''
        for path in dataset_path_tree:
            if not os.path.isdir(path):
                os.makedirs(path)
                print("Created directory: " + path)
            else:
                print("Directory already exists: " + path)

        '''
        Populate folders the folders with the images from the dataset and dump labels to text for later
        '''
        current_point = 0
        end_point = len(usable_gender)
        file = open("dataset_adience/labels.csv", "w") # Instead of .txt
        print("Copying image files to new locations...")


        print(source_path)

        print(type(usable_gender))

        usable_gender = "".join(str(usable_gender)).replace("'", "")
        print(usable_gender)

        # print("".join(str(usable_gender)))

        exit(0)

        for image in usable_gender:
            # eg. faces/114978798@N03/coarse_tilt_aligned_face.823.12071867334_2c8265da7c_o.jpg | 7153718@N04
            # Real example: coarse_tilt_aligned_face.909.11129028253_2e9d0c4b18_o.jpg
            print(filepath)
            print(source_path)
            source = os.path.join(source_path, filepath)
            print(source)
            # print(type(gender))
            # print(home_path)
            # print(gender)
            # print(role)
            print(gender)
            print(role)
            destination = os.path.join(home_path, "dataset_adience", role,
                                       gender if role != "test" else "test")
            print(destination)
            shutil.copy(source, destination)
            # reduced_image = dict((key, value) for key, value in image.items() if key in ('filepath', 'gender', 'role'))
            file.write(str(image) + '\n')
            current_point += 1
            print(current_point) # bar.show(current_point)
        file.close()
        print("Copied.")