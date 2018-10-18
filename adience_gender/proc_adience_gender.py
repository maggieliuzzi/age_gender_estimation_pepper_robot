import math
import os
import shutil
import csv

home_path = os.path.dirname(__file__)
# local path to data.csv in Adience dataset
source_path = '/Users/maggieliuzzi/NeuralNetworks/Adience/'
csv_path = os.path.join(source_path, "data.csv")
dataset_path = os.path.join(home_path, "dataset_adience_gender/")

train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
train_path_0 = os.path.join(train_path, "f")
train_path_1 = os.path.join(train_path, "m")
validate_path_0 = os.path.join(validate_path, "f")
validate_path_1 = os.path.join(validate_path, "m")
test_path_t = os.path.join(dataset_path, "test", "test")
test_path = os.path.join(dataset_path, "test/")
dataset_path_tree = [train_path_0, train_path_1, validate_path_0, validate_path_1, test_path, test_path_t]

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
            usable_gender.append(row)
        role = None
        total_images += 1

    print("Loaded data from " + csv_path)

    for path in dataset_path_tree:
        if not os.path.isdir(path):
            os.makedirs(path)
            print("Created directory: " + path)
        else:
            print("Directory already exists: " + path)

    # Assumes all records with an age value are usable
    good_images = len(usable_gender)
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(good_images * train_cutoff))
    validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
    test_images = good_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []

    current_point = 0
    end_point = len(usable_gender)
    with open("labels.csv", "w") as file, open(test_path + "test_labels.csv", "w") as test_labels_file:

        # For all training images
        for i in range(0,train_images):
            train_images_data.append(usable_gender[i])
            role = "train"
            image_folder = usable_gender[i][0]
            original_image = usable_gender[i][1]
            face_id = usable_gender[i][2]
            image_name = "coarse_tilt_aligned_face."+face_id+"."+original_image
            gender = usable_gender[i][4]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(source_path, filepath)
            destination = os.path.join(home_path, "dataset_adience_gender", "train", gender)
            shutil.copy(source, destination)
            file.write(str(usable_gender[i]) + '\n')
            current_point += 1
        # For all validation images
        for i in range(train_images,train_images + validate_images):
            validate_images_data.append(usable_gender[i])
            role = "validate"
            image_folder = usable_gender[i][0]
            original_image = usable_gender[i][1]
            face_id = usable_gender[i][2]
            image_name = "coarse_tilt_aligned_face."+face_id+"."+original_image
            gender = usable_gender[i][4]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(source_path, filepath)
            destination = os.path.join(home_path, "dataset_adience_gender", "validate", gender)
            shutil.copy(source, destination)
            file.write(str(usable_gender[i]) + '\n')
            current_point += 1
        # For all testing images
        for i in range(train_images + validate_images,good_images):
            test_images_data.append(usable_gender[i])
            role = "test"
            image_folder = usable_gender[i][0]
            original_image = usable_gender[i][1]
            face_id = usable_gender[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            gender = usable_gender[i][4]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(source_path, filepath)
            shutil.copy(source, test_path_t)
            file.write(str(usable_gender[i]) + '\n')
            test_labels_file.write(str(usable_age[i]) + '\n')
            current_point += 1

    print("Total images: " + str(total_images))
    print("Usable images: " + str(good_images))
    print("Training images: " + str(train_images))
    print("Validation images: " + str(validate_images))
    print("Testing images: " + str(test_images))

print("Processing done.")
