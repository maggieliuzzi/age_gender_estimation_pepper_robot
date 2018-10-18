import math
import os
import shutil
import csv
import argparse

parser = argparse.ArgumentParser(
    description="Processes and divides Adience dataset into training, validation and testing sets",
    epilog="Created by Maria Liuzzi & Mitchell Clarke")
parser.add_argument('--bucketsize', default=None, required=True,
                    help="the age range of a single bucket in the prediction; required; must be: 15, 10 or 5.")
args = parser.parse_args()
bucket_size = int(args.bucketsize)


home_path = os.path.dirname(__file__)
# local path to downloaded Adience dataset, i.e. source directories
dataset_path = "/Users/maggieliuzzi/NeuralNetworks/Adience/"
csv_path = os.path.join(dataset_path, "adience_age_"+str(bucket_size)+"y_data.csv")
# path to formatted dataset, i.e. destination directories
form_dataset_path = os.path.join(home_path, "dataset_adience_age_"+str(bucket_size)+"y/")

train_path = os.path.join(form_dataset_path, "train/")
validate_path = os.path.join(form_dataset_path, "validate/")
test_path = os.path.join(form_dataset_path, "test/")
test_path_t = os.path.join(test_path, "test/")
processed_paths = [train_path, validate_path, test_path, test_path_t]

train_path_0 = os.path.join(train_path, "1-"+str(bucket_size)+"/")
train_path_1 = os.path.join(train_path, str(bucket_size + 1)+"-"+str(bucket_size * 2)+"/")
train_path_2 = os.path.join(train_path, str(bucket_size * 2 + 1)+"-"+str(bucket_size * 3)+"/")
train_path_3 = os.path.join(train_path, str(bucket_size * 3 + 1)+"-"+str(bucket_size * 4)+"/")
processed_paths.extend([train_path_0, train_path_1, train_path_2, train_path_3])

if bucket_size < 15:
    train_path_4 = os.path.join(train_path, str(bucket_size * 4 + 1) + "-" + str(bucket_size * 5) + "/")
    train_path_5 = os.path.join(train_path, str(bucket_size * 5 + 1) + "-" + str(bucket_size * 6) + "/")
    processed_paths.extend([train_path_4, train_path_5])

if bucket_size < 10:
    train_path_6 = os.path.join(train_path, str(bucket_size * 6 + 1) + "-" + str(bucket_size * 7) + "/")
    train_path_7 = os.path.join(train_path, str(bucket_size * 7 + 1) + "-" + str(bucket_size * 8) + "/")
    train_path_8 = os.path.join(train_path, str(bucket_size * 8 + 1) + "-" + str(bucket_size * 9) + "/")
    train_path_9 = os.path.join(train_path, str(bucket_size * 9 + 1) + "-" + str(bucket_size * 10) + "/")
    train_path_10 = os.path.join(train_path, str(bucket_size * 10 + 1) + "-" + str(bucket_size * 11) + "/")
    train_path_11 = os.path.join(train_path, str(bucket_size * 11 + 1) + "-" + str(bucket_size * 12) + "/")
    processed_paths.extend([train_path_6, train_path_7, train_path_8, train_path_9, train_path_10, train_path_11])

validate_path_0 = os.path.join(validate_path, "1-"+str(bucket_size)+"/")
validate_path_1 = os.path.join(validate_path, str(bucket_size+1)+"-"+str(bucket_size*2)+"/")
validate_path_2 = os.path.join(validate_path, str(bucket_size*2+1)+"-"+str(bucket_size*3)+"/")
validate_path_3 = os.path.join(validate_path, str(bucket_size*3+1)+"-"+str(bucket_size*4)+"/")
processed_paths.extend([validate_path_0, validate_path_1, validate_path_2, validate_path_3])

if bucket_size < 15:
    validate_path_4 = os.path.join(validate_path, str(bucket_size * 4 + 1) + "-" + str(bucket_size * 5) + "/")
    validate_path_5 = os.path.join(validate_path, str(bucket_size * 5 + 1) + "-" + str(bucket_size * 6) + "/")
    processed_paths.extend([validate_path_4, validate_path_5])

if bucket_size < 10:
    validate_path_6 = os.path.join(validate_path, str(bucket_size * 6 + 1) + "-" + str(bucket_size * 7) + "/")
    validate_path_7 = os.path.join(validate_path, str(bucket_size * 7 + 1) + "-" + str(bucket_size * 8) + "/")
    validate_path_8 = os.path.join(validate_path, str(bucket_size * 8 + 1) + "-" + str(bucket_size * 9) + "/")
    validate_path_9 = os.path.join(validate_path, str(bucket_size * 9 + 1) + "-" + str(bucket_size * 10) + "/")
    validate_path_10 = os.path.join(validate_path, str(bucket_size * 10 + 1) + "-" + str(bucket_size * 11) + "/")
    validate_path_11 = os.path.join(validate_path, str(bucket_size * 11 + 1) + "-" + str(bucket_size * 12) + "/")
    processed_paths.extend([validate_path_6, validate_path_7, validate_path_8, validate_path_9, validate_path_10, validate_path_11])

for path in processed_paths:
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created directory: " + path)
    else:
        print("Directory already exists: " + path)


with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)

    total_images = 0
    usable_age = []
    usable_gender = []
    for row in reader:
        image_folder = row[0]
        original_image = row[1]
        face_id = row[2]
        image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
        actual_age = row[3]
        binned_age = row[12]
        if int(actual_age) > 0 and binned_age is not None:
            usable_age.append(row)
        gender = row[4]
        if gender is not None:
            usable_gender.append(row)
        role = None
        total_images += 1
    print("Loaded data from " + csv_path)

    good_images = len(usable_age)
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(good_images * train_cutoff))
    validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
    test_images = good_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []
    current_point = 0

    with open(form_dataset_path + "labels.csv", "w") as file, open(test_path + "test_labels.csv", "w") as test_labels_file:
        # For all training images
        for i in range(0, train_images):
            train_images_data.append(usable_age[i])
            role = "train"
            image_folder = usable_age[i][0]
            original_image = usable_age[i][1]
            face_id = usable_age[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable_age[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(train_path, age)
            shutil.copy(source, destination)
            file.write(str(usable_age[i]) + '\n')
            current_point += 1
        # For all validation images
        for i in range(train_images, train_images + validate_images):
            validate_images_data.append(usable_age[i])
            role = "validate"
            image_folder = usable_age[i][0]
            original_image = usable_age[i][1]
            face_id = usable_age[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable_age[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(validate_path, age)
            shutil.copy(source, destination)
            file.write(str(usable_age[i]) + '\n')
            current_point += 1
        # For all testing images
        for i in range(train_images + validate_images, good_images):
            test_images_data.append(usable_age[i])
            role = "test"
            image_folder = usable_age[i][0]
            original_image = usable_age[i][1]
            face_id = usable_age[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable_age[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(test_path_t)
            shutil.copy(source, destination)
            file.write(str(usable_age[i]) + '\n')
            test_labels_file.write(str(usable_age[i]) + '\n')
            current_point += 1

    print("Total images: " + str(total_images))
    print("Usable images: " + str(good_images))
    print("Training images: " + str(len(train_images_data)))
    print("Validation images: " + str(len(validate_images_data)))
    print("Testing images: " + str(len(test_images_data)))

print("Processing done.")
