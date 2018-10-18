import math
import os
import shutil
import csv

home_path = "/Users/maggieliuzzi/agerecognition/"
csv_path = "/Users/maggieliuzzi/NeuralNetworks/Adience/adience_age_10_ED_bins.csv"
source_path = '/Users/maggieliuzzi/NeuralNetworks/Adience/'
dataset_path = os.path.join(home_path, "dataset_adience_age_10_ED_bins/")
test_path_t = os.path.join(dataset_path, "test/test/")

train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
test_path = os.path.join(dataset_path, "test/")

train_path_0 = os.path.join(train_path, "1-4") # bin1
train_path_1 = os.path.join(train_path, "5-7") # bin2
train_path_2 = os.path.join(train_path, "8-12") # bin3
train_path_3 = os.path.join(train_path, "13-16") # bin4
train_path_4 = os.path.join(train_path, "17-21") # bin5
train_path_5 = os.path.join(train_path, "22-27") # bin6
train_path_6 = os.path.join(train_path, "28") # bin7 # 3992 rows (double than bin size)
train_path_7 = os.path.join(train_path, "29-35") # bin8
train_path_8 = os.path.join(train_path, "36-46") # bin9
train_path_9 = os.path.join(train_path, "47-60") # bin10

validate_path_0 = os.path.join(validate_path, "1-4")
validate_path_1 = os.path.join(validate_path, "5-7")
validate_path_2 = os.path.join(validate_path, "8-12")
validate_path_3 = os.path.join(validate_path, "13-16")
validate_path_4 = os.path.join(validate_path, "17-21")
validate_path_5 = os.path.join(validate_path, "22-27")
validate_path_6 = os.path.join(validate_path, "28")
validate_path_7 = os.path.join(validate_path, "29-35")
validate_path_8 = os.path.join(validate_path, "36-46")
validate_path_9 = os.path.join(validate_path, "47-60")

test_path_t = os.path.join(test_path, "test")

processed_paths = [train_path_0, train_path_1, train_path_2, train_path_3, train_path_4, train_path_5,
                   train_path_6, train_path_7, train_path_8, train_path_9,
                   validate_path_0, validate_path_1, validate_path_2, validate_path_3, validate_path_4,
                   validate_path_5, validate_path_6, validate_path_7, validate_path_8, validate_path_9,
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
        image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
        age = row[3]
        if age is not None:
            usable_age.append(row)
        gender = row[4]
        if gender is not None:
            usable_gender.append(row)
        role = None
        total_images += 1
    csvfile.close()
    print("Loaded data from " + csv_path)

    for path in processed_paths:
        if not os.path.isdir(path):
            os.makedirs(path)
            print("Created directory: " + path)
        else:
            print("Directory already exists: " + path)

    # Assumes all records with an age value are usable
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
    end_point = len(usable_age)
    file = open(dataset_path+"/labels.csv", "w")
    test_labels_file = open(test_path + "test_labels.csv", "w")

    # For all training images
    for i in range(0, train_images):
        train_images_data.append(usable_age[i])
        role = "train"
        image_folder = usable_age[i][0]
        original_image = usable_age[i][1]
        face_id = usable_age[i][2]
        image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
        age = usable_age[i][3]
        bin = usable_age[i][13]
        filepath = 'faces/' + image_folder + '/' + image_name
        source = os.path.join(source_path, filepath)
        destination = os.path.join(home_path, "dataset_adience_age_10_ED_bins", "train", bin)
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
        age = usable_age[i][3]
        bin = usable_age[i][13]
        filepath = 'faces/' + image_folder + '/' + image_name
        source = os.path.join(source_path, filepath)
        destination = os.path.join(home_path, "dataset_adience_age_10_ED_bins", "validate", bin)
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
        age = usable_age[i][3]
        bin = usable_age[i][13]
        filepath = 'faces/' + image_folder + '/' + image_name
        source = os.path.join(source_path, filepath)
        shutil.copy(source, test_path_t)
        file.write(str(usable_age[i]) + '\n')
        test_labels_file.write(str(usable_age[i]) + '\n')
        current_point += 1

    file.close()
    test_labels_file.close()

    print("Total images: " + str(total_images))
    print("Usable images: " + str(good_images))
    print("Training images: " + str(train_images))
    print("Validation images: " + str(validate_images))
    print("Testing images: " + str(test_images))

print("Processing done.")