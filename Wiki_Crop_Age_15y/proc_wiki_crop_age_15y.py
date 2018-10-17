from scipy.io import loadmat
import os
import shutil
import math
from datetime import datetime

source_path = "/Users/maggieliuzzi/NeuralNetworks/wiki_crop"
home_path = "/Users/maggieliuzzi/agerecognition/"
mat_path = os.path.join(source_path, "wiki.mat")

train_path = os.path.join(home_path, "wiki_dataset/train")
validate_path = os.path.join(home_path, "wiki_dataset/validate")
test_path = os.path.join(home_path, "wiki_dataset/test/")

train_path_0 = os.path.join(train_path, "1-15")
train_path_1 = os.path.join(train_path, "16-30")
train_path_2 = os.path.join(train_path, "31-45")
train_path_3 = os.path.join(train_path, "46-60")
train_path_4 = os.path.join(train_path, "61-75")
train_path_5 = os.path.join(train_path, "76-90")
train_path_6 = os.path.join(train_path, "91-115")
validate_path_0 = os.path.join(validate_path, "1-15")
validate_path_1 = os.path.join(validate_path, "16-30")
validate_path_2 = os.path.join(validate_path, "31-45")
validate_path_3 = os.path.join(validate_path, "46-60")
validate_path_4 = os.path.join(validate_path, "61-75")
validate_path_5 = os.path.join(validate_path, "76-90")
validate_path_6 = os.path.join(validate_path, "91-115")
test_path_t = os.path.join(test_path, "test")
processed_paths = [source_path, train_path, validate_path, test_path, train_path_0, train_path_1, train_path_2,
                   train_path_3, train_path_4, train_path_5, train_path_6, validate_path_0, validate_path_1,
                   validate_path_2, validate_path_3, validate_path_4, validate_path_5, validate_path_6,
                   test_path_t]

# Load all of the labels from wiki.mat into a list of dictionaries
mat = loadmat(mat_path)['wiki'][0][0]
print(mat)
total_images = len(mat[0][0])
mat_data = [{"dob": mat[0][0][i],
             "date": mat[1][0][i],
             "filepath": mat[2][0][i][0],
             "gender": mat[3][0][i],
             "first_score": mat[6][0][i],
             "second_score": mat[7][0][i],
             "usable": True,
             "role": None,
             "destination": None,
             "age": (mat[1][0][i]-datetime.fromordinal(max(int(mat[0][0][i]) - 366, 1)).year)}
            for i in range(0, total_images)]
print("Loaded data from " + mat_path)

# Find all of the images with: only one face, a good score, a defined gender and a filepath
for i in range(0, len(mat_data)):
    if (mat_data[i]["second_score"] > 0) or (mat_data[i]["first_score"] < 1.5):
        mat_data[i]["usable"] = False
        continue
    if not (mat_data[i]["filepath"] and (mat_data[i]["gender"] == 0 or mat_data[i]["gender"] == 1)):
        mat_data[i]["usable"] = False
    if mat_data[i]["usable"]:
        if mat_data[i]["age"] <= 0 or mat_data[i]["age"] >= 116:
            mat_data[i]["usable"] = False

# Remove any unusable data and split remainder into training, validation and testing
mat_data_good = [image for image in mat_data if image["usable"]]
good_images = len(mat_data_good)
train_cutoff = 0.64
validate_cutoff = 0.8
train_images = int(math.floor(good_images * train_cutoff))
validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
test_images = good_images - (train_images + validate_images)

for i in range(0, train_images):
    mat_data_good[i]["role"] = "train"
for i in range(train_images, train_images + validate_images):
    mat_data_good[i]["role"] = "validate"
for i in range(train_images + validate_images, good_images):
    mat_data_good[i]["role"] = "test"

print("Total images: " + str(total_images))
print("Usable images: " + str(good_images))
print("Training images: " + str(train_images))
print("Validation images: " + str(validate_images))
print("Testing images: " + str(test_images))

# Create wiki_dataset directory, the 'train', 'validate' and 'test' subfolders
for path in processed_paths:
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created directory: " + path)

# Populate folders the folders with the images from the wiki_dataset and dump labels to text for later
current_point = 0
end_point = len(mat_data_good)
file = open(home_path+"wiki_dataset/labels.txt", "w")
test_labels_file = open(test_path + "test_labels.csv", "w")


for image in mat_data_good:
    source = os.path.join(source_path, image["filepath"])
    if image["age"] >= 1 and image["age"] <= 15:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                   "1-15" if image["role"] != "test" else "test")
    elif image["age"] >= 16 and image["age"] <= 30:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                       "16-30" if image["role"] != "test" else "test")
    elif image["age"] >= 31 and image["age"] <= 45:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                       "31-45" if image["role"] != "test" else "test")
    elif image["age"] >= 46 and image["age"] <= 60:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                   "46-60" if image["role"] != "test" else "test")
    elif image["age"] >= 61 and image["age"] <= 75:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                       "61-75" if image["role"] != "test" else "test")
    elif image["age"] >= 76 and image["age"] <= 90:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                       "76-90" if image["role"] != "test" else "test")
    elif image["age"] >= 91 and image["age"] <= 115:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                       "91-115" if image["role"] != "test" else "test")
    else:
        destination = os.path.join(home_path, "wiki_dataset", image["role"],
                                   str(int(image["age"])) if image["role"] != "test" else "test")
    if image["role"] == "test":
        test_labels_file.write(str(image) + '\n')
    shutil.copy(source, destination)
    file.write(str(image) + '\n')
    current_point += 1
    print("Progress: " + str(current_point) + "/" + str(end_point) + " (Copied " + source + " to " + destination + ")")
file.close()
test_labels_file.close()
