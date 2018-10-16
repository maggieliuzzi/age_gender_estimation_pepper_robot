from predict_gender_age import predict_from_file, prepare_model
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(
        description="Tests a CNN model passed as an argument against the images in the Test folder.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
args = parser.parse_args()
model = prepare_model(args.model)

with open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/test/test_labels.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    bin_vector = [8, 23, 38, 53] # bin_vector = [2.5, 6, 10, 14.5, 19, 24.5, 28, 32, 41.5, 48.5]

    for line in reader:
        newline = line
        actual_age = newline[3]

        path_to_test = "/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/test/test/"
        original_image = newline[1]
        face_id = newline[2]
        # Had to remove the 's, spaces and [ ]s around values in the .csv file manually
        image = path_to_test + "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model, image)
        print(probability_vector)
        predicted_age = np.dot(bin_vector, probability_vector)
        newline[14] = predicted_age
        # inaccuracy = predicted_age - # bin_vector[i] of the right answer
        # print(predicted_age)

        writer.writerow(newline)
f.close()
newf.close()
print("End of file.")

# To Do: Add histogram visualisation

# age: 28 /Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test/coarse_tilt_aligned_face.828.11682572164_053efb176f_o.jpg
# age: 17 /Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test/coarse_tilt_aligned_face.512.11081151624_dbfb2edaaf_o.jpg
