from predict_gender_age import predict_from_file, prepare_model
import argparse
import numpy as np
import csv

# --model /Users/maggieliuzzi/agerecognition_13-10-2018/Comparable_Models/Age/10_ED_Bins/final_model_adience_age_10_ED_bins_10e_Theano_2.h5

parser = argparse.ArgumentParser(
        description="Tests a CNN model passed as an argument against the images in the Test folder.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
args = parser.parse_args()
model = prepare_model(args.model)

with open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test_labels.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    bin_vector = [2.5, 6, 10, 14.5, 19, 24.5, 28, 32, 41.5, 48.5]

    line_count = 0
    agg_age_bias = 0
    agg_abs_age_bias = 0

    for line in reader:
        newline = line
        actual_age = newline[3]

        path_to_test = "/Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test/"
        original_image = newline[1]
        face_id = newline[2]
        # Had to remove the 's, spaces and [ ]s around values in the .csv file manually
        image = path_to_test + "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model, image)
        print(probability_vector)
        predicted_age = np.dot(bin_vector, probability_vector)
        predicted_age = int(predicted_age)
        print(int(round(predicted_age,0)))
        newline.append(predicted_age) # will append it in newline[13]
        # newline[13] = predicted_age
        # inaccuracy = predicted_age - # bin_vector[i] of the right answer
        actual_age = int(actual_age)
        age_bias = predicted_age - actual_age
        abs_age_bias = abs(predicted_age - actual_age)
        newline.append(age_bias)
        newline.append(abs_age_bias)

        writer.writerow(newline)
        line_count += 1
        print("line_count: " + str(line_count))
        agg_age_bias += age_bias
        agg_abs_age_bias += age_bias

        # Add test for gender under Adience_Gender

    print("Number of test images: " + str(line_count))
    print("Average age_bias: " + str(agg_age_bias / line_count))
    print("Average abs_age_bias: " + str(agg_abs_age_bias / line_count))
f.close()
newf.close()
print("End of file.")

# To Do: Add histogram visualisation
