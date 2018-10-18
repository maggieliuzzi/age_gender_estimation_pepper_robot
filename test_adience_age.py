from predict_gender_age import predict_from_file, prepare_model
import argparse
import numpy as np
import csv
import os

parser = argparse.ArgumentParser(
        description="Tests a CNN model passed as an argument against the images in the Test folder.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
parser.add_argument('--bucketsize', default=None, required=True,
                        help="the age range of a single bucket in the prediction; required; must be: 15, 10 or 5.")
args = parser.parse_args()
model = prepare_model(args.model)
bucket_size = int(args.bucketsize)

home_path = os.path.dirname(__file__)

# Had to remove [ ] ' spaces and rows aged -60100 or 0 manually before running script
with open(home_path+"dataset_adience_age_5y/test/test_labels.csv",'r') as f, open(home_path+"dataset_adience_age_5y/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    bin_vector = []
    bin_vector.extend(
        [(1 + bucket_size) / 2, (bucket_size + bucket_size * 2) / 2, (bucket_size * 2 + bucket_size * 3) / 2,
         (bucket_size * 3 + bucket_size * 4) / 2])
    if bucket_size < 15:
        bin_vector.extend([(bucket_size * 4 + bucket_size * 5) / 2, (bucket_size * 5 + bucket_size * 6) / 2])
    if bucket_size < 10:
        bin_vector.extend([(bucket_size * 6 + bucket_size * 7) / 2, (bucket_size * 7 + bucket_size * 8) / 2,
                          (bucket_size * 8 + bucket_size * 9) / 2,
                          (bucket_size * 9 + bucket_size * 10) / 2, (bucket_size * 10 + bucket_size * 11) / 2,
                          (bucket_size * 11 + bucket_size * 12) / 2])

    line_count = 0
    agg_age_bias = 0
    agg_abs_age_bias = 0

    for line in reader:
        newline = line

        actual_age = newline[3]

        path_to_test = home_path+"dataset_adience_age_5y/test/test/"
        original_image = newline[1]
        face_id = newline[2]
        # Had to remove the 's, spaces and [ ]s around values in the .csv file manually
        image = path_to_test + "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model, image)
        predicted_age = np.dot(bin_vector, probability_vector)
        predicted_age = int(round(predicted_age,0))
        newline.append(predicted_age)
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

    print("Number of test images: " + str(line_count))
    print("Average age_bias: " + str(agg_age_bias / line_count))
    print("Average abs_age_bias: " + str(agg_abs_age_bias / line_count))

print("End of file.")
# To Do: Add histogram visualisation
