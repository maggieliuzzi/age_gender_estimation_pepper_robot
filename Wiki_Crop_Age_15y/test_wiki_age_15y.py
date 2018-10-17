from predict_gender_age import predict_from_file, prepare_model
import argparse
import numpy as np
import csv

# --model_gender /Users/maggieliuzzi/Comparable_Models/Gender/Wiki-Crop/model.03.hdf5
#  --model_age /Users/maggieliuzzi/Comparable_Models/Gender/Wiki-Crop/model_wiki_age_15y_theano.h5

parser = argparse.ArgumentParser(
        description="Tests a CNN model passed as an argument against the images in the Test folder.")
parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model file.")
parser.add_argument('--model_age', default=None, required=True,
                        help="required; path to the neural network model file.")
args = parser.parse_args()
model_gender = prepare_model(args.model_gender)
model_age = prepare_model(args.model_age)

with open("/Users/maggieliuzzi/agerecognition/wiki_dataset/test/form_test_labels.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/wiki_dataset/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    bin_vector_age = [8, 23, 38, 53, 68, 83, 103]

    line_count = 0

    correct_guesses = 0
    wrong_guesses = 0

    agg_age_bias = 0
    agg_abs_age_bias = 0

    for line in reader:
        newline = line

        path_to_test = "/Users/maggieliuzzi/agerecognition/wiki_dataset/test/test/"
        image_name = newline[4]
        image = path_to_test + image_name

        actual_gender = newline[6]
        actual_age = newline[7]

        probability_vector_gender = predict_from_file(model_gender, image)
        predicted_gender = "Female" if probability_vector_gender[0] >= probability_vector_gender[1] else "Male"
        newline.append(predicted_gender)

        probability_vector_age = predict_from_file(model_age, image)
        print(probability_vector_age)
        predicted_age = np.dot(bin_vector_age, probability_vector_age)
        predicted_age = int(predicted_age)
        print(int(round(predicted_age,0)))
        newline.append(predicted_age)

        if predicted_gender == actual_gender:
            accuracy = "Y"
            correct_guesses += 1
        else:
            accuracy = "N"
            wrong_guesses += 1
        newline.append(accuracy)

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
    print("correct_guesses: " + str(correct_guesses))
    perc_correct_guesses = correct_guesses / line_count
    print("% correct_guesses: " + perc_correct_guesses)
    print("wrong_guesses: " + str(wrong_guesses))
    perc_wrong_guesses = wrong_guesses / line_count
    print("% wrong_guesses: " + perc_wrong_guesses)

    print("Number of test images: " + str(line_count))
    print("Average age_bias: " + str(agg_age_bias / line_count))
    print("Average abs_age_bias: " + str(agg_abs_age_bias / line_count))

f.close()
newf.close()
print("End of file.")

# To Do: Add histogram visualisation
