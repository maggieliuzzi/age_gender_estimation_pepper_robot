from predict_gender_age import predict_from_file, prepare_model
import argparse
import csv

# --model_gender /Users/maggieliuzzi/Comparable_Models/Gender/Adience/final_model_adience_gender_8e_Theano.h5

parser = argparse.ArgumentParser(
        description="Tests a CNN model passed as an argument against the images in the Test folder.")
parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model file.")
args = parser.parse_args()
model_gender = prepare_model(args.model_gender)

# I had to remove [ ] ' and spaces from test_labels.csv before running this script
with open("/Users/maggieliuzzi/agerecognition/dataset_adience_gender/test/test_labels.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/dataset_adience_gender/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    line_count = 0
    correct_guesses = 0
    wrong_guesses = 0

    for line in reader:
        newline = line
        f.close()

        actual_gender = newline[4]

        path_to_test = "/Users/maggieliuzzi/agerecognition/dataset_adience_gender/test/test/"
        original_image = newline[1]
        face_id = newline[2]
        # Had to remove the 's, spaces and [ ]s around values in the .csv file manually
        image = path_to_test + "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model_gender, image)
        predicted_gender = "Female" if probability_vector[0] >= probability_vector[1] else "Male"
        newline.append(predicted_gender)

        if predicted_gender == actual_gender:
            accuracy = "Y"
            correct_guesses += 1
        else:
            accuracy = "N"
            wrong_guesses += 1
        newline.append(accuracy)

        writer.writerow(newline)
        line_count += 1
        print("line_count: " + str(line_count))

    newf.close()

    print("Number of test images: " + str(line_count))
    print("correct_guesses: " + str(correct_guesses))
    perc_correct_guesses = correct_guesses / line_count
    print("% correct_guesses: " + perc_correct_guesses)
    print("wrong_guesses: " + str(wrong_guesses))
    perc_wrong_guesses = wrong_guesses / line_count
    print("% wrong_guesses: " + perc_wrong_guesses)

print("End of file.")
# To Do: Add histogram visualisation
