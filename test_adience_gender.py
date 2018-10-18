from predict_gender_age import predict_from_file, prepare_model
import argparse
import csv
import os

parser = argparse.ArgumentParser(
        description="Tests a CNN gender model trained with Adience against the images in the Adience Test folder.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
args = parser.parse_args()
model = prepare_model(args.model)

home_path = os.path.dirname(__file__)

# I had to remove [ ] ' and spaces from test_labels.csv before running this script
with open(home_path+"dataset_adience_gender/test/test_labels.csv",'r') as f, open(home_path+"dataset_adience_gender/test/test_labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    line_count = 0
    females = 0
    males = 0
    correct_guesses = 0
    corr_females = 0
    corr_males = 0

    for line in reader:
        newline = line
        actual_gender = newline[4]

        path_to_test = home_path+"/dataset_adience_gender/test/test/"
        original_image = newline[1]
        face_id = newline[2]
        image = path_to_test + "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model, image)

        if probability_vector[0] >= probability_vector[1]:
            predicted_gender = "f"
            females += 1
        else:
            predicted_gender = "m"
            males += 1

        newline.append(predicted_gender)

        if predicted_gender == actual_gender:
            accuracy = "Y"
            correct_guesses += 1
            if actual_gender == "f":
                corr_females += 1
            elif actual_gender == "m":
                corr_males += 1
        else:
            accuracy = "N"

        newline.append(accuracy)

        writer.writerow(newline)
        line_count += 1
        print("line_count: " + str(line_count))

    print("Number of test images: " + str(line_count))
    print("correct_guesses: " + str(correct_guesses))
    perc_corr_guesses = correct_guesses / line_count
    print("% correct guesses: " + perc_corr_guesses)
    perc_corr_females = corr_females / females
    print("% correct females/ total females: " + perc_corr_females)
    perc_corr_males = corr_males / males
    print("% correct females/ total males: " + perc_corr_males)

print("End of file.")
# To Do: Add histogram visualisation
