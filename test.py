# Pick one bucket size. You don't want to spend time coding a bunch of stuff we're not going to actually use this/next week in the demo.
# Find the mid-point of each bucket. For [1..15][16..30][31..45][46..60] that'd be 8, 23, 38, 53. Make it into a vector, so [8, 23, 38, 53], which we'll call the bucket vector, lol.
# Make the prediction on the picture. We'll call what you get the 'probability vector'.
# Compute the dot product of the bucket vector and the probability vector. There's probably a numpy function to do it. The result is the predicted age.
# Then for when you want to test it, compute the distance between the predicted age and the mid-point of the bucket of the correct answer."
# ...To get the answer for ONE prediction
# Do that for every picture in the test folder (using a loop) and append all of the differences to an array
# Then figure out what the average is of all of those differences (to find the bias) and the absolute value of all of those differences (to find the accuracy/precision)
# Differences as in, differences between the predicted age and the middle of the bucket
# If you're badly stuck, I can look at it tomorrow, but I'd appreciate it a lot if, in the meantime, you figure out a way to load the .csv file with the correct answers in it into python, since that's the annoying part for me.


from predict import predict_from_file, prepare_model
import argparse
import numpy as np
import csv


parser = argparse.ArgumentParser(
        description="Predicts age of person in image using a provided CNN model.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
''' parser.add_argument('--pic', default=None, required=True,
                        help="required; path to the image to be tested.") '''
args = parser.parse_args()
model = prepare_model(args.model)
# image = args.pic


with open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/labels_new.csv",'r') as f, open("/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/labels_predictions.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    bin_vector = [8, 23, 38, 53]

    for line in reader:
        newline = line
        actual_age = newline[3]

        path_to_test = "/Users/maggieliuzzi/agerecognition/dataset_adience_age_15y/test"
        original_image = newline[1]
        face_id = newline[2]
        image = "coarse_tilt_aligned_face." + face_id + "." + original_image

        probability_vector = predict_from_file(model, image)
        print(probability_vector)
        predicted_age = np.dot(bin_vector, probability_vector)
        newline[14] = predicted_age

        writer.writerow(newline)

f.close()
newf.close()
print("End of file.")

# To Do: Add histogram visualisation


# bin_vector = [2.5, 6, 10, 14.5, 19, 24.5, 28, 32, 41.5, 48.5]
# age: 28 /Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test/coarse_tilt_aligned_face.828.11682572164_053efb176f_o.jpg
# age: 17 /Users/maggieliuzzi/agerecognition/dataset_adience_age_10_ED_bins/test/test/coarse_tilt_aligned_face.512.11081151624_dbfb2edaaf_o.jpg
'''
probability_vector = predict_from_file(model, image)
print(probability_vector)
predicted_age = np.dot(bin_vector, probability_vector)
# inaccuracy = predicted_age - # bin_vector[i] of the right answer
print(predicted_age)
'''
