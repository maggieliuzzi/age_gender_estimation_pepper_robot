from predict_gender_age import prepare_model, predict_from_file
import argparse
import os
import yaml
import csv

# Parse and check arguments received at the command line
parser = argparse.ArgumentParser(
    description="Test the quality of a CNN model passed as argument using test data in the dataset folder.",
    epilog="Created by Maria Liuzzi & Mitchell Clarke")
parser.add_argument('--model', default=None, required=True,
                    help="Path to the neural network model file.")
args = parser.parse_args()
if not os.path.isfile(args.model):
    print("ERROR: Could not find the neural network model file.")
    exit(1)

# Define useful filepaths for later
home_dir = os.path.dirname(__file__)
train_dir = os.path.join(home_dir, 'dataset_adience_age_10_ED_bins/test/test')
label_filename = os.path.join(home_dir, 'dataset_adience_age_10_ED_bins/labels.csv') # needs .txt?

# Load label.txt for checking answers later
mat_test_data = {}

print("Reading labels.csv ...")
with open(label_filename, 'r') as file:
    for line in file:
        record = yaml.load(line)
        key = os.path.basename(record['filepath'])
        mat_test_data[key] = record
print("Labels loaded.")

# Test and report on network quality
print("Loading model ...")
model = prepare_model(args.model)
print("Model loaded.")

print("Running test ...")
total = 0
'''
female_pred = 0
male_pred = 0
female_actual = 0
male_actual = 0
male_true = 0
female_true = 0
male_false = 0
female_false = 0
'''
pred_1_4 = 0
pred_5_7 = 0
pred_8_12 = 0
pred_13_16 = 0
pred_17_21 = 0
pred_22_27 = 0
pred_28 = 0
pred_29_35 = 0
pred_36_46 = 0
pred_47_60 = 0
actual_1_4 = 0
actual_5_7 = 0
actual_8_12 = 0
actual_13_16 = 0
actual_17_21 = 0
actual_22_27 = 0
actual_28 = 0
actual_29_35 = 0
actual_36_46 = 0
actual_47_60 = 0
true_1_4 = 0
true_5_7 = 0
true_8_12 = 0
true_13_16 = 0
true_17_21 = 0
true_22_27 = 0
true_28 = 0
true_29_35 = 0
true_36_46 = 0
true_47_60 = 0
false_1_4 = 0
false_5_7 = 0
false_8_12 = 0
false_13_16 = 0
false_17_21 = 0
false_22_27 = 0
false_28 = 0
false_29_35 = 0
false_36_46 = 0
false_47_60 = 0

predictions = []

test_image_dir = os.listdir(train_dir)
end_point = len(test_image_dir)
with progress.Bar(expected_size=end_point) as bar:
    for test_image in test_image_dir:
        img_path = os.path.join(train_dir, test_image)
        raw_prediction = predict_from_file(model, img_path)
        actual_prediction = 0 if raw_prediction[0] >= raw_prediction[1] else 1
        actual_answer = mat_test_data[test_image][13]  # mat_test_data[test_image]['gender']
        predictions.append({
            'filepath':mat_test_data[test_image]['filepath'],
            'bin':mat_test_data[test_image][13],
            'test_score':abs(raw_prediction[0]-raw_prediction[1])
            })
        total += 1
        if actual_prediction == 0:
            female_pred += 1
            if actual_answer == 0.0:
                female_actual += 1
                female_true += 1
            else:
                male_actual += 1
                female_false += 1
                predictions[-1]['test_score'] *= -1
        else:
            male_pred += 1
            if actual_answer == 0.0:
                female_actual += 1
                male_false += 1
                predictions[-1]['test_score'] *= -1
            else:
                male_actual += 1
                male_true += 1
        bar.show(total)

print("Total Images: " + str(total))
print("Predicted/Actual Females: " + str(female_pred) + "/" + str(female_actual) + " (" + str(
    (float(female_pred) / float(female_actual)) * 100) + "% of actual)")
print("Predicted/Actual Males: " + str(male_pred) + "/" + str(male_actual) + " (" + str(
    (float(male_pred) / float(male_actual)) * 100) + "% of actual)")
print("True Females vs. False Males: " + str(female_true) + "/" + str(male_false) + " (" + str(
    (float(female_true) / float(female_true+male_false)) * 100) + "% correct)")
print("True Males vs. False Females: " + str(male_true) + "/" + str(female_false) + " (" + str(
    (float(male_true) / float(male_true+female_false)) * 100) + "% correct)")

print("\nCreating csv file containing results...")
with open(args.model + '.csv', 'w') as csvfile:
    fieldnames = ['filepath', 'bin', 'test_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for prediction in predictions:
        writer.writerow(prediction)
