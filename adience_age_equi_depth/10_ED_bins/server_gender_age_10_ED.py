# Starts a server which can receive images via HTTP POST and return predictions
# Based on: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from predict_gender_age import prepare_model, predict_from_pil
import argparse
import flask
import os

app = flask.Flask(__name__)
model_age = None
model_gender = None

# Eg. model_gender: /Users/maggieliuzzi/agerecognition/Comparable_Models/Gender/final_model_adience_gender_8e_Theano.h5
# Eg. model_age_10_ED: /Users/maggieliuzzi/agerecognition/Comparable_Models/Age/10_ED_Bins/final_model_adience_age_10_ED_bins_10e_Theano_2.h5

# Receive images via POST at /predict and respond with JSON containing the prediction vector
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request.")

    # Ensure there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()

        raw_prediction_age = predict_from_pil(model_age, image)
        print("Probability of [1-4, 5-7, 8-12, 13-16, 17-21, 22-27, 28, 29-35, 36-46, 47-60]: " + str(raw_prediction_age))
        data["prediction_age"] = {"1-4": float(raw_prediction_age[0]), "5-7": float(raw_prediction_age[1]), "8-12": float(raw_prediction_age[2]),
                              "13-16": float(raw_prediction_age[3]), "17-21": float(raw_prediction_age[4]), "22-27": float(raw_prediction_age[5]),
                              "28": float(raw_prediction_age[6]), "29-35": float(raw_prediction_age[7]), "36-46": float(raw_prediction_age[8]),
                              "47-60": float(raw_prediction_age[9])}

        raw_prediction_gender = predict_from_pil(model_gender, image)
        print("Probability of [Female, Male]: " + str(raw_prediction_gender))
        data["prediction_gender"] = {"Female": float(raw_prediction_gender[0]), "Male": float(raw_prediction_gender[1])}

        data["success"] = True
    else:
        print("Image attribute not found in POST request.")

    return flask.jsonify(data)

# When this script is ran directly, prepare the model + server to take requests
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts server to predict if a person in an image is male or female using a CNN model.",
        epilog="Created by Maria Liuzzi & Mitchell Clarke")
    parser.add_argument('--model_age_10_ED', default=None, required=True,
                        help="required; path to the neural network model_age file.")
    parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model_gender file.")
    parser.add_argument('--port', default=4000, help="the port the server occupies; defaults to 4000.")
    args = parser.parse_args()

    if not os.path.isfile(args.model_age):
        print("ERROR: Could not find model_age file.")
        exit(1)
    if not os.path.isfile(args.model_gender):
        print("ERROR: Could not find model_gender file.")
        exit(1)
    try:
        n = int(args.port)
        if n < 1:
            print("ERROR: Port number must be greater than or equal to 1.")
            exit(1)
    except ValueError:
        print("ERROR: Port number must be an integer.")
        exit(1)

    print("\nLoading Keras model_age and model_gender...")
    model_age = prepare_model(args.model_age)
    model_gender = prepare_model(args.model_gender)
    print("\nLoading Flask server.")
    app.run(host="0.0.0.0", port='4000')
