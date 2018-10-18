# The purpose of this file is to start a server which can receive images via HTTP POST and return predictions
# Based on: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from predict_gender_age import prepare_model, predict_from_pil
import numpy as np
import argparse
import flask
import os

app = flask.Flask(__name__)
model_age = None
model_gender = None

'''
Receive images via POST at /predict and respond with JSON containing the prediction vector
'''
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request.")

    # Ensures there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        # age
        midpoints = [8, 23, 38, 53]
        raw_prediction_age = predict_from_pil(model_age, image)
        calculated_age = np.dot(midpoints, raw_prediction_age)
        print("Probability of [1-15, 16-30, 31-45, 46-60]: " + str(raw_prediction_age))
        data["prediction_age"] = {"1-15": float(raw_prediction_age[0]), "16-30": float(raw_prediction_age[1]), "31-45": float(raw_prediction_age[2]),
                              "46-60": float(raw_prediction_age[3]), "Estimate": calculated_age}
        # gender
        raw_prediction_gender = predict_from_pil(model_gender, image)
        calculated_gender = "Female" if raw_prediction_gender[1] <= raw_prediction_gender[0] else "Male"
        print("Probability of [Female, Male]: " + str(raw_prediction_gender))
        data["prediction_gender"] = {"Female": float(raw_prediction_gender[0]), "Male": float(raw_prediction_gender[1]), "Estimate": calculated_gender}

        data["success"] = True
    else:
        print("Image attribute not found in POST request.")

    return flask.jsonify(data)


'''
When server.py is ran directly, prepare the model + server to take requests
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts server to predicts if an image of a face is male or female using a CNN model.",
        epilog="Created by Maria Liuzzi & Mitchell Clarke")
    parser.add_argument('--model_age', default=None, required=True,
                        help="required; path to the neural network model_age file.")
    parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model_gender file.")
    parser.add_argument('--port', default=4000, help="the port the server occupies; defaults to 4000.")
    args = parser.parse_args()

    if not os.path.isfile(args.model_age):
        print("ERROR: Could not find the neural network model_age file.")
        exit(1)
    if not os.path.isfile(args.model_gender):
        print("ERROR: Could not find the neural network model_gender file.")
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
    print("\nLoading Flask server...")
    app.run(host="0.0.0.0", port='4000')