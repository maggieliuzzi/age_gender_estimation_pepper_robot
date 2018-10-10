# The purpose of this file is to start a server which can receive images via HTTP POST and return predictions
# Based on: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from predict import prepare_model, predict_from_pil
import argparse
import flask
import os

app = flask.Flask(__name__)
model_age = None
model_gender = None

# gender model: /Users/maggieliuzzi/agerecognition/Comparable_Models/Gender/final_model_adience_gender_8e_Theano.h5
# age 15y: /Users/maggieliuzzi/agerecognition/Comparable_Models/Age/15y/final_model_adience_age_15y_2e.h5

'''
Receive images via POST at /predict and respond with JSON containing the prediction vector
'''
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request...")

    # Ensures there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        # age
        raw_prediction_age = predict_from_pil(model_age, image)
        print("Probability of [1-15, 16-30, 31-45, 46-60]: " + str(raw_prediction_age))
        data["prediction_age"] = {"1-15": float(raw_prediction_age[0]), "16-30": float(raw_prediction_age[1]), "31-45": float(raw_prediction_age[2]),
                              "46-60": float(raw_prediction_age[3])}
        # gender
        raw_prediction_gender = predict_from_pil(model_gender, image)
        print("Probability of [Female, Male]: " + str(raw_prediction_gender))
        data["prediction_gender"] = {"Female": float(raw_prediction_gender[0]), "Male": float(raw_prediction_gender[1])}

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