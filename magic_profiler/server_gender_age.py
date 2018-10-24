# Starts a server which can receive images via HTTP POST and return predictions
# Based on: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from predict_gender_age import prepare_model, predict_from_pil
import flask
import os
import argparse
import numpy as np

app = flask.Flask(__name__)
model_age = None
model_gender = None

# Receive images via POST at /predict and respond with JSON containing the prediction vector
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request.")

    # Ensure there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()

        raw_prediction_age = predict_from_pil(model_age, image)

        midpoints = []
        midpoints.extend([(1+bucket_size)/2, (bucket_size+bucket_size*2)/2, (bucket_size*2+bucket_size*3)/2, (bucket_size*3+bucket_size*4)/2])
        if bucket_size < 15:
            midpoints.extend([(bucket_size*4+bucket_size*5)/2, (bucket_size*5 + bucket_size*6)/2])
        if bucket_size < 10:
            midpoints.extend([(bucket_size*6+bucket_size*7)/2, (bucket_size*7 + bucket_size*8)/2, (bucket_size*8+bucket_size*9)/2,
                              (bucket_size*9+bucket_size*10)/2, (bucket_size*10+bucket_size*11)/2, (bucket_size*11 + bucket_size*12)/2])

        calculated_age = np.dot(midpoints, raw_prediction_age)

        print("Probability of 1-"+str(bucket_size)+", "+str(bucket_size+1)+"-"+str(bucket_size*2)+", "+str(bucket_size*2+1)+"-"+
              str(bucket_size*3)+", "+str(bucket_size*3+1)+"-"+str(bucket_size*4))
        if bucket_size < 15:
            print(str(bucket_size*4+1)+"-"+str(bucket_size*5)+", "+str(bucket_size*5+1)+"-"+str(bucket_size*6)+":")
        if bucket_size < 10:
            print(str(bucket_size*6+1)+"-"+str(bucket_size*7)+", "+str(bucket_size*7+1)+"-"+str(bucket_size*8)+", "+
                  str(bucket_size*8+1)+"-"+str(bucket_size*9)+", "+str(bucket_size*9+1)+"-"+str(bucket_size*10)+", "+
                  str(bucket_size*10+1)+"-"+str(bucket_size*11)+", "+str(bucket_size*11+1)+"-"+str(bucket_size*12)+":")
        print(raw_prediction_age)

        data["prediction_age"] = {"1-"+str(bucket_size): float(raw_prediction_age[0]), str(bucket_size+1)+"-"+str(bucket_size*2): float(raw_prediction_age[1]),
                                  str(bucket_size*2+1)+"-"+str(bucket_size*3): float(raw_prediction_age[2]), str(bucket_size*3+1)+"-"+str(bucket_size*4): float(raw_prediction_age[3])}
        if bucket_size < 15:
            data["prediction_age"].update({str(bucket_size*4+1)+"-"+str(bucket_size*5): float(raw_prediction_age[4]),
                                           str(bucket_size*5+1)+"-"+str(bucket_size*6): float(raw_prediction_age[5])})
        if bucket_size < 10:
            data["prediction_age"].update({str(bucket_size*6+1)+"-"+str(bucket_size*7): float(raw_prediction_age[6]),
                                           str(bucket_size*7+1)+"-"+str(bucket_size*8): float(raw_prediction_age[7]),
                                           str(bucket_size*8+1)+"-"+str(bucket_size*9): float(raw_prediction_age[8]),
                                           str(bucket_size*9+1)+"-"+str(bucket_size*10): float(raw_prediction_age[9]),
                                           str(bucket_size*10+1)+"-"+str(bucket_size*11): float(raw_prediction_age[10]),
                                           str(bucket_size*11+1)+"-"+str(bucket_size*12): float(raw_prediction_age[11])})
        data["prediction_age"].update({"Guessed Age": calculated_age})


        raw_prediction_gender = predict_from_pil(model_gender, image)
        predicted_gender = "Female" if raw_prediction_gender[0] >= raw_prediction_gender[1] else "Male"

        print("Probability of [Female, Male]: " + str(raw_prediction_gender))
        data["prediction_gender"] = {"Female": float(raw_prediction_gender[0]), "Male": float(raw_prediction_gender[1]), "Guessed Gender": predicted_gender}

        data["success"] = True
    else:
        print("Image attribute not found in POST request.")

    return flask.jsonify(data)

# When this script is ran directly, prepare the model + server to take requests
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts server to predict a person's age and gender using a CNN model.",
        epilog="Created by Maria Liuzzi & Mitchell Clarke")
    parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model_gender file.")
    parser.add_argument('--model_age', default=None, required=True,
                        help="required; path to the neural network model_age file.")
    parser.add_argument('--bucketsize', default=None, required=True,
                        help="the age range of a single bucket in the prediction; required; must be: 15, 10 or 5.")
    parser.add_argument('--port', default=4000, help="the port the server occupies; defaults to 4000.")
    args = parser.parse_args()
    bucket_size = int(args.bucketsize)

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
    print("\nLoading Flask server...")
    app.run(host="0.0.0.0", port='4000')
