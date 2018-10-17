# The purpose of this file is to define functions needed to make predictions with the trained models
# If the file is run directly, it can make predictions against one image
# If the file is imported, it can load models and make predictions via functions

from PIL import Image
import numpy as np
import argparse
import io
import os

# Ensures that Keras is using Theano as a back-end, then loads Keras
keras_path = os.path.join(os.path.expanduser('~'), '.keras')
keras_json_path = os.path.join(keras_path, 'keras.json')
if not os.path.isdir(keras_path):
    os.makedirs(keras_path)
with open(keras_json_path, 'w') as kf:
    contents = "{\"epsilon\": 1e-07,\"image_data_format\": \"channels_last\",\"backend\": \"theano\",\"floatx\": \"float32\"}"
    kf.write(contents)

from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model

def prepare_model(model_path):
    # Loads the keras model specified by model_path and returns the loaded model
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model._make_predict_function()
    return model

def predict_from_file(loaded_model, pic_path):
    # Makes prediction against loaded_model for an image specified by pic_path and returns the probability vector
    # Use prepare_model first to create the loaded model
    img = image.load_img(pic_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predict = loaded_model.predict(img)
    return predict[0]

def predict_from_pil(loaded_model, pil_file):
    # Makes prediction against loaded_model for an image provided as a PIL object and returns the probability vector
    # Use prepare_model first to create the loaded model
    img = Image.open(io.BytesIO(pil_file))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predict = loaded_model.predict(img)
    return predict[0]

if __name__ == "__main__":
    # If predict_gender_age.py is run directly, it makes a prediction for an image from a file
    parser = argparse.ArgumentParser(
        description="Predicts if an image of a face is male or female using a provided CNN model.",
        epilog="Created by Maria Liuzzi & Mitchell Clarke")
    parser.add_argument('--model_gender', default=None, required=True,
                        help="required; path to the neural network model file.")
    parser.add_argument('--model_age', default=None, required=True,
                        help="required; path to the neural network model file.")
    parser.add_argument('--pic', default=None, required=True,
                        help="required; path to the image to be tested.")
    args = parser.parse_args()

    if not os.path.isfile(args.model_age):
        print("ERROR: Could not find the age-recognition model file.")
        exit(1)
    if not os.path.isfile(args.model_gender):
        print("ERROR: Could not find the gender-recognition model file.")
        exit(1)
    if not os.path.isfile(args.pic):
        print("ERROR: Could not find the image to be classified.")
        exit(1)

    model_gender = prepare_model(args.model_gender)
    raw_prediction_gender = predict_from_file(model_gender, args.pic)
    model_age = prepare_model(args.model_age)
    raw_prediction_age = predict_from_file(model_age, args.pic)

    # print("\nProbabilities (Gender): " + str(raw_prediction_gender))
    # print("\nProbabilities (Age): " + str(raw_prediction_age))
