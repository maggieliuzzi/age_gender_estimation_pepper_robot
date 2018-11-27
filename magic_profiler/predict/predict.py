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

class Predictor:
    def __init__(self):
        self.loaded_model = None
        
    def prepare(self, model_path):
        model = load_model(model_path)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model._make_predict_function()
        self.loaded_model = model

    def from_file(self, pic_path):
        img = image.load_img(pic_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        predict = self.loaded_model.predict(img)
        return predict[0]

    def from_pil(self, pil_file):
        img = Image.open(io.BytesIO(pil_file))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        predict = self.loaded_model.predict(img)
        return predict[0]
