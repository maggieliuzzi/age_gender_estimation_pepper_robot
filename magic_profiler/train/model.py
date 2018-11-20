from magic_profiler.utils import warning, error, timer
import os


# Makes numpy's RNG constant for reproducable results
import numpy as np
np.random.seed(2)

# Ensures that Keras is using Theano as a back-end, then loads Keras
keras_path = os.path.join(os.path.expanduser('~'), '.keras')
if not os.path.isdir(keras_path):
    os.makedirs(keras_path)
with open(os.path.join(keras_path, 'keras.json'), 'w') as f:
    contents = "{\"epsilon\": 1e-07,\"image_data_format\": \"channels_last\",\"backend\": \"theano\",\"floatx\": \"float32\"}"
    f.write(contents)

# Import keras
from keras import models, layers
from keras.models import Sequential


class Model:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.base_model = None
        self.new_model = None

    def say(self, m):
        '''Prints a message if self.verbose is > 0'''
        if self.verbose>0:
            print(m)

    def load_base(self, model_uri="MobileNetV2"):
        '''Loads a new keras model into memory, downloading first if necessary'''
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            if model_uri.lower() == "mobilenetv2":
                from keras.applications.mobilenetv2 import MobileNetV2
                self.base_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
        except Exception as e:
            warning("Keras model could not be loaded", e)
            return False
        else:
            self.say("Keras model loaded successfully.")
            return True

def main():
    mm = Model()
    mm.load_base("MobileNetV2")
    print(len(mm.base_model.layers))

timer(main)
