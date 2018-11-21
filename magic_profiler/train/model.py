import numpy as np
import os

from magic_profiler.utils import warning, error, timer


# Makes numpy's RNG constant for reproducable results
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


class Model:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.base_model = None
        self.new_layers = None

    def say(self, m):
        '''Prints a message if self.verbose is > 0'''
        if self.verbose>0:
            print(m)

    def load(self, model_uri="MobileNetV2", input_shape=(224, 224, 3), trainable_layers=0, pooling='avg'):
        '''Loads a new keras model into memory, downloading first if necessary'''
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            if model_uri.lower() == "mobilenetv2":
                from keras.applications.mobilenetv2 import MobileNetV2
                self.base_model = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False, pooling=pooling)
            elif os.path.isfile(model_uri):
                self.base_model = models.load_model(model_uri)
            for layer in self.base_model.layers[:-trainable_layers]:
                layer.trainable = False
        except Exception as e:
            warning("Keras base model could not be loaded", e)
            return False
        else:
            if self.verbose>0:
                self.say("BASE MODEL ...")
                self.base_model.summary()
            self.say("Keras base model loaded successfully.")
            return True
    
    def create(self, nodes):
        '''Creates the new top layer for the Keras model'''
        try:
            top_layer = models.Sequential()
            top_layer.add(layers.Dense(nodes, activation='softmax', input_dim=self.base_model.layers[-1].output_shape[1]))
            self.top_layer = top_layer
        except Exception as e:
            warning("Could not create top layer of the Keras model", e)
            return False
        else:
            if self.verbose>0:
                self.say("TOP LAYERS ...")
                self.top_layer.summary()
            self.say("Successfully created the top layer of the Keras model.")
            return True
    
    def save(self, filepath):
        '''Saves the keras model to disk'''
        try:
            final_model = models.Sequential()
            final_model.add(self.base_model)
            final_model.add(self.top_layer)
            final_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
            final_model.summary()
            final_model.save(filepath)
        except Exception as e:
            warning("Could not save Keras model", e)
            return False
        else:
            self.say("Successfully saved Keras model.")
            return True

def main():
    mm = Model(verbose=0)
    mm.load("MobileNetV2", trainable_layers=4)
    mm.create(4)
    mm.save("test.h5")

timer(main)
