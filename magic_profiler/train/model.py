import numpy as np
import os

from magic_profiler.utils import warning, error, timer

# Makes numpy's RNG constant for reproducable results
np.random.seed(1)

# Ensures that Keras is using Theano as a back-end, then loads Keras
keras_path = os.path.join(os.path.expanduser('~'), '.keras')
if not os.path.isdir(keras_path):
    os.makedirs(keras_path)
with open(os.path.join(keras_path, 'keras.json'), 'w') as f:
    contents = "{\"epsilon\": 1e-07,\"image_data_format\": \"channels_last\",\"backend\": \"theano\",\"floatx\": \"float32\"}"
    f.write(contents)

# Import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator


class Model:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.model = None

    def say(self, m):
        '''Prints a message if self.verbose is > 0'''
        if self.verbose>0:
            print(m)
    
    def summary(self):
        if self.verbose>0:
            self.say("MODEL STRUCTURE ...")
            self.model.summary()

    def load(self, model_uri="MobileNetV2", input_shape=(224, 224, 3), trainable_layers=0, pooling='avg'):
        '''Loads a new keras model into memory, downloading first if necessary'''
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            if model_uri.lower() == "mobilenetv2":
                from keras.applications.mobilenetv2 import MobileNetV2
                self.model = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False, pooling=pooling)
            elif os.path.isfile(model_uri):
                self.model = models.load_model(model_uri)
            for layer in self.model.layers[:-trainable_layers]:
                layer.trainable = False
            self.summary()
        except Exception as e:
            warning("Keras base model could not be loaded", e)
            return False
        else:
            self.say("Keras base model loaded successfully.")
            return True
    
    def add(self, nodes):
        '''Creates the new top layer for the Keras model'''
        try:
            new_model = models.Sequential()
            new_model.add(self.model)
            new_model.add(layers.Dense(nodes, activation='softmax'))
            self.model = new_model
            self.summary()
        except Exception as e:
            warning("Could not create top layer of the Keras model", e)
            return False
        else:
            self.say("Successfully created the top layer of the Keras model.")
            return True

    def save(self, filepath):
        '''Saves the keras model to disk'''
        try:
            self.model.save(filepath)
        except Exception as e:
            warning("Could not save Keras model", e)
            return False
        else:
            self.say("Successfully saved Keras model.")
            return True
    
    def train(self, dataset_directory, batch_size, val_batch_size, epochs=1):
        input_shape = self.model.layers[0].input_shape[1:-1]
        train_trans = ImageDataGenerator(rescale=1./255, rotation_range=36, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        validate_trans = ImageDataGenerator(rescale=1./255)
        train_generator = train_trans.flow_from_directory(os.path.join(dataset_directory, "train"), target_size=input_shape, batch_size=batch_size, seed=1, class_mode='categorical')
        validate_generator = validate_trans.flow_from_directory(os.path.join(dataset_directory, "validate"), target_size=input_shape, batch_size=val_batch_size, seed=1, class_mode='categorical', shuffle=False)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=len(train_generator), verbose=1, validation_data=validate_generator, validation_steps=len(validate_generator))
