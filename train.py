# The purpose of this file is to load and train the neural network
# This is done by repurposing a MobileNetV2 model trained on ImageNet

from keras.applications.mobilenetv2 import MobileNetV2
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import ssl

'''
+-----------------------+
| MobileNetV2 structure |
+-----------------------+

Input                   (224, 224, 3)
ZeroPadding2D           (225, 225, 3)
<BLOCK 0>
<BLOCK 1>                
<BLOCK 2>
...
<BLOCK 15>
<BLOCK 16>
Conv2D                  (7, 7, 1280)
BatchNormalization+ReLu (7, 7, 1280 each)
GlobalAveragePooling2D  (1280)
Dense                   (1000)

Where <BLOCK> consists of:
Conv2D
BatchNormalization+ReLU
DepthwiseConv2D
BatchNormalization+ReLU
Conv2D
BatchNormalization

Structure described here: http://machinethink.net/blog/mobilenet-v2/

The final Conv2D layer and onwards are the target of re-training
The final Dense layer is replaced with a new one of size (2) for classifying male and female
'''

np.random.seed(3)

'''
Define useful filepaths for later
'''
home_path = os.path.dirname(__file__)
train_path = os.path.join(home_path, "dataset/train")
validate_path = os.path.join(home_path, "dataset/validate")
test_path = os.path.join(home_path, "dataset/test")

'''
Downloads MobileNetV2 without its classifier and freezes all but the last 4 layers
'''
ssl._create_default_https_context = ssl._create_unverified_context
model_mobile = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False, pooling='avg')
for layer in model_mobile.layers[:-4]:
    layer.trainable = False

'''
Adds new classifier for 2 classes "Male" and "Female"
'''
model_new = models.Sequential()
model_new.add(model_mobile)
model_new.add(layers.Dense(7))
model_new.add(layers.LeakyReLU(alpha=0.25))
model_new.summary()

'''
Define the data generators, including random data transforms as it is being input
'''
train_trans = ImageDataGenerator(rescale=1. / 255, rotation_range=36, width_shift_range=0.2,
                                 height_shift_range=0.2, horizontal_flip=True)
validate_trans = ImageDataGenerator(rescale=1. / 255)
train_generator = train_trans.flow_from_directory(train_path, target_size=(224, 224), batch_size=80, seed=3,
                                                  class_mode='categorical')
validate_generator = validate_trans.flow_from_directory(validate_path, target_size=(224, 224),
                                                        batch_size=16, seed=3, class_mode='categorical',
                                                        shuffle=False)

'''
Train the neural network!
'''
model_new.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
model_new.fit_generator(train_generator, epochs=1, steps_per_epoch=len(train_generator), verbose=1,
                        validation_data=validate_generator, validation_steps=len(validate_generator))
model_new.save('final_model.h5')