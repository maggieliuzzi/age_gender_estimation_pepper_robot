#!/usr/bin/env python

import argparse
import magic_profiler.preprocess.labels as mpl
import magic_profiler.train.model as mpm
import os
import scipy.io as sio

# Retrieve path to dataset from the command line
parser = argparse.ArgumentParser(description='Preprocesses the WIKI Face Crop dataset for use with Keras')
parser.add_argument('--source', metavar='PATH', help='Root directory of the dataset', required=True)
parser.add_argument('--destination', metavar='PATH', help='Desired location of processed images', required=True)
parser.add_argument('--skip', help='If set, preprocessing is skipped', action='store_true')
args = parser.parse_args()

if args.skip == False:
    # Load labels from wiki.mat file into memory
    source = os.path.join(args.source, "wiki.mat")
    mat_file = sio.loadmat(source)['wiki'][0][0]
    mat_labels = [{
        "dob": mat_file[0][0][i],
        "date": mat_file[1][0][i],
        "source": mat_file[2][0][i][0],
        "gender": mat_file[3][0][i],
        "first_score": mat_file[6][0][i],
        "second_score": mat_file[7][0][i]
    } for i in range(0, len(mat_file[0][0]))]

    # Load labels into magic_profiler's Label handler object
    labels = mpl.Labels(mat_labels, verbose=1)

    # Filter out unwanted images
    labels.filter("first_score", lessthan=6)
    labels.filter("second_score", greaterthan=0)
    labels.filter("gender", isnotin=[0.0, 1.0])

    # Upsample the labels to even out the genders
    labels.upsample('gender')

    # Randomly assign images to be for training, validation and test
    labels.shuffle()
    labels.assign()

    # Process the labels by loading, transforming and saving dataset images to the new location
    labels.process("gender", args.source, "source", args.destination, filter_noise=0.002, filter_contrast=(10,82))
    csv_destination = os.path.join(args.destination, "labels.csv")
    labels.export_csv(csv_destination)

# Create the machine learning model
model = mpm.Model()
model.load(model_uri="MobileNetV2", input_shape=(224, 224, 3), trainable_layers=0, pooling='avg')
model.add(2)

# Train the model
model.train(dataset_directory=args.destination, batch_size=20, val_batch_size=10, epochs=1)
model.save(args.destination)