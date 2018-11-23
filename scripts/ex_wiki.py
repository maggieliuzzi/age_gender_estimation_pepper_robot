#!/usr/bin/env python

import argparse
import magic_profiler.preprocess.labels as mpl
import magic_profiler.train.model as mpm
import os
import scipy.io as sio

'''
python scripts/ex_wiki.py -so "../wiki_crop/wiki_crop" -de "./data/dataset_mn2_up" -fa 2 -sa "up" -mo "mobilenetv2" -re 4 -ba 12 -va 8 -ep 10 -mn "mn2_up" -nof
'''

# Retrieve information from the command line
parser = argparse.ArgumentParser(description='Preprocesses the WIKI Face Crop dataset for use with Keras')
parser.add_argument('-source', metavar='PATH', help='Root directory of the dataset', required=True)
parser.add_argument('-destination', metavar='PATH', help='Desired location of processed images', required=True)
parser.add_argument('-facescore', metavar='FLOAT', help='Minimum face score required to use image in training', default=2)
parser.add_argument('-sampling', metavar='STR', help='Applies resampling to labels, if specified. Can be "up" or "down"', default="")
parser.add_argument('-model', metavar='STR', help='Which Keras model to use as basis for retraining', default="MobileNetV2")
parser.add_argument('-pooling', metavar='STR', help='Type of pooling to include at the top of base model. Can be "avg", "max", or "none"', default="avg")
parser.add_argument('-retrain', metavar='INT', help='Number of layers in model to retrain', default='')
parser.add_argument('-batchsize', metavar='INT', help='Batch size to use when training networks', default=20)
parser.add_argument('-valsize', metavar='INT', help='Validation batch size to use when training networks', default=10)
parser.add_argument('-epochs', metavar='INT', help='Number of epochs to train the networks for', default=10)
parser.add_argument('-mname', metavar="STR", help="The base name of intermediate models saved after each epoch", default="model")
parser.add_argument('-noprocess', help='If set, preprocessing is skipped', action='store_true')
parser.add_argument('-nofilter', help='If set, preprocessing applies no image filters', action='store_true')
parser.add_argument('-notrain', help='If set, the model will be saved without training', action='store_true')
args = parser.parse_args()

if args.noprocess == False:
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
    labels.filter("first_score", lessthan=float(args.facescore))
    labels.filter("second_score", greaterthan=0)
    labels.filter("gender", isnotin=[0.0, 1.0])

    # Randomly resample labels if required
    if args.sampling.lower() == "up":
        labels.shuffle()
        labels.upsample('gender')
    elif args.sampling.lower() == "down":
        labels.shuffle()
        labels.downsample('gender')

    # Randomly assign images to be for training, validation and test
    labels.shuffle()
    labels.assign()

    # Process the labels by loading, transforming (if desired) and saving dataset images to the new location
    if args.nofilter == False:
        labels.process("gender", args.source, "source", args.destination, filter_noise=0.002, filter_contrast=(10,82))
    else:
        labels.process("gender", args.source, "source", args.destination)
    csv_destination = os.path.join(args.destination, "labels.csv")
    labels.export_csv(csv_destination)

# Create the machine learning model
model = mpm.Model(verbose=1)
if args.retrain=='' or not args.retrain:
    if args.model.lower()=="mobilenetv2":
        tl = 4
    elif args.model.lower()=="inceptionv3":
        tl = 32
    elif args.model.lower()=="nasnetmobile":
        tl = 45
    elif args.model.lower()=="densenet121"
        tl = 117
    else:
        tl = 0
else:
    tl = int(args.retrain)
model.load(model_uri=args.model, input_shape=(224, 224, 3), trainable_layers=tl, pooling=args.pooling)
model.add(2)

# Train and save the model
if args.notrain == False:
    model.train(dataset_directory=args.destination, batch_size=int(args.batchsize), val_batch_size=int(args.valsize), epochs=int(args.epochs), model_base_path=os.path.join(args.destination, args.mname))
model.save(os.path.join(args.destination,"{}.h5".format(args.mname)))
