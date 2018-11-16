#!/usr/bin/env python

import magic_profiler.preprocess.labels as mpl
import scipy.io as sio

mat_file = sio.loadmat("D:\\Dev\\wiki.mat")['wiki'][0][0]
mat_labels = [{
    "dob": mat_file[0][0][i],
    "date": mat_file[1][0][i],
    "source": mat_file[2][0][i][0],
    "gender": mat_file[3][0][i],
    "first_score": mat_file[6][0][i],
    "second_score": mat_file[7][0][i]
} for i in range(0, len(mat_file[0][0]))]

labels = mpl.Labels(mat_labels)
labels.filter("first_score", lessthan=5)
labels.filter("second_score", greaterthan=0)
labels.filter("gender", isnotin=[0.0, 1.0])
labels.bin('first_score', 5, 10, 1)
labels.shuffle()
labels.assign()
labels.process("gender", "D:\\Dev\\wiki_crop\\wiki_crop", "source", "data\\dataset", filter_noise=0.002, filter_contrast=(10,82))
labels.export_csv("data\\labels.csv")
