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

labels = mpl.Labels()
labels.adds(mat_labels)
labels.filter("first_score", lessthan=4)
labels.filter("second_score", greaterthan=0)
labels.filter("gender", isnotin=[0.0, 1.0])
labels.assign()
labels.process("gender", "D:\\Dev\\wiki_crop\\wiki_crop", "source", "data\\dataset")
labels.export_csv("data\\labels.csv")