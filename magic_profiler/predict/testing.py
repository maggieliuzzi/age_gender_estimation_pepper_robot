import magic_profiler.predict.predict as mpp
import magic_profiler.preprocess.labels as mpl

class Tester:
    def __init__(self):
        self.model = None
        self.labels = []
        self.results = None
        self.class_key = None
    
    def load_model(self, model_path):
        model = mpp.Predictor()
        model.prepare(model_path)
        self.model = model
    
    def load_labels(self, labels_path):
        self.labels = mpl.Labels(labels_path)

    def test_model(self, dataset_path, class_key, destination_key="_destination"):
        results = []
        for label in self.labels:
            result = self.model.from_file(label[destination_key])[0]
            print(result)

tester = Tester()
tester.load_model(".\\data\\mnv2_filter.h5")
tester.load_labels(".\\data\\dataset_filter\\labels.csv")
tester.test_model(".\\data\\dataset_filter","gender")