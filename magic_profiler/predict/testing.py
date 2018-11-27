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
        labels = mpl.Labels(labels_path)
        self.labels = labels.labels

    def test_model(self, dataset_path, class_key, destination_key="_destination"):
        results = []
        for label in self.labels:
            print(destination_key)
            print(label)
            result = self.model.from_file(label[destination_key])
            print(result)

tester = Tester()
tester.load_model("./data/datasets/mnv2_filter.h5")
tester.load_labels("./data/datasets/labels.csv")
tester.test_model("./data/datasets","gender")