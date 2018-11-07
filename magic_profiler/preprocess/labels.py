from magic_profiler.utils import warning, error, timer

class Labels:
    def __init__(self):
        self.labels = []
        self.removed = []
        #self.properties = {
        #    "base_dir": "",
        #    "source_key": "",
        #    "target_key": ""
        #}

    def add(self, label):
        try:
            new_label = dict(label)
            new_label['_copy'] = 0
            self.labels.append(new_label)
        except Exception as e:
            warning("Could not add new label to Labels object", e)
            return False
        else:
            return True
    
    def adds(self, labels=[]):
        try:
            for label in labels:
                self.add(label)
        except Exception as e:
            warning("Could not add new labels to Label object", e)
            return False
        else:
            return True

    def filter(self, filter_key, lessthan=None, greaterthan=None):
        try:
            new_labels = []
            for label in self.labels:
                if (lessthan and label[filter_key]<lessthan) or (greaterthan and label[filter_key]>greaterthan):
                    self.removed.append(label)
                else:
                    new_labels.append(label)
            self.labels = new_labels
        except Exception as e:
            warning("Could not filter labels in Label object", e)
            return False
        else:
            return True
    
    def shuffle(self):
        try:
            import random
            random.shuffle(self.labels)
        except Exception as e:
            warning("Could not shuffle labels in Label object", e)
            return False
        else:
            return True

    def sort(self, class_key, target_classes=[]):
        try:
            classes = {}
            for i in range(0, len(self.labels)):
                label_class = self.labels[i][class_key]
                if label_class in target_classes or target_classes == []:
                    if label_class not in classes:
                        classes[label_class] = []
                    classes[label_class].append(i)
        except Exception as e:
            warning("Could not sort labels by class name", e)
            return {}
        else:
            return classes
    

    def count(self, class_key, target_classes=[]):
        try:
            class_count = {}
            for i in range(0, len(self.labels)):
                label_class = self.labels[i][class_key]
                if label_class in target_classes or target_classes == []:
                    if label_class not in class_count:
                        class_count[label_class] = 0
                    class_count[label_class] += 1
        except Exception as e:
            warning("Could not count frequency of classes", e)
            return {}
        else:
            return class_count
    

    def upsample(self, class_key, target_classes=[]):
        try:
            count = self.count(class_key, target_classes)
            sort = self.sort(class_key, target_classes)
            max_amount = count[max(count, key=count.get)]
            new_labels = []
            for key, value in sort.iteritems():
                if key in target_classes or target_classes==[]:
                    total_delta = max_amount - count[key]
                    full_sets = int(total_delta/count[key])
                    partial_set = total_delta - count[key]*full_sets
                    new_labels += value*full_sets + value[:partial_set]
            for label in new_labels:
                self.labels.append(self.labels[label])
        except Exception as e:
            warning("Could not upsample labels", e)
            return False
        else:
            return True


    def downsample(self, class_key, target_classes=[]):
        try:
            count = self.count(class_key, target_classes)
            sort = self.sort(class_key, target_classes)
            min_amount = count[min(count, key=count.get)]
            removed_labels = []
            for key, value in sort.iteritems():
                if key in target_classes or target_classes==[]:
                    total_delta = count[key] - min_amount
                    removed_labels += value[:total_delta]
            new_labels = []
            for i in range(0, len(self.labels)):
                if i not in removed_labels:
                    new_labels.append(self.labels[i])
                else:
                    self.removed.append(self.labels[i])
            self.labels = new_labels
        except Exception as e:
            warning("Could not downsample labels", e)
            return False
        else:
            return True
    

    # def config(self, properties):
    #    try:
    #        for key, value in properties.iteritems():
    #            if key in self.properties:
    #                self.properties[key] = value
    #            else:
    #                warning("'"+str(key)+"' is not a property of Labels.")
    #    except Exception as e:
    #        warning("Could not update properties of Labels object")


dummydata = [
    {'name': 'Mitchell','age': 25,'gender': 'male', 'score': 9, 'filepath':'54/m1.jpg'},
    {'name': 'Anthony','age': 19,'gender': 'male', 'score': 4, 'filepath':'15/a1.jpg'},
    {'name': 'Diana','age': 24,'gender': 'female', 'score': 7, 'filepath':'73/d1.jpg'},
    {'name': 'Sarah','age': 37,'gender': 'female', 'score': 8, 'filepath':'9/s1.jpg'},
    {'name': 'Joshua','age': 50,'gender': 'male', 'score': 8, 'filepath':'23/j1.jpg'},
    {'name': 'Kane','age': 21,'gender': 'male', 'score': 7, 'filepath':'5/k1.jpg'},
    {'name': 'William', 'age': 41, 'gender': 'male', 'score': 9, 'filepath':'30/w1.jpg'},
    {'name': 'Erika','age': 30,'gender': 'female', 'score': 10, 'filepath':'49/e1.jpg'},
    {'name': 'Adrian','age': 24,'gender': 'other', 'score': 7, 'filepath':'66/a1.jpg'}
]

def main():
    mylabels = Labels()
    mylabels.adds(dummydata)
    mylabels.filter('score',lessthan=7)
    mylabels.shuffle()
    mylabels.downsample('gender')
    print(str(mylabels.labels))
    
timer(main)
