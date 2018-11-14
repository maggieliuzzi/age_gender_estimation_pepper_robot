import clint.textui as cl
import csv
import math
import os
import random

import magic_profiler.preprocess.images as mpimg
from magic_profiler.utils import warning, error, timer

class Labels:
    def __init__(self):
        self.labels = []
        self.removed = []

    def add(self, label):
        try:
            new_label = dict(label)
            new_label["_role"] = None
            new_label["_destination"] = None
            self.labels.append(new_label)
        except Exception as e:
            warning("Could not add new label to Labels object", e)
            return False
        else:
            return True
    
    def adds(self, labels=[]):
        n = 0
        try:
            print("Adding labels...")
            for label in labels:
                self.add(label)
                n += 1
        except Exception as e:
            warning("Could not add new labels to Label object", e)
            return False
        else:
            print("Successfully added {} labels.".format(n))
            return True

    def filter(self, filter_key, isin=None, isnotin=None, lessthan=None, greaterthan=None):
        try:
            print("Filtering labels by {}...".format(filter_key))
            new_labels = []
            for label in self.labels:
                if (lessthan != None and label[filter_key]<lessthan) or (greaterthan != None and label[filter_key]>greaterthan) or (isin != None and label[filter_key] in isin) or (isnotin != None and label[filter_key] not in isnotin):
                    self.removed.append(label)
                else:
                    new_labels.append(label)
            self.labels = new_labels
        except Exception as e:
            warning("Could not filter labels in Label object", e)
            return False
        else:
            print("Successfully filtered labels. {} labels remain.".format(len(self.labels)))
            return True
    
    def shuffle(self):
        try:
            print("Shuffling order of labels...")
            random.shuffle(self.labels)
        except Exception as e:
            warning("Could not shuffle labels in Label object", e)
            return False
        else:
            print("Successfully shuffled {} labels.".format(len(self.labels)))
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
            print("Upsampling labels by {}...".format(class_key))
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
            print("Labels upsampled successfully. Total labels is {}.".format(len(self.labels)))
            return True

    def downsample(self, class_key, target_classes=[]):
        try:
            print("Downsampling labels by {}...".format(class_key))
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
            print("Labels downsampled successfully. {} labels remain.".format(len(self.labels)))
            return True

    def assign(self, training=0.65, validation=0.10):
        try:
            if training+validation<1:
                images = len(self.labels)
                train_images = int(math.floor(images*training))
                validate_images = int(math.floor(images*validation))
                print("Assigning roles to labels: {} training; {} validation; {} testing]...".format(train_images, validate_images, images-(train_images+validate_images)))
                for i in range(0, train_images):
                    self.labels[i]["_role"] = "train"
                for i in range(train_images, train_images+validate_images):
                    self.labels[i]["_role"] = "validate"
                for i in range(train_images+validate_images, images):
                    self.labels[i]["_role"] = "test"
            else:
                warning("Could not assign roles to labels. Training + validation should be < 1")
        except Exception as e:
            warning("Could not assign roles to labels", e)
            return False
        else:
            print("Roles assigned successfully to {} labels.".format(len(self.labels)))
            return True

    def process(self, class_key, source_base, source_key, destination_base, filter_noise=0, filter_contrast=(0,100)):
        bar_current = 0
        bar_end = len(self.labels)
        print("Processing labels to generate dataset...")
        with cl.progress.Bar(expected_size=bar_end) as bar:
            for i in range(0, bar_end):
                copy = 0
                source_path = os.path.join(source_base, self.labels[i][source_key])
                while copy < 1000:
                    file_name, file_ext = os.path.splitext(os.path.basename(source_path))
                    destination_cont = os.path.join(str(self.labels[i]["_role"]), str(self.labels[i][class_key]), file_name+"_"+str(copy)+file_ext)
                    destination_path = os.path.join(destination_base, destination_cont)
                    if not os.path.isfile(destination_path):
                        self.labels[i]["_destination"] = destination_cont
                        img = mpimg.load_image(source_path)
                        img = mpimg.apply_filter(img, filter_noise, filter_contrast)
                        mpimg.save_image(img, destination_path)
                        break
                    else:
                        copy += 1
                bar_current += 1
                bar.show(bar_current)
        print("Dataset of {} images generated successfully.".format(len(self.labels)))
        return True

    def export_csv(self, destination):
        try:
            print("Writing labels to file...")
            with open(destination,'wb') as f:
                k = self.labels[0].keys()
                w = csv.DictWriter(f, k)
                w.writerow({str(k_): k_ for k_ in k})
                w.writerows(self.labels)
        except Exception as e:
            warning("Could not save labels to file", e)
        else:
            print("Labels saved successfully to {}.".format(destination))
        
    def import_csv(self, source):
        try:
            print("Loading labels from file...")
            with open(source,'r') as f:
                w = csv.DictReader(f)
                imported_labels = [row for row in w]
                self.adds(imported_labels)
        except Exception as e:
            warning("Could not load labels from file", e)
        else:
            print("Labels loaded successfully from {}.".format(source))

    def bin():
        pass

    def remove():
        pass
    
    def restore():
        pass
    
    def reset():
        pass
