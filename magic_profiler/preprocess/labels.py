import clint.textui as cl
import csv
import math
import os
import random

import magic_profiler.preprocess.images as mpimg
from magic_profiler.utils import warning, error, timer

# TO DO:
# * Adding catches for using functions without any labels loaded
# * Improved error checking with specific messages
# * Adding comments + docstrings
# * Number <-> category functions
# * Stratified sampling
# * General optimization

class Labels:
    def __init__(self, labels=None, verbose=0):
        say("Creating new Labels object...")
        self.labels = []
        self.removed = []
        self.verbose = verbose
        if labels:
            if isinstance(labels, list):
                self.adds(labels)
            elif os.path.isfile(labels):
                self.import_csv(labels)
            else:
                warning("Could not add labels to new Label object", e)
        say("New Labels object created successfully.")


    def say(self, m):
        '''Prints a message if self.verbose is > 0'''
        if self.verbose>0:
            print(m)


    def add(self, label):
        '''Adds a dictionary to the list of labels'''
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
        '''Adds a list of multiple dictionaries to the list of labels'''
        n = 0
        try:
            say("Adding labels...")
            for label in labels:
                self.add(label)
                n += 1
        except Exception as e:
            warning("Could not add new labels to Label object", e)
            return False
        else:
            say("Successfully added {} labels.".format(n))
            return True

    def filter(self, filter_key, isin=None, isnotin=None, lessthan=None, greaterthan=None):
        '''Filters out labels which satisfies the specified critiera'''
        try:
            say("Filtering labels by {}...".format(filter_key))
            new_labels = []
            for label in self.labels:
                if (
                    (lessthan != None and label[filter_key]<lessthan) or 
                    (greaterthan != None and label[filter_key]>greaterthan) or 
                    (isin != None and label[filter_key] in isin) or 
                    (isnotin != None and label[filter_key] not in isnotin)
                   ):
                    self.removed.append(label)
                else:
                    new_labels.append(label)
            self.labels = new_labels
        except Exception as e:
            warning("Could not filter labels in Label object", e)
            return False
        else:
            say("Successfully filtered labels. {} labels remain.".format(len(self.labels)))
            return True
    
    def shuffle(self):
        '''Shuffles the order of the labels'''
        try:
            say("Shuffling order of labels...")
            random.shuffle(self.labels)
        except Exception as e:
            warning("Could not shuffle labels in Label object", e)
            return False
        else:
            say("Successfully shuffled {} labels.".format(len(self.labels)))
            return True

    def sort(self, class_key, target_classes=[]):
        '''Returns the indexes of all labels belonging to each class'''
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
        '''Returns the number of labels belonging to each class'''
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
        '''Upsamples by duplicating labels, to achieve an equal number of each class'''
        try:
            say("Upsampling labels by {}...".format(class_key))
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
            say("Labels upsampled successfully. Total labels is {}.".format(len(self.labels)))
            return True

    def downsample(self, class_key, target_classes=[]):
        '''Downsamples by removing labels, to achieve an equal number of each class'''
        try:
            say("Downsampling labels by {}...".format(class_key))
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
            say("Labels downsampled successfully. {} labels remain.".format(len(self.labels)))
            return True

    def assign(self, training=0.65, validation=0.10):
        '''Assigns all labels with training, validation an testing roles'''
        try:
            if training+validation<1:
                images = len(self.labels)
                train_images = int(math.floor(images*training))
                validate_images = int(math.floor(images*validation))
                say("Assigning roles to labels: {} training; {} validation; {} testing...".format(train_images, validate_images, images-(train_images+validate_images)))
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
            say("Roles assigned successfully to {} labels.".format(len(self.labels)))
            return True

    def process(self, class_key, source_base, source_key, destination_base, filter_noise=0, filter_contrast=(0,100)):
        '''Loads, processes (including image filtering) and saves the corresponding images for each label'''
        bar_current = 0
        bar_end = len(self.labels)
        say("Processing labels to generate dataset...")
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
        say("Dataset of {} images generated successfully.".format(len(self.labels)))
        return True

    def export_csv(self, destination):
        '''Exports the list of labels as a csv file'''
        try:
            say("Writing labels to file...")
            with open(destination,'wb') as f:
                k = self.labels[0].keys()
                w = csv.DictWriter(f, k)
                w.writerow({str(k_): k_ for k_ in k})
                w.writerows(self.labels)
        except Exception as e:
            warning("Could not save labels to file", e)
        else:
            say("Labels saved successfully to {}.".format(destination))
        
    def import_csv(self, source):
        '''Imports a list of labels from a csv file'''
        try:
            say("Loading labels from file...")
            with open(source,'r') as f:
                w = csv.DictReader(f)
                imported_labels = [row for row in w]
                self.adds(imported_labels)
        except Exception as e:
            warning("Could not load labels from file", e)
        else:
            say("Labels loaded successfully from {}.".format(source))

    def remove(self, n):
        '''Removes labels from the list of labels'''
        self.removed = self.labels[:n]
        self.labels = self.labels[n:]
    
    def restore(self):
        '''Restores removed labels to the list of labels'''
        self.adds(self.removed)
        self.removed = None
    
    def reset(self):
        '''Empties the list of labels'''
        try:
            say("Resetting Labels object.")
            self.labels = None
            self.removed = None
        except Exception as e:
            warning("Could not reset Labels object", e)
            return False
        else:
            say("Successfully reset Labels object.")
            return True
    
    def bin(self, bin_key, mini, maxi, step):
        '''Sorts labels into bins according to a numerical attribute'''
        try:
            say("Binning labels by {}...".format(bin_key))
            binned_key = bin_key+"_bin"
            for i in range(0,len(self.labels)):
                value = float(self.labels[i][bin_key])
                if value<mini:
                    lower = "inf"
                    upper = mini
                elif value>maxi:
                    low = maxi
                    upper = "inf"
                else:
                    lower = value - (value-mini)%step
                    upper = value + step - (value-mini)%step
                self.labels[i][binned_key] = "["+str(lower)+"-"+str(upper)+")"
        except Exception as e:
            warning("Could not bin labels", e)
            return False
        else:
            say("Successfully binned labels to {}".format(binned_key))
            return True
