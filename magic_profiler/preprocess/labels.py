from magic_profiler.utils import warning, error

class Labels:
    def __init__(self):
        self.labels = []
        self.properties = {
            "base_dir": "",
            "source_key": "",
            "target_key": ""
        }
    
    def set_properties(self, properties):
        if isinstance(properties, dict):
            for key, value in properties.iteritems():
                if key in self.properties:
                    self.properties[key] = value
                else:
                    warning("Could not update '"+str(key)+"'. It is not a valid property.")
        else:
            warning("Could not update properties. Input should be a dictionary.")

    def add_label(self, label):
        print("Adding a label.")
    
    def add_labels(self, labels):
        print("Adding lots of labels.")        

mylabels = Labels()
mylabels.set_properties({'target_key':'gender','source_key':'source','whatever':'anything'})
print(mylabels.properties)
