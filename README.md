# Age Recognition Project

This project is for the development of a convolutional neural network (CNN) for assessing the gender of faces.

It forms part of Maggie Liuzzi's & Mitchell Clarke's Spring 2018 Research Project at UTS, and is developed in collaboration with the UTS Magic Lab.

The dataset currently being explored is the Wiki half of the IMDB-WIKI dataset found here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ - In particular, "Wiki -> Download faces only (1GB)"



## Installation

The project may be moved to a docker file at a later date, but for the time being, here is how you install it:

1. Install **Python 3.6** if you haven't - it needs to be 3.6, since some dependencies don't support 3.7. Also make sure that you have installed **virtualenv**. If you haven't, you can install it globally using pip:

```shell
pip install virtualenv
```

2. Clone this repo to a directory of your choice:

```shell
git clone https://gitlab.com/maggieliuzzi/agerecognition.git
```

3. Enter the folder and create a new virtual environment:

```
cd AgeRecognition
virtualenv -p [path to your python 3.6 interpreter] venv
```

4. Activate the new virtual environment (omit "source" if on Windows):

```shell
source venv/bin/activate
```

5. Once your environment is activated, install the project's dependencies:

```shell
pip install -r requirements.txt
```

With that, you should be good to go!

Alternatively, you can follow installation up to the end of Step 2 and then use PyCharm to create the virtual environment for you. If you do that, make sure pip is installed at /GenderRecognition/venv/Scripts and that the terminal in PyCharm says "(venv)" before the current location. If it does, you should be good to continue from the start of Step 5.



Currently:

* **train.py** imports a model for training

* **mat.py** converts IMDB-WIKI's .mat file containing the data labels into an easier-to-read .csv
