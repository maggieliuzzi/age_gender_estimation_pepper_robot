# Age Recognition Project

This project is for the development of a convolutional neural network (CNN) for assessing the age and gender of people's faces.

It forms part of Maggie Liuzzi's & Mitchell Clarke's Spring 2018 Research Project at UTS, and is developed in collaboration with the UTS Magic Lab.

The datasets explored are the following: a) Adience (https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification - 1GB), 
     b) Wiki half of IMDB-WIKI (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ - In particular, "Wiki -> Download faces only (1GB)")
    




## Installation

The project may be moved to a docker file at a later date, but for the time being, here is how you install it:

1. Install **Python 2.7** if you haven't - (also works with Python 3.6). Also make sure that you have installed **virtualenv**. If you haven't, you can install it globally using pip:

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
virtualenv -p [path to your python 2.7 interpreter] venv
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

Alternatively, you can follow installation up to the end of Step 2 and then use PyCharm to create the virtual environment for you. If you do that, make sure pip is installed at /AgeRecognition/venv/bin and that the terminal in PyCharm says "(venv)" before the current location. If it does, you should be good to continue from the start of Step 5.



* **Adience_Age_5y/** contains the required python scripts to train an age-recognition model with equi-width 5-year bins.
* **Adience_Age_10y/** contains the required python scripts to train an age-recognition model with equi-width 10-year bins.
* **Adience_Age_15y/** contains the required python scripts to train an age-recognition model with equi-width 15-year bins.
* **Adience_Age_Equi_Depth/** contains the required python scripts to train an age-recognition model with equi-depth bins.
* **Adience_Age_Equi_Depth/10_ED_bins/** contains the required python scripts to train an age-recognition model with 10 equi-depth bins.

* **Adience_Gender/** contains the required python scripts to train a gender-recognition model.


* **form...py** scripts perform data processing to get the image list to a usable format.
* **proc...py** scripts separate data into training, validation and testing sets.
* **train...py** scripts train the network over a certain number of epochs and outputs an .h5 model.
* **server...py** scripts start a server that receives HTTP POST request with individual test images and outputs the estimated probabilities. 
Eg: http://0.0.0.0:4000/predict
"prediction_age": {
    "1-15": 0.01101667433977127,
    "16-30": 0.9543766379356384,
    "31-45": 0.03458355739712715,
    "46-60": 0.000023107986635295674
},
"prediction_gender": {
    "Female": 0.10200900584459305,
    "Male": 0.8979910016059875
}

* **test...py** scripts test the quality of a model with the images in the test/ folder.

* **predict_gender_age.py** takes a gender-recognition model, an age-recognition model and an image as arguments and predicts the gender and age of the person in the image.
