# Function that moves a file from a source to a location.
# Add error checking

from magic_profiler.utils import warning, error, timer
import matplotlib.image as mplimg
import numpy as np
import skimage.exposure as skiexposure
import skimage.util as skiutil

def loadImage(sourcepath):
    # Loads image from sourcepath as a numpy array and ensures it is in the right format
    # To Do: try with a greyscale png
    try:
        img = mplimg.imread(sourcepath)
        print(img)
        print(img.shape)
        # Converting png to jpg (i.e. slicing/ chopping off the last dimension of the nparray)
        print("img.ndim:")
        print(img.ndim)
        # Converting greyscale to rgb
        if img.ndim == 2:
            print("Greyscale image")
            img = np.dstack((img, img, img))
        if img.shape[2] == 4:
            print("PNG image")
            img = img[:, :, :-1]
        print(img)
        print(img.shape)
    except Exception as e:
        error("Could not load image; "+str(e)) # warning
        return np.array([])
    else:
        return img

def applyFilter(img, noise=0.001, contrast=(15,80)):
    # For no added noise and/or contrast: noise=0, constrast(0,100)
    # Applies filters to array
    plow, phigh = np.percentile(img, (10, 82))
    img = skiexposure.rescale_intensity(img, in_range=(plow, phigh))
    img = skiutil.random_noise(img, var=0.005)
    return img

def saveImage(img, destpath):
    # Saves copy of image to destination folder
    mplimg.imsave(destpath, img)
    return True

def processImage(sourcepath, destpath):
    img = loadImage(sourcepath)
    img = applyFilter(img)
    saveImage(img, destpath)
    return True

def main():
    sourcepath = "/Users/maggieliuzzi/predict_images/110.jpg"
    destpath = "/Users/maggieliuzzi/predict_images/saved/11.jpg"
    processImage(sourcepath, destpath)

timer(main)
