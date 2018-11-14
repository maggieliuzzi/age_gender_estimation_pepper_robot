# Function that moves a file from a source to a location.
# To-do: Add error checking

from magic_profiler.utils import warning, error, timer
import numpy as np
import matplotlib.image as mplimg
import skimage.exposure as skiexposure
import skimage.util as skiutil

def load_image(sourcepath):
    # Loads image from sourcepath as a numpy array and ensures it is in the right format
    # To Do: try with a greyscale png
    try:
        img = mplimg.imread(sourcepath)
        if img.ndim == 2:
            img = np.dstack((img, img, img))
        if img.shape[2] == 4:
            img = img[:, :, :-1]
    except Exception as e:
        error("Could not load image; "+str(e)) # warning
        return np.array([])
    else:
        return img

def apply_filter(img, noise=0.001, contrast=(15,80)):
    # For no added noise and/or contrast: noise=0, constrast(0,100)
    # Applies filters to array
    plow, phigh = np.percentile(img, (10, 82))
    img = skiexposure.rescale_intensity(img, in_range=(plow, phigh))
    img = skiutil.random_noise(img, var=0.005)
    return img

def save_image(img, destpath):
    # Saves copy of image to destination folder
    import os
    destdir = os.path.dirname(destpath)
    if not os.path.isdir(destdir):
        os.makedirs(destdir)
    mplimg.imsave(destpath, img)
    return True

def process_image(sourcepath, destpath):
    img = load_image(sourcepath)
    img = apply_filter(img)
    save_image(img, destpath)
    return True