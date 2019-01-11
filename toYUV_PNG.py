from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import functions
import time



if __name__ == "__main__":

    print("Converting a .png to YUV")

    infilename = "/Users/pam/Documents/data/Breast/p00001/CC.png"
    img = Image.open(infilename)
    print(img)
    I = img.load()
    orig_width, orig_height = img.size
    width = (orig_width//16)*16
    height = (orig_height//16)*16
    print("width {}, height {}".format(width, height))
    #print(pixels)
    pixels = np.asarray(Image.open(infilename))
    pixels = pixels[:height, :width]
    pixels = pixels.flatten()
    frameSize = width * height
    uv = np.full((frameSize // 2), 128)

    outfilename = "/Users/pam/Documents/data/Breast/p00001/CC_{}x{}.yuv".format(width, height)
    if os.path.exists(outfilename):
       os.remove(FLAGS.heatmap)
    functions.appendToFile(pixels, outfilename)
    functions.appendToFile(uv, outfilename)
    quit()
