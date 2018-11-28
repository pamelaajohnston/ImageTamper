from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import functions
import time
import glob


def convertFileToYUV(infilename, outfilename):

    if "TIF" in infilename:
        width, height, pixels = functions.convertTifToYUV420(infilename)
        functions.appendToFile(pixels, outfilename)
        return width, height
    if "PNG" in infilename:
        img = Image.open(infilename)
        print(img)
        print(img.size)
        print(img.mode)
        #I = img.load()
        width, height = img.size
        print("width {}, height {}".format(width, height))
        #print(pixels)
        #pixels = np.asarray(Image.open(infilename))
        if "L" in img.mode:
            print("It's greyscale")
            if "A" in img.mode:
                print("There's an alpha channel")
                pixels = np.asarray(img.getdata()).reshape(height, width, 2)
                pixels = pixels[:, :, 0]
            else:
                pixels = np.asarray(img.getdata()).reshape(img.size)
        print("The shame of the pixels: {}".format(pixels.shape))
        pixels = pixels.flatten()
        frameSize = width * height
        uv = np.full((frameSize // 2), 128)
        print("The framesize is {}, width {}, height {}".format(frameSize, width, height))
        print(uv.shape)
        functions.appendToFile(pixels, outfilename)
        functions.appendToFile(uv, outfilename)

        return width, height

if __name__ == "__main__":

    print("Converting a .png to YUV")

    # Get a list of file names
    dataRoot = "/Volumes/LaCie/data/RealisticTampering/multiscale-prnu/data/images_1080p/"
    vidlist = glob.glob(os.path.join(dataRoot, "*", "tampered-realistic", "*.TIF"))
    print vidlist
    print (len(vidlist))
    numFiles = len(vidlist)
    numImgsPerOutFile = 20
    outputDir = "/Users/pam/Documents/data/realisticTampering/"
    logName = "{}mylog.txt".format(outputDir)
    log = open(logName, "w")
    log.write("original file ---> yuv file; mask file ---> mask yuv \n")

    print("Clean up any existing stuff")
    for i,j in enumerate(range(0, len(vidlist), numImgsPerOutFile)):
        outfilename = "{}all_1080p_{}.yuv".format(outputDir, i)
        maskoutfilename = "{}mask_1080p_{}.yuv".format(outputDir, i)
        authoutfilename = "{}auth_1080p_{}.yuv".format(outputDir, i)
        if os.path.exists(outfilename):
            os.remove(outfilename)
            print("Removing {}".format(outfilename))
        if os.path.exists(maskoutfilename):
            os.remove(maskoutfilename)
            print("Removing {}".format(maskoutfilename))
        if os.path.exists(authoutfilename):
            os.remove(authoutfilename)
            print("Removing {}".format(authoutfilename))

    print("Convert the videos")
    for i, vid in enumerate(vidlist):

        infilename = vid
        maskfilename = infilename.replace("tampered-realistic", "ground-truth")
        maskfilename = maskfilename.replace("TIF", "PNG")
        authfilename = infilename.replace("tampered-realistic", "pristine")

        index = int(i//numImgsPerOutFile)
        outfilename = "{}all_1080p_{}.yuv".format(outputDir, index)
        maskoutfilename = "{}mask_1080p_{}.yuv".format(outputDir, index)
        authoutfilename = "{}auth_1080p_{}.yuv".format(outputDir, index)

        print("Converting {} and its mask {}".format(infilename, maskfilename))
        log.write("{} ---> {}; {} ---> {}; {} ---> {}\n".format(infilename, outfilename,
                                                                maskfilename, maskoutfilename,
                                                                authfilename, authoutfilename))
        convertFileToYUV(infilename, outfilename)
        convertFileToYUV(maskfilename, maskoutfilename)
        convertFileToYUV(authfilename, authoutfilename)

    log.close()

    quit()
