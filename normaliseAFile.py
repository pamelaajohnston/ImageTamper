# Written by Pam to patch up tampered YUV video

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import cv2 as cv
import re
#import sys
#sys.path.append('/Users/pam/Documents/dev/git/cifar10/')
import functions


# dataset, yuv420 filename, width, height, tamper start frame, tamper end frame, threshold for tampering
tvd_dataset = 0
tvd_filename = 1
tvd_width = 2
tvd_height = 3
tvd_firstTframe = 4
tvd_lastTframe = 5
tvd_diffthresh = 6 # This is for the difference threshold that silences most noise, manually estimated with yuvdiff
tvd_spatialOrNot = 7 # This is for whether or not the difference is spatial or temporal (i.e. intra (0) or inter (1) frame)
tampered_video_data=[
    ["VTD",     "archery",     1280, 720, 236,  278,    28, 0],
    ["VTD",     "audirs7",      854, 480, 25,    81,    32, 1],
    ["VTD",     "basketball",  1280, 720, 0,    450,    64, 0],
    ["VTD",     "billiards",   1280, 720, 0,    420,    64, 0],
    ["VTD",     "bowling",     1280, 720, 0,    434,    16, 0],
    ["VTD",     "bullet",      1280, 720, 3,    301,    16, 1],
    ["VTD",     "cake",        1280, 720, 160,  299,    64, 0],
    ["VTD",     "camera",      1280, 720, 0,    188,    64, 0],
    ["VTD",     "carpark",     1280, 720, 0,    450,    64, 0],
    ["VTD",     "carplate",    1280, 720, 180,  212,    64, 0],
    ["VTD",     "cctv",        1280, 720, 384,  420,    64, 0],
    ["VTD",     "clarity",     1280, 720, 370,  450,    64, 0],
    ["VTD",     "cuponk",      1280, 720, 49,   96,     64, 1],
    ["VTD",     "dahua",       1280, 720, 0,    420,    64, 0],
    ["VTD",     "football",    1280, 720, 0,    143,    64, 0],
    ["VTD",     "highway",     1280, 720, 48,   235,    64, 0],
    ["VTD",     "kitchen",     1280, 720, 0,    94,     64, 0],
    ["VTD",     "manstreet",   1280, 720, 0,    111,    64, 0],
    ["VTD",     "passport",    1280, 720, 59,   296,    16, 0],
    ["VTD",     "plane",       1280, 720, 0,    38,     16, 0],
    ["VTD",     "pong",        1280, 720, 0,    100,    32, 1], # unknown, messy sequence
    ["VTD",     "studio",      1280, 720, 0,    68,     32, 0],
    ["VTD",     "swann",       1280, 720, 0,    122,    32, 0],
    ["VTD",     "swimming",    1280, 720, 361,  422,    32, 0],
    ["VTD",     "whitecar",    1280, 720, 245,  333,    32, 0],
    ["VTD",     "yellowcar",   1280, 720, 0,    100,    32, 1], # unknown, messy sequence
    ["SULFA",   "01",           320, 240, 0,    209,     0, 0], # SULFA *looks* like frame shuffling but it's all copy-move
    ["SULFA",   "02",           320, 240, 114,  161,     0, 0],
    ["SULFA",   "03",           320, 240, 231,  312,     0, 0],
    ["SULFA",   "04",           320, 240, 60,   122,     0, 0],
    ["SULFA",   "05",           320, 240, 0,    189,     0, 0],
    ["SULFA",   "06",           320, 240, 206,  261,     0, 0],
    ["SULFA",   "07",           320, 240, 0,    128,     0, 0],
    ["SULFA",   "08",           320, 240, 129,  161,     0, 0],
    ["SULFA",   "09",           320, 240, 152,  362,     0, 0],
    ["SULFA",   "10",           320, 240, 93,   157,     0, 0],
    ["Davino",  "01_TANK",      1280,720, 3,    194,     8, 0],
    ["Davino",  "02_MAN",       1280,720, 0,    205,     0, 0],
    ["Davino",  "03_CAT",       1280,720, 146,  216,     0, 0],
    ["Davino",  "04_HELICOPTER",1280, 720,196,  487,     0, 0],
    ["Davino",  "05_HEN",       1280, 720,  0,  167,     0, 0],
    ["Davino",  "06_LION",      1280, 720, 58,  293,     0, 0],
    ["Davino",  "07_UFO",       1280, 720,203,  298,     0, 0],
    ["Davino",  "08_TREE",      1280, 720,  0,  244,     0, 0],
    ["Davino",  "09_GIRL",      1280, 720,207,  370,     0, 0],
    ["Davino",  "10_DOG",       1280, 720,  0,  185,     0, 0],
]


dataset_paths=[
    ["VTD", "/Users/pam/Documents/data/VTD_yuv"],
    ["SULFA", "/Users/pam/Documents/data/SULFA_yuv"],
    ["DAVINO", "/Users/pam/Documents/data/DAVINO_yuv"],
]

# Assumes YUV planar patches (or planar patches at any rate), so channel, height, width order.
def makeNormalisedPatches(frame, pic_w, pic_h, crop_w, crop_h, crop_step, channels, bit_depth, label, addLabel=True):
    mybytes = frame.reshape(channels, pic_h, pic_w)

    patchesList = []

    for y in range(0, (pic_h-crop_h), crop_step):
        yend = y + crop_h
        for x in range(0, (pic_w-crop_w), crop_step):
            xend = x + crop_w
            patch = mybytes[:, y:yend, x:xend]
            patch = patch.flatten()
            if addLabel:
                patch = np.insert(patch, 0, (label*bit_depth))
            patchesList.append(patch)

    patches_array = np.array(patchesList)
    patches_array = np.divide(patches_array, bit_depth)
    print(patches_array.shape)
    return patches_array

def makePatchLabels(frame, pic_w, pic_h, crop_w, crop_h, crop_step, channels):
    mybytes = frame.reshape(channels, height, width)

    labelsList = []

    for y in range(0, (pic_h-crop_h), crop_step):
        yend = y + crop_h
        for x in range(0, (pic_w-crop_w), crop_step):
            xend = x + crop_w
            patch = mybytes[0, y:yend, x:xend]
            patch = patch.flatten()
            nonzs = np.count_nonzero(patch)
            if nonzs == 0:
                labelsList.append(0) # authentic
            else:
                labelsList.append(1)

    labels_array = np.array(labelsList)
    return labels_array

def getFrameDetailsFromFilename(filename):
    b, e = os.path.splitext(filename)
    p, b = os.path.split(b)
    p, f = os.path.split(p)

    width = 0
    height = 0
    firstTampFrame = 0
    interFrameOnly = 0
    #print("Folder: {}, file: {}, ext: {}".format(f, b, e))

    for entry in tampered_video_data:
        #print("searching for {} and {} cf {} \n".format(entry[tvd_dataset], entry[tvd_filename], f))
        if entry[tvd_dataset] in f:
            if entry[tvd_filename] in b:
                width = entry[tvd_width]
                height = entry[tvd_height]
                firstTampFrame = entry[tvd_firstTframe]
                interFrameOnly = entry[tvd_spatialOrNot]
                break

    #print("Folder: {}, file: {}, ext: {}".format(f, b, e))
    #print("Read width: {}, height: {}, t: {}".format(width, height, firstTampFrame))
    return width, height, firstTampFrame, interFrameOnly

def getManuallyEstimatedTamperingThreshold(filename):
    b, e = os.path.splitext(filename)
    p, b = os.path.split(b)
    p, f = os.path.split(p)

    threshold = 0

    for entry in tampered_video_data:
        #print("searching for {} and {} ".format(entry[tvd_dataset], entry[tvd_filename]))
        if entry[tvd_dataset] in f:
            if entry[tvd_filename] in b:
                threshold = entry[tvd_diffthresh]
                break

    #print("Folder: {}, file: {}, ext: {}".format(f, b, e))
    #print("Read width: {}, height: {}, t: {}".format(width, height, firstTampFrame))
    return threshold

def findTamperedRoiFromMask(mask, width, height, frameSize):
    numFrames = len(mask) // frameSize
    mask = mask.reshape((numFrames, frameSize))
    mask = mask[:, 0:(width*height)] # only the y-channel is of interests in the masks
    mask = mask.reshape((numFrames, height, width))
    #print(mask)
    cos = np.nonzero(mask)
    print(cos)
    frame = cos[0][0]
    y= cos[1][0]
    x= cos[2][0]

    return frame, x, y

def getDimsFromFileName(vid):
    if "qcif" in vid:
        return 176, 144
    if "cif" in vid:
        return 352, 288
    if "720p" in vid:
        return 1280, 720
    if "1080p" in vid:
        return 1920, 1080

    dims = re.search(r'([0-9]+)[x|X]([0-9]+)', vid)
    if dims:
        width = int(dims.group(1))
        height = int(dims.group(2))
        return width, height

    # if we hit this, we're getting desperate...
    if "Davino" in vid or "SULFA" in vid or "VTD" in vid:
        width, height, firstTampFrame, interFrameOnly = getFrameDetailsFromFilename(vid)
        return width, height

    return 0,0

def getQuantFromFileName(vid):
    qp = -1
    r = re.search(r'quant_([0-9]+)', vid)
    if r:
        qp = int(r.group(1))
    else:
        r = re.search(r'q([0-9]+)', vid)
        if r:
            qp = int(r.group(1))
    return qp



if __name__ == "__main__":
    summary = "intraForQP"
    dirs = ['/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train','/Volumes/LaCie/data/UCID/train']
    dirs = ['/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_test','/Volumes/LaCie/data/UCID/test']
    #dirs = ['/Users/pam/Documents/data/testyuv']
    cropDim = 80
    cropSpacStep = 48
    cropTempStep = 40
    num_channels = 3
    bit_depth = 8 # please remember to normalise
    patchSize = cropDim * cropDim * 3
    frameStart=0
    frameEnd = 0

    normalise = True



    inFilename = "heatmap_tank_forged.yuv"
    b, ext = os.path.splitext(inFilename)
    outFilename = "{}_normalised{}".format(b, ext)
    if os.path.exists(outFilename):
        os.remove(outFilename)


    width, height = getDimsFromFileName(inFilename)
    if width == 0 or height == 0:
        height = 720
        width = 1280
    frameArea = int(width*height)
    frameSize = width * height * 3 // 2

    print("File is {} and dims are {} x {}".format(inFilename, width, height))

    with open(inFilename, "rb") as f:
        mybytes = np.fromfile(f, 'u1')
    print("There are {} bytes in the file width: {} height: {}".format(len(mybytes), width, height))
    num_frames = len(mybytes) / frameSize

    a = mybytes.reshape((num_frames, frameSize))

    # Here's the crop!
    if (frameStart != 0) or (frameEnd !=0):
        print("cropping")
        a = a[frameStart:frameEnd, :]

    if normalise:
        row_sums = a[:, 0:frameArea].sum(axis=1)
        print(a[0, :])
        #print(row_sums)
        a = a  - (row_sums[:, np.newaxis]//frameArea)
        print(a[0, :])
        a = abs(a)
        print(a[0, :])
        #a = a * 256
        # now reset the uv components to 128...?
        a[:, frameArea: ] = 128
        print(a[0, :])


    functions.appendToFile(a, outFilename)


    quit()


