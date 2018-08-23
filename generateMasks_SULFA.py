# Written by Pam to patch up tampered YUV video

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import cv2 as cv
#import sys
#sys.path.append('/Users/pam/Documents/dev/git/cifar10/')
import functions
import patchIt





#for Au_pic in Au_pic_list:
#    backgrounds = find_background(Au_pic, Sp_pic_list)
#    splice_save(Au_pic, backgrounds, save_dir)

VTD_32thresholders = ["studio", "swann", "swimming", "whitecar", "yellowcar"]
VTD_24thresholders = ["archery"]
VTD_16thresholders = ["bowling", "bullet", "passport", "plane", "highway"]

if __name__ == "__main__":
    myDir = '/Users/pam/Documents/data/SULFA_yuv'
    height = 720
    width = 1280
    num_channels = 3

    shuffled = False

    # for the unshuffled - lets have the file names in order
    unshuffledNames = []

    datasetList = []
    numPatches = 0
    failedList = []

    kernel = np.ones((3, 3), np.uint8)
    #kernel2 = np.ones((8, 8), np.uint8)
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))

    Tp_vid_list = glob(myDir + os.sep + '*' + '_f.yuv')
    Au_vid_list = glob(myDir + os.sep + '*' + '_r.yuv')

    #print(Tp_vid_list)
    #print(Au_vid_list)
    #quit()
    #np.set_printoptions(threshold=np.nan)

    for Tp_vid in Tp_vid_list:
        Au_vid = Tp_vid.replace("_f.yuv","_r.yuv")
        Mask_vid = Tp_vid.replace("_f.yuv","_mask.yuv")
        print("Authentic vid: {} tampered vid: {}".format(Au_vid, Tp_vid))

        width, height, firstTampFrame, interFrameOnly = patchIt.getFrameDetailsFromFilename(Tp_vid)

        frameSize = width * height * 3 // 2
        diffthresh = 64 # try 64 for the first bit

        # Visual inspection gave this
        diffthresh = patchIt.getManuallyEstimatedTamperingThreshold(Tp_vid)

        with open(Au_vid, "rb") as f:
            Au_vid_bytes = np.fromfile(f, 'u1')

        with open(Tp_vid, "rb") as f:
            Tp_vid_bytes = np.fromfile(f, 'u1')


        print("We have {} authentic bytes and {} tampered bytes".format(len(Au_vid_bytes), len(Tp_vid_bytes)))
        Au_num_frames = len(Au_vid_bytes) / frameSize
        Tp_num_frames = len(Tp_vid_bytes) / frameSize
        num_frames = Au_num_frames
        if Tp_num_frames < Au_num_frames:
            num_frames = Tp_num_frames
        bytses = num_frames*frameSize
        print("which means {} authentic frames and {} tampered frames".format(Au_num_frames, Tp_num_frames))
        print("The diffthresh is: {}".format(diffthresh))

        Au_vid_bytes = Au_vid_bytes[0:bytses]
        Tp_vid_bytes = Tp_vid_bytes[0:bytses]


        diff = np.subtract(Au_vid_bytes, Tp_vid_bytes, dtype=np.float32)
        absdiff = np.absolute(diff)
        print(absdiff[0:10])
        absdiff[absdiff < diffthresh] = 0
        print(absdiff[0:10])
        absdiff[absdiff != 0] = 1

        diffMask = absdiff
        diffMask = diffMask.reshape(num_frames, frameSize)

        Uoffset = (width*height)
        Voffset = Uoffset + ((width*height)//4)
        for i in range(0, (width*height)):
        #for i in range(0, 1290):
            chromaOffsetX = (i%width)//2
            chromaOffsetY = (i//width)//2
            totalOffset = chromaOffsetY * (width//2) + chromaOffsetX
            #print("Offsets: Y: {} U: {} V:{} totalOffset: {}".format(i, Uoffset+totalOffset, Voffset+totalOffset, totalOffset))
            diffMask[:, i] = diffMask[:, i] + diffMask[:, Uoffset+totalOffset] + diffMask[:, Voffset+totalOffset]
        diffMask[diffMask != 0] = 255
        diffMask[:, Uoffset:] = 128

        uv = np.full(((frameSize - (width*height))), 128)

        # Temporal filtering
        for i in range(1, num_frames - 1):
            before = diffMask[i - 1, :Uoffset]
            during = diffMask[i, :Uoffset]
            after = diffMask[i + 1, :Uoffset]

            # majority vote
            filtered = np.add(before, during, dtype=np.float32)
            filtered = np.add(filtered, after, dtype=np.float32)
            filtered[filtered < 256.0] = 0
            filtered[filtered > 255.0] = 255
            diffMask[i, :] = np.append(filtered, uv)

        #Now spatial filtering - morphological operators
        for i in range(0, num_frames):
            # single Y channel
            diff = diffMask[i, :Uoffset]
            diff = diff.reshape(height, width)
            #diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel)
            diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
            diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel2)
            ret, diff = cv.threshold(diff, 1, 255, cv.THRESH_BINARY)
            diffMask[i, :] = np.append(diff, uv)









        # it's binary, but make it visible, dammit!
        datayuv = np.asarray(diffMask, 'u1')
        datayuv = datayuv.flatten()
        yuvByteArray = bytearray(datayuv)
        with open(Mask_vid, "wb") as yuvFile:
            yuvFile.write(yuvByteArray)

