# A quick and nasty program to turn .bin files into .csv (and maybe remove some channels

import numpy as np
import csv as csv
from glob import glob
import os

def keep3channels_greyOut(inputFileName, outputFileName, patchWidth=256, patchHeight=256, patchChannels=3, labelSize=1):

    pixelSize = patchWidth * patchHeight * patchChannels
    recordSize = labelSize + pixelSize

    with open(inputFilename, "rb") as f:
        allTheData = np.fromfile(f, 'u1')
    numRecords = allTheData.shape[0] / recordSize
    allTheData = allTheData.reshape(numRecords, recordSize)
    print(allTheData.shape)


    # Black and white it...remember it's been normalised so 128/8!
    newRecordSize = labelSize + (patchHeight*patchWidth)
    # allTheData[:, newRecordSize:] = 16 # y-only
    allTheData[:, labelSize:newRecordSize] = 16 # uv-only
    #allTheData[:, (newRecordSize + (patchWidth * patchHeight)):] = 16 # zero out the v
    allTheData[:, newRecordSize:(newRecordSize + (patchWidth * patchHeight))] = 16 # zero out the u

    datayuv = np.asarray(allTheData, 'u1')
    yuvByteArray = bytearray(datayuv)
    with open(outputFilename, "ab") as yuvFile:
        yuvFile.write(yuvByteArray)
    print("All done!")

def onlyKeepKeptChannels(inputFileName, outputFileName, patchWidth=256, patchHeight=256, patchChannels=3, labelSize=1):
    pixelSize = patchWidth * patchHeight * patchChannels
    recordSize = labelSize + pixelSize

    with open(inputFilename, "rb") as f:
        allTheData = np.fromfile(f, 'u1')
    numRecords = allTheData.shape[0] / recordSize
    allTheData = allTheData.reshape(numRecords, recordSize)
    print(allTheData.shape)


    start = labelSize + (patchHeight*patchWidth) # start at the U
    end = start + (patchHeight*patchWidth)
    allTheData = allTheData[:, start:end] # uv-only
    print(allTheData.shape)

    datayuv = np.asarray(allTheData, 'u1')
    yuvByteArray = bytearray(datayuv)
    with open(outputFilename, "ab") as yuvFile:
        yuvFile.write(yuvByteArray)
    print("All done!")

if __name__ == "__main__":
    inputFilename = "test_crop_unshuffled.bin"
    outputFilename = "test_crop_unshuffled_u_only.bin"
    patchWidth = 256
    patchHeight = 256
    patchChannels = 3

    labelSize = 1

    inDir = '/Volumes/LaCie/data/CASIA2/patches_CASIA2'
    outDir = '/Volumes/LaCie/data/CASIA2/patches_CASIA2_colourOnly'

    inFilenames =  glob(inDir + os.sep + '*' + '.bin')

    outFilenames = [n.replace('patches_CASIA2', 'patches_CASIA2_colourOnly') for n in inFilenames]


    for i, inputFilename in enumerate(inFilenames):
        outputFilename = outFilenames[i]
        onlyKeepKeptChannels(inputFilename, outputFilename)

