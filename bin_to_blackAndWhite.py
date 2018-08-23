# A quick and nasty program to turn .bin files into .csv (and maybe remove some channels

import numpy as np
import csv as csv
from glob import glob
import os
import functions

x264 = "../x264/x264"

def keep3channels_greyOut(inputFileName, outputFileName, patchWidth=256, patchHeight=256, patchChannels=3, labelSize=1):

    pixelSize = patchWidth * patchHeight * patchChannels
    recordSize = labelSize + pixelSize

    with open(inputFilename, "rb") as f:
        allTheData = np.fromfile(f, 'u1')
    numRecords = allTheData.shape[0] / recordSize
    allTheData = allTheData.reshape(numRecords, recordSize)
    print(allTheData.shape)


    # Black and white it...remember it's been normalised so 128/8!
    yStart = labelSize
    uStart = labelSize + (patchHeight*patchWidth)
    vStart = uStart + (patchHeight*patchWidth)
    newRecordSize = labelSize + (patchHeight*patchWidth)
    allTheData[:, yStart:uStart] = 16 # y_channel
    #allTheData[:, uStart:vStart] = 16 # y_channel
    allTheData[:, vStart:      ] = 16 # y_channel

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

def compressWithConstantQuant(inputFileName, theOutputFileName, qps=[0,], patchWidth=256, patchHeight=256, patchChannels=3, labelSize=1):
    pixelSize = patchWidth * patchHeight * patchChannels
    recordSize = labelSize + pixelSize
    bitDepth = 8

    with open(inputFileName, "rb") as f:
        allTheData = np.fromfile(f, 'u1')
    numRecords = allTheData.shape[0] / recordSize
    allTheData = allTheData.reshape(numRecords, recordSize)
    print(allTheData.shape)

    # Convert each picture to a YUV file
    tempYUVFileName = "temp_{}x{}.yuv".format(patchWidth, patchHeight)
    open(tempYUVFileName, 'w').close()

    labels = allTheData[:, 0].copy()
    yuvFrames = allTheData[:, 1:].copy()
    yuvFrames = yuvFrames * bitDepth
    print("Saving frames to {}, shape {}".format(tempYUVFileName, yuvFrames.shape))
    for frame in yuvFrames:
        #print("Frame")
        yuv420 = functions.YUV444_2_YUV420(frame, patchWidth, patchHeight)
        functions.appendToFile(yuv420, tempYUVFileName)

    # encode that file as all intra...
    for qp in qps:
        tempH264FileName = "temp_{}x{}_{}.264".format(patchWidth, patchHeight, qp)
        tempDecompFileName = "temp_{}x{}_{}_decomp.yuv".format(patchWidth, patchHeight, qp)
        open(tempH264FileName, 'w').close()
        open(tempDecompFileName, 'w').close()
        #outputFileName = theOutputFileName.replace('.bin', '_{}.bin'.format(qp))
        outputFileName = theOutputFileName.replace('/t', '/qp{}/t'.format(qp))
        functions.compressFile(x264, tempYUVFileName, patchWidth, patchHeight, qp, tempH264FileName, tempDecompFileName, deblock=False, intraOnly=True, verbose = False)

        # read in the decompressed YUV file:
        with open(tempDecompFileName, "rb") as f:
            allTheYUV420 = np.fromfile(f, 'u1')
        allTheYUV420 = allTheYUV420.reshape((numRecords, -1))
        datasetList = []
        for idx, frame in enumerate(allTheYUV420):
            datayuv = functions.YUV420_2_YUV444(frame, patchWidth, patchHeight)
            datayuv = np.divide(datayuv, 8)
            label = labels[idx]
            datayuv = np.concatenate((np.array([label]), datayuv), axis=0)
            datayuv = datayuv.flatten()
            datasetList.append(datayuv)

        dataset_array = np.array(datasetList)
        print("Size of Dataset: {}".format(dataset_array.shape))
        functions.appendToFile(dataset_array, outputFileName)

    print("All done!")

if __name__ == "__main__":
    patchWidth = 128
    patchHeight = 128
    patchChannels = 3

    labelSize = 1

    #inDir = '/Volumes/LaCie/data/CASIA2/patches_CASIA2_256'
    inDir = '/Volumes/LaCie/data/CASIA2/CASIA_patches_128'


    inFilenames =  glob(inDir + os.sep + '*' + '.bin')

    #inFilenames = ['/Volumes/LaCie/data/CASIA2/CASIA_patches_128/train_crop_9.bin',]

    outFilenames = [n.replace('CASIA_patches_128', 'patches_CASIA2_compressed') for n in inFilenames]

    qps = [0,7,14,17,21,25,28,35,42,49]

    for i, inputFilename in enumerate(inFilenames):
        print("Doing {}".format(inputFilename))
        outputFilename = outFilenames[i]
        compressWithConstantQuant(inputFilename, outputFilename, qps=qps, patchWidth=patchWidth, patchHeight=patchHeight,
                                  patchChannels=3, labelSize=labelSize)

