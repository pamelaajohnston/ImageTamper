# A quick and nasty program to turn .bin files into .csv (and maybe remove some channels

import numpy as np
import csv as csv

if __name__ == "__main__":

    mydir =

    inputFilename = "test_crop_unshuffled.bin"
    outputFilename = "test_crop_unshuffled_u.bin"

    patchWidth = 256
    patchHeight = 256
    patchChannels = 3

    labelSize = 1
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
