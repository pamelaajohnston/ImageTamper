import random
import os
import functions
import shlex, subprocess
import numpy as np
import time
import re
import sys

# A function to get the QP for each MB from a bunch of video files and histogram it (or something)

saveFolder = "qpFiles/"

datadir = '/Volumes/LaCie/data/yuv'
#videoFilesBase = 'Data/VID/snippets/train/ILSVRC2017_VID_train_0000/'
#annotFilesBase = 'Annotations/VID/train/ILSVRC2017_VID_train_0000/'
#baseFileName = 'ILSVRC2017_train_00000000'

x264 = "x264"
ldecod = "ldecod"
ffmpeg = "ffmpeg"

from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import re

fileSizes = [
    ['qcif', 176, 144],
    ['512x384', 512, 384],
    ['384x512', 384, 512],
    ['cif', 352, 288],
    ['sif', 352, 240],
    ['720p', 1280, 720],
    ['1080p', 1920, 1080]
]

quantDiv = 6
quants = [0, 6, 12, 18, 24, 30, 36, 42, 48]
quantDiv = 7
quants = [0, 7, 14, 21, 28, 35, 42, 49]
quants = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
quants = [15, 18]
# bitrates as expressed in multiples of pixels (for different file sizes)
bitrates = [0.05, 0.1, 0.2, 0.5, 1.0]
bitrates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
#bitrates = [0.01, 0.02]
testfileNames = ["bus_cif", "flower_cif", "news_cif", "news_qcif", "tempete_cif"]

def getFile_Name_Width_Height(fileName):
    for fileSize in fileSizes:
        if fileSize[0] in fileName:
            return fileName, fileSize[1], fileSize[2]

    dims = re.search(r'([0-9]+)[x|X]([0-9]+)', fileName)
    if dims:
        width = int(dims.group(1))
        height = int(dims.group(2))
        return fileName, width, height

def createFileList(myDir, takeAll = False, format='.yuv'):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(myDir):
        print(filenames)
        for filename in filenames:
            if filename.endswith(format):
                if takeAll or '_cif' in filename:
                    fileName = os.path.join(myDir, dirName, filename)
                    baseFileName, ext = os.path.splitext(filename)
                    print("The filename is {} baseFileName {}".format(fileName, baseFileName))

                    if format=='.yuv':
                        #for fileSize in fileSizes:
                        #    if fileSize[0] in fileName:
                        #        tuple = [fileName, fileSize[1], fileSize[2]]
                        #        fileList.append(tuple)
                        #        break
                        #Untested replacement code
                        f, w, h = getFile_Name_Width_Height(fileName)
                        tuple = [f,w,h]
                        fileList.append(tuple)

                    elif format=='.tif':
                        tuple = [fileName, -1, -1]
                        fileList.append(tuple)

    #hacky hack
    #fileList = [['/Volumes/LaCie/data/yuv_quant_noDeblock/quant_0/mobile_cif_q0.yuv', 352, 288],]
    random.shuffle(fileList)
    #print(fileList)
    return fileList


def compressFile(app, yuvfilename, w, h, qp, outcomp, outdecomp, iFrameFreq=250, deblock=True, intraOnly=False, verbose=False):
    if verbose:
        print("************Compressing the yuv************")
    inputres = '{}x{}'.format(w, h)

    # app = "../x264/x264"
    # if sys.platform == 'win32':
    # app = "..\\x264\\x264.exe"
    # appargs = '-o {} -q {} --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(outcomp, qp, inputres, outdecomp, yuvfilename)
    appargs = '-o {} -q {} --ipratio 1.0 --pbratio 1.0 --no-psy --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(
        outcomp, qp, inputres, outdecomp, yuvfilename)
    if deblock == False:
        appargs = appargs + ' --no-deblock'
    if intraOnly:
        appargs = appargs + ' -I 1'

    ####### WARNING WARNING constant bitrate is totally different!!!) #####################
    if qp > 100:
        # it's bitrate, not qp, set up args accordingly (bitrate in kbps)
        kbps = int((qp + 512) / 1024)
        appargs = '-o {} -B {} --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} -I {} {}'.format(outcomp, kbps,
                                                                                                          inputres,
                                                                                                          outdecomp,
                                                                                                          iFrameFreq,
                                                                                                          yuvfilename)
    # IBBP: 2 b-frames
    # appargs += ' -b 2 --b-adapt 0'

    print appargs

    exe = app + " " + appargs
    # print exe

    if sys.platform == 'win32':
        args = exe
    else:
        args = shlex.split(exe)

    # subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()

    try:
        outlines = out.splitlines()
        iline = outlines[5].split()
        pline = outlines[6].split()
        bline = outlines[7].split()

        isize = iline[-1]
        psize = pline[-1]
        bsize = bline[-1]
    except:
        isize = 0
        psize = 0
        bsize = 0
        print out
    # print ("iframe average size {}".format(isize))

    return isize, psize, bsize

    if verbose:
        print err
        print out


def encodeAWholeFolderAsH264(myDir, takeAll=False, iFrameFreq=250, intraOnly=False, deblock=True, encodedFolder=""):
    fileList = createFileList(myDir, takeAll=takeAll)
    print("The file list:")
    print(fileList)

    for quant in bitrates:
        # make a directory
        dirName = "quant_{}".format(quant)
        dirName = os.path.join(myDir, dirName)
        if encodedFolder=="":
            if not os.path.exists(dirName):
                os.makedirs(dirName)

        for entry in fileList:
            filename = entry[0]
            baseFileName = os.path.basename(filename)
            baseFileName, ext = os.path.splitext(baseFileName)
            baseFileName = "{}/{}".format(dirName, baseFileName)

            #print("The baseFileName is: {}".format(baseFileName))
            width = entry[1]
            height = entry[2]
            actualBitrate = width * height * 30 * quant
            print("width: {} height: {} filename:{}".format(entry[1], entry[2], entry[0]))
            h264Filename = "{}_q{}.h264".format(baseFileName, quant)
            #compYuvFilename = "{}_q{}_intra{}.yuv".format(baseFileName, quant)
            compYuvFilename = "{}_q{}.yuv".format(baseFileName, quant)
            if encodedFolder != "":
                print("Storing in {}".format(encodedFolder))
                d, n = os.path.split(filename)
                n, e = os.path.splitext(n)
                h264Filename = "{}_q{}.h264".format(n, quant)
                h264Filename =  os.path.join(encodedFolder, h264Filename)
                compYuvFilename = "{}_q{}.yuv".format(n, quant)
                compYuvFilename = os.path.join(encodedFolder, compYuvFilename)
            #Do the compression
            isize, psize, bsize = compressFile(x264, filename, width, height, actualBitrate, h264Filename, compYuvFilename,
                                               iFrameFreq=iFrameFreq, intraOnly=intraOnly, deblock=deblock)



def getQuant(inputString):
    print(inputString)
    m = re.search('(?<=quant_)(\d{1,2})', inputString)
    if m:
        quant = int(m.group(0))
        quant = quant/quantDiv
    else:
        print("Didn't find the right quant")
        m = re.search('(?<=_q)(\d{1,2})', inputString)
        quant = int(m.group(0))
        print("But found {}".format(quant))
        quant = quant/quantDiv

    return quant

def extractPatches(fileList, outFileBaseName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3):
    patchesList = []
    numPatches = 0
    filesWritten = 0
    for file in fileList:
        filename = file[0]
        width = file[1]
        height = file[2]
        frameSize = height * width * 3/2 # for i420 only
        quant = getQuant(filename)
        with open(filename, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        numFrames = allTheData.shape[0] / frameSize

        allTheData = allTheData.reshape(numFrames, frameSize)
        frameSample = 0
        lastFrame = numFrames
        if width == 1280:
            frameSample = 2
            lastFrame = numFrames - 2

        while frameSample < lastFrame:
            frameData = allTheData[frameSample]
            ySize = width*height
            uvSize = (width*height)/4
            yData = frameData[0:ySize]
            uData = frameData[ySize:(ySize+uvSize)]
            vData = frameData[(ySize+uvSize):(ySize+uvSize+uvSize)]
            #print("yData shape: {}".format(yData.shape))
            #print("uData shape: {}".format(uData.shape))
            #print("vData shape: {}".format(vData.shape))
            yData = yData.reshape(height, width)
            uData = uData.reshape(height/2, width/2)
            vData = vData.reshape(height/2, width/2)
            pixelSample = 0
            xCo = 0
            yCo = 0
            maxPixelSample = ((height-patchDim) * width) + (width-patchDim)
            #print("maxPixelSample: {}".format(maxPixelSample))
            while yCo < (height - patchDim):
                #print("Taking sample from: ({}, {})".format(xCo, yCo))
                patchY = yData[yCo:(yCo+patchDim), xCo:(xCo+patchDim)]
                patchU = uData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchU = np.repeat(patchU, 2, axis=0)
                patchU = np.repeat(patchU, 2, axis=1)
                patchV = vData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchV = np.repeat(patchV, 2, axis=0)
                patchV = np.repeat(patchV, 2, axis=1)

                #print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
                yuv = np.concatenate((np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
                #print("patch dims: {}".format(yuv.shape))
                yuv = yuv.flatten()
                datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1


                xCo = xCo + patchStride
                if xCo > (width - patchDim):
                    xCo = 0
                    yCo = yCo + patchStride
                pixelSample = (yCo*width) + xCo
                #print("numPatches: {}".format(numPatches))

                if numPatches > 9999:
                    patches_array = np.array(patchesList)
                    print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
                    np.random.shuffle(patches_array)
                    outFileName = "{}_{}.bin".format(outFileBaseName, filesWritten)
                    functions.appendToFile(patches_array, outFileName)
                    filesWritten += 1
                    patchesList = []
                    numPatches = 0

            frameSample = frameSample + frameSampleStep

    patches_array = np.array(patchesList)
    print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
    np.random.shuffle(patches_array)
    outFileName = "{}_{}.bin".format(outFileBaseName, filesWritten)
    functions.appendToFile(patches_array, outFileName)



# This function does the same as the above except it puts all the patches with a single quant value (label) in a single bin file.
# extractPatches() above extracts patches from random files (one whole file at a time) until it has sufficient patches to fill a bin file;
# Then it shuffles them and writes to file. extractPatches_byQuant() doesn't do any shuffling.
def extractPatches_byQuant(fileList, outFileBaseName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3):
    patchesList = []
    numPatches = 0
    filesWritten = 0
    for file in fileList:
        filename = file[0]
        width = file[1]
        height = file[2]
        frameSize = height * width * 3/2 # for i420 only
        quant = getQuant(filename)
        with open(filename, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        numFrames = allTheData.shape[0] / frameSize
        print("There are {} frames".format(numFrames))

        allTheData = allTheData.reshape(numFrames, frameSize)
        frameSample = 0
        lastFrame = numFrames
        if width == 1280:
            frameSample = 2
            lastFrame = numFrames - 2

        while frameSample < lastFrame:
            frameData = allTheData[frameSample]
            ySize = width*height
            uvSize = (width*height)/4
            yData = frameData[0:ySize]
            uData = frameData[ySize:(ySize+uvSize)]
            vData = frameData[(ySize+uvSize):(ySize+uvSize+uvSize)]
            #print("yData shape: {}".format(yData.shape))
            #print("uData shape: {}".format(uData.shape))
            #print("vData shape: {}".format(vData.shape))
            yData = yData.reshape(height, width)
            uData = uData.reshape(height/2, width/2)
            vData = vData.reshape(height/2, width/2)
            pixelSample = 0
            xCo = 0
            yCo = 0
            maxPixelSample = ((height-patchDim) * width) + (width-patchDim)
            #print("maxPixelSample: {}".format(maxPixelSample))
            while yCo < (height - patchDim):
                #print("Taking sample from: ({}, {})".format(xCo, yCo))
                patchY = yData[yCo:(yCo+patchDim), xCo:(xCo+patchDim)]
                patchU = uData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchU = np.repeat(patchU, 2, axis=0)
                patchU = np.repeat(patchU, 2, axis=1)
                patchV = vData[(yCo/2):((yCo+patchDim)/2), (xCo/2):((xCo+patchDim)/2)]
                patchV = np.repeat(patchV, 2, axis=0)
                patchV = np.repeat(patchV, 2, axis=1)

                #print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
                yuv = np.concatenate((np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)), axis=0)
                #print("patch dims: {}".format(yuv.shape))
                yuv = yuv.flatten()
                datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
                datayuv = datayuv.flatten()
                patchesList.append(datayuv)
                numPatches = numPatches + 1


                xCo = xCo + patchStride
                if xCo > (width - patchDim):
                    xCo = 0
                    yCo = yCo + patchStride
                pixelSample = (yCo*width) + xCo
                #print("numPatches: {}".format(numPatches))

                patches_array = np.array(patchesList)
                print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
                outFileName = "{}_{}.bin".format(outFileBaseName, quant)
                functions.appendToFile(patches_array, outFileName)
                filesWritten += 1
                patchesList = []
                numPatches = 0

            frameSample = frameSample + frameSampleStep


def main(argv=None):
    print("Butcher the test files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_test'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_train'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName)

def main_2(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_test'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_train'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 80, patchStride = 80, frameSampleStep = 40, numChannels=3)


def encodeAWholeFolderOfImagesAsSingleH264Frames(myDir):
    fileList = createFileList(myDir, takeAll=True, format='.tif')

    quants = [0,7,14,21,28,35,42,49]
    #quants = [0,49]
    #quants = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    quants = [15, 18]
    for quant in quants:
        # make a directory
        dirName = "quant_{}".format(quant)
        dirName = os.path.join(myDir, dirName)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        for entry in fileList:
            filename = entry[0]
            baseFileName = os.path.basename(filename)
            baseFileName, ext = os.path.splitext(baseFileName)
            baseFileName = "{}/{}".format(dirName, baseFileName)

            width, height, yuvData = functions.convertTifToYUV420(filename)
            tempYUVFilename = "myyuv.yuv"
            functions.saveToFile(yuvData, tempYUVFilename)
            print("width: {} height: {} filename:{}".format(width, height, filename))
            h264Filename = "{}_s{}x{}_q{}.h264".format(baseFileName, width, height, quant)
            compYuvFilename = "{}_s{}x{}_q{}.yuv".format(baseFileName, width, height, quant)
            isize, psize, bsize = functions.compressFile(x264, tempYUVFilename, width, height, quant, h264Filename, compYuvFilename)


def encodeAllOfUCID():
    ucidData = '/Volumes/LaCie/data/UCID/test'
    encodeAWholeFolderOfImagesAsSingleH264Frames(ucidData)
    ucidData = '/Volumes/LaCie/data/UCID/validate'
    encodeAWholeFolderOfImagesAsSingleH264Frames(ucidData)
    ucidData = '/Volumes/LaCie/data/UCID/train'
    encodeAWholeFolderOfImagesAsSingleH264Frames(ucidData)

def encodeVidIntraFrames():
    myDir = '/Volumes/LaCie/data/yuv'
    encodeAWholeFolderAsH264(myDir, takeAll=True, intraOnly=True, deblock=False)

def encodeVid():
    #myDir = '/Volumes/LaCie/data/yuv_testOnly/test'
    myDir = '/Volumes/LaCie/data/yuv_testOnly/1stComp_new'
    encodeAWholeFolderAsH264(myDir, iFrameFreq=25, takeAll=True)

def main_UCID(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/UCID/test'

    fileList = createFileList(startHere, takeAll=True)
    for file in fileList:
        print(file)

    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 80, patchStride = 48, frameSampleStep = 1, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/UCID/train'
    fileList = createFileList(startHere, takeAll=True)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 80, patchStride = 80, frameSampleStep = 1, numChannels=3)

def main_UCID_smallerPatches(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/UCID/test'

    fileList = createFileList(startHere, takeAll=True)
    for file in fileList:
        print(file)

    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 1, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/UCID/train'
    fileList = createFileList(startHere, takeAll=True)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 1, numChannels=3)

def main_UCID_smallerPatches_fewer(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/UCID/test'

    fileList = createFileList(startHere, takeAll=True)
    for file in fileList:
        print(file)

    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 32, patchStride = 80, frameSampleStep = 1, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/UCID/train'
    fileList = createFileList(startHere, takeAll=True)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 32, patchStride = 80, frameSampleStep = 1, numChannels=3)

def allVid_smallerPatches(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_test'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_train'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

def cifVidIntra_smallerPatches(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_test'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_train'

    fileList = createFileList(startHere)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

def allVidIntra_smallerPatches(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_test'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_train'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 32, patchStride = 32, frameSampleStep = 60, numChannels=3)

def allVidIntra_Patches(argv=None):
    print("Butcher the test files (in a slightly different way")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_test'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3)

    print("Butcher the train files")
    startHere = '/Volumes/LaCie/data/yuv_quant_intraOnly_noDeblock_train'

    fileList = createFileList(startHere, takeAll=True)

    for file in fileList:
        print(file)

    #quit()

    #patchesBinFileName = "{}/patches.bin".format(startHere)
    patchesBinFileName = "patches"
    patchArray = extractPatches(fileList, patchesBinFileName, patchDim = 80, patchStride = 80, frameSampleStep = 40, numChannels=3)

def createPatches(argv=None):
    print("Butcher the test files (in a slightly different way")
    #startHere = '/Volumes/LaCie/data/yuv_quant_noDeblock_test_noNews'
    startHere = '/Volumes/LaCie/data/UCID/validate'

    fileList = createFileList(startHere, takeAll=True)
    for file in fileList:
        print(file)

    patchesBinFileName = "patches_test"
    patchArray = extractPatches_byQuant(fileList, patchesBinFileName, patchDim = 80, patchStride = 48, frameSampleStep = 30, numChannels=3)

def addABorder(data, width, height, leftPels, rightPels, topPels, bottomPels):
    newWidth = width + leftPels + rightPels
    newHeight = height + topPels + bottomPels

    pixels = data.reshape(3, height, width)
    y = pixels[0]
    u = pixels[1]
    v = pixels[2]
    #yuvpixels = []
    # First the Y (0 for a black border)
    top = np.full([topPels, newWidth], 0)
    bottom = np.full([bottomPels, newWidth], 0)
    left = np.full([1, leftPels], 0)
    right = np.full([1, rightPels], 0)
    yuvpixels = top
    #print("number of pixels = {}".format(np.size(yuvpixels)))
    for yrow in y:
        row = np.append(left, yrow)
        row = np.append(row, right)
        yuvpixels = np.append(yuvpixels, row)
        #print("row length = {}".format(np.size(row)))
        #print("number of pixels = {}".format(np.size(yuvpixels)))
    yuvpixels = np.append(yuvpixels, bottom)

    #print("number of pixels = {}".format(np.size(yuvpixels)))

    # Now the U and V (128 for a black border)
    top = np.full([topPels, newWidth], 128)
    bottom = np.full([topPels, newWidth], 128)
    left = np.full([1, leftPels], 128)
    right = np.full([1, rightPels], 128)
    yuvpixels = np.append(yuvpixels, top)
    for urow in u:
        row = np.append(left, urow)
        row = np.append(row, right)
        yuvpixels = np.append(yuvpixels, row)
    yuvpixels = np.append(yuvpixels, bottom)
    # And V
    yuvpixels = np.append(yuvpixels, top)
    for vrow in v:
        row = np.append(left, vrow)
        row = np.append(row, right)
        yuvpixels = np.append(yuvpixels, row)
    yuvpixels = np.append(yuvpixels, bottom)



    return yuvpixels, newWidth, newHeight

def prepareDataForHeatMapGeneration():
    print("Preparing a bin file that's suitable for a heat map")
    patchDim = 80
    patchStride = 16 # for macroblock-ness
    fileList = [['/Users/pam/Documents/data/yuv/flower_1f_q0.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q7.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q14.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q21.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q28.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q35.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q42.yuv', 352, 288],
                ['/Users/pam/Documents/data/yuv/flower_1f_q49.yuv', 352, 288]]
    fileList = [['/Users/pam/Documents/data/SULFA/forged_sequences_avi/01_forged_1f.yuv', 320, 240],
                ['/Users/pam/Documents/data/SULFA/forged_sequences_avi/01_original_1f.yuv', 320, 240]]
    fileList = [['/Users/pam/Documents/data/DeepFakes/creepy2_1f.yuv', 1280, 720]]
    for file in fileList:
        filename = file[0]
        fileBaseName, ext = os.path.splitext(filename)
        width = file[1]
        height = file[2]
        frameSize = height * width * 3/2 # for i420 only
        try:
            quant = getQuant(filename)
        except:
            quant = 0
        print("The quant for {} is {}".format(quant, filename))
        with open(filename, "rb") as f:
            allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        numFrames = allTheData.shape[0] / frameSize
        print("There's {} frames".format(numFrames))

        frameData = allTheData
        frame444 = functions.YUV420_2_YUV444(frameData, height, width)
        frame444, newWidth, newHeight = addABorder(frame444, width, height, 32, 32, 32, 32)
        outFileName = "{}.YUV444_{}x{}".format(fileBaseName, newWidth, newHeight)
        functions.appendToFile(frame444, outFileName)

        # And now the generation of patches
        patchesList = []
        numPatches = 0
        filesWritten = 0
        frameData = frame444
        ySize = newWidth * newHeight
        uvSize = newWidth * newHeight
        yData = frameData[0:ySize]
        uData = frameData[ySize:(ySize + uvSize)]
        vData = frameData[(ySize + uvSize):(ySize + uvSize + uvSize)]
        # print("yData shape: {}".format(yData.shape))
        # print("uData shape: {}".format(uData.shape))
        # print("vData shape: {}".format(vData.shape))
        yData = yData.reshape(newHeight, newWidth)
        uData = uData.reshape(newHeight, newWidth)
        vData = vData.reshape(newHeight, newWidth)
        pixelSample = 0
        xCo = 0
        yCo = 0
        maxPixelSample = ((newHeight - patchDim) * newWidth) + (newWidth - patchDim)
        # print("maxPixelSample: {}".format(maxPixelSample))
        while yCo <= (newHeight - patchDim):
            # print("Taking sample from: ({}, {})".format(xCo, yCo))
            patchY = yData[yCo:(yCo + patchDim), xCo:(xCo + patchDim)]
            patchU = uData[yCo:(yCo + patchDim), xCo:(xCo + patchDim)]
            patchV = vData[yCo:(yCo + patchDim), xCo:(xCo + patchDim)]

            # print("patch dims: y {} u {} v {}".format(patchY.shape, patchU.shape, patchV.shape))
            yuv = np.concatenate(
                (np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)),
                axis=0)
            # print("patch dims: {}".format(yuv.shape))
            yuv = yuv.flatten()
            datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
            datayuv = datayuv.flatten()
            patchesList.append(datayuv)
            numPatches = numPatches + 1

            xCo = xCo + patchStride
            if xCo > (newWidth - patchDim):
                xCo = 0
                yCo = yCo + patchStride
            pixelSample = (yCo * newWidth) + xCo

            patches_array = np.array(patchesList)
            print("Dims: {}, numPatches {}".format(patches_array.shape, numPatches))
            ############## Here's where you name the files!!!!###########
            outFileName = "{}.bin".format(fileBaseName)
            outFileName = "patches_test_{}.bin".format(quant)
            functions.appendToFile(patches_array, outFileName)
            patchesList = []

        #Add more "patches" so that we have blank patches up to a multiple of 128 (batch size)
        batchPatchNum = (np.ceil(numPatches/128.0) * 128) - numPatches
        print("Adding a further {} patches to round to batches".format(batchPatchNum))
        patchY = np.full([patchDim, patchDim], 0)
        patchU = np.full([patchDim, patchDim], 128)
        patchV = np.full([patchDim, patchDim], 128)
        while batchPatchNum > 0:
            yuv = np.concatenate(
                (np.divide(patchY.flatten(), 8), np.divide(patchU.flatten(), 8), np.divide(patchV.flatten(), 8)),
                axis=0)
            # print("patch dims: {}".format(yuv.shape))
            yuv = yuv.flatten()
            datayuv = np.concatenate((np.array([quant]), yuv), axis=0)
            datayuv = datayuv.flatten()
            patchesList.append(datayuv)
            patches_array = np.array(patchesList)
            functions.appendToFile(patches_array, outFileName)
            patchesList = []
            batchPatchNum = batchPatchNum - 1

def forKyle():
    path = "/Volumes/LaCie/data/CIFAR-10/cifar_data_constantQuant/cifar-10-batches-bin_yuv"
    myfiles = ['data_batch_1',]
    width = 32
    height = 32
    channels = 3

    for myfile in myfiles:
        thefile = os.path.join(path, myfile)
        thefile = '{}.bin'.format(thefile)
        f = open(thefile, 'rb')
        allTheData = np.fromfile(f, 'u1')
        print(allTheData.shape)
        recordSize = (width * height * channels) + 1
        num_cases_per_batch = allTheData.shape[0] / recordSize
        allTheData = allTheData.reshape(num_cases_per_batch, recordSize)
        indexes = np.argsort(allTheData[:, 0])

        sortedData = allTheData[indexes]
        print("sortedData: {}".format(sortedData[0]))
        print("sortedData: {}".format(sortedData[1]))
        print("sortedData: {}".format(sortedData[2]))
        print("...")
        print("sortedData: {}".format(sortedData[-3]))
        print("sortedData: {}".format(sortedData[-2]))
        print("sortedData: {}".format(sortedData[-1]))
        #for idx, id in enumerate(indexes):
        #    mapFile.write("{} {} {} {} \n".format(fileName, id, "newfile", idx))

        sarrays = np.split(sortedData, np.where(np.diff(sortedData[:, 0]))[0] + 1)
        print("Now we have {} arrays".format(len(sarrays)))
        leftData = []
        leftData = np.array(leftData)
        #takenData = []

        for idx, myArray in enumerate(sarrays):
            if idx == 0 or idx == 8:
                print("The size of the array: {}".format(myArray.size))
                print(myArray)
                #leftData.append(myArray[0:, 0:])
                leftData = np.append(leftData, myArray)
            #else:
            #    leftData.append(myArray[takeNum:, :])

        leftData = np.asarray(leftData, 'u1')
        leftData = leftData.flatten()
        #print("The size of the selected labels {}".format(leftData.size))
        #leftData = leftData.reshape(leftData.shape[0])
        print("The left data with size: {}".format(leftData.size))
        print(leftData)
        #leftData = np.asarray(leftData, 'u1')
        leftData = bytearray(leftData)
        leftName = '{}_onlyLabels0and8.bin'.format(myfile)
        with open(leftName, 'wb') as f:
            f.write(leftData)


def processMyldecodOP(filename):
    #filename = "/Users/pam/Documents/data/Parkinsons/op2.txt"
    intras = []
    skippeds = []
    mvs = []
    qps = []
    frameCount = 0

    with open(filename, "r") as f:
        for line in f:
            print(line)
            line.strip()
            line = line.replace("\n", "")
            line = line.split(';')
            print(line)
            if "Frame" not in line[0]:
                continue
            for terms in line:
                terms = terms.strip()
                terms = (terms).split(':')
                print("First term: {}".format(terms[0]))
                if terms[0] == "qp":
                    qps.append(terms[1])
                elif terms[0] == "intra":
                    intras.append(terms[1])
                elif terms[0] == "skipped":
                    skippeds.append(terms[1])
                elif terms[0] == "motion vectors":
                    mvs.append(terms[1])
                elif terms[0] == "Frame":
                    frameNo = terms[1]
                elif terms[0] == "MB no":
                    print("This is macroblock number {}".format(terms[1]))
                    if terms[1] == 0:
                        frameCount = frameCount + 1

    print(qps)

def createFileList2(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def jpgsToCsv(dir):
    from PIL import Image
    import numpy as np
    import sys
    import os
    import csv

    #dir = "/Users/pam/Documents/data/pictures/"
    fileList = createFileList2(dir)

    for file in fileList:
        print(file)
        img_file = Image.open(file)
        # img_file.show()

        # get original image parameters...
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Make image Greyscale
        img_grey = img_file.convert('L')
        #img_grey.save('result.png')
        #img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        print(value)
        with open("img_pixels.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)

if __name__ == "__main__":
    #main_2()

    # For UCID - all of these will be intra frames!
    #encodeAllOfUCID()
    #main_UCID()
    #createPatches()

    # a smaller patches dataset
    #main_UCID_smallerPatches()
    #allVid_smallerPatches()

    # Trying out an intra-only dataset
    #encodeVidIntraFrames()
    encodeVid()
    #cifVidIntra_smallerPatches()
    #allVidIntra_smallerPatches()
    #allVidIntra_Patches()
    #main_UCID_smallerPatches_fewer()

    #prepareDataForHeatMapGeneration()

    #forKyle()

