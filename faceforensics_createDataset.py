import numpy as np
import sys
import os
import shutil
import shlex
import subprocess
import re
import functions as f


def saveToFile(data, filename):
    datayuv = np.asarray(data, 'u1')
    yuvByteArray = bytearray(datayuv)
    mylen = len(yuvByteArray)
    yuvFile = open(filename, "wb")
    yuvFile.write(yuvByteArray)
    yuvFile.close()

def createFileList(srcDir="/Volumes/LaCie/data/YUV_temp", baseNamesOnly=True):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("avi"):
                n = os.path.join(dirName, filename)
                if baseNamesOnly:
                    p, n = os.path.split(filename)
                fileList.append(n)
    return fileList

def getAVIFileDims(infilename):
    # first get the dimensions so they can go in the file name
    probeCmd = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {}".format(infilename)
    if sys.platform == 'win32':
        args = probeCmd
    else:
        args = shlex.split(probeCmd)

    #subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()
    dims = out.rstrip()
    #print("The dimensions are {}".format(dims))
    return dims


def convertAVItoYUV(infilename):
    dims = getAVIFileDims(infilename)

    outfilename = "temp_{}.yuv".format(dims)
    outfilename = infilename.replace(".avi", "_{}.yuv".format(dims))
    if os.path.isfile(outfilename):
        os.remove(outfilename)


    app = "ffmpeg"
    appargs = "-i {} -pix_fmt yuv420p {}".format(infilename, outfilename)

    exe = app + " " + appargs
    #print exe

    if sys.platform == 'win32':
        args = exe
    else:
        args = shlex.split(exe)

    #subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()

    return outfilename

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
    #if "Davino" in vid or "SULFA" in vid or "VTD" in vid or "DeepFakes" in vid:
    #    width, height, firstTampFrame, interFrameOnly = getFrameDetailsFromFilename(vid)
    #    return width, height

    return 0,0


def createDataset(indexNumer, pathToSource, pathToDst):
    print("creating dataset")

def createFilePairs(sourceTopDir):
    filePairs = []
    #for set in ["test", "train", "val"]:
    for set in ["test"]:
        mainDir = os.path.join(sourceTopDir, set)
        # create related file pairs
        originalDir = os.path.join(mainDir, "original")
        alteredDir = os.path.join(mainDir, "altered")
        alist = createFileList(originalDir)
        for file in alist:
            o = os.path.join(originalDir, file)
            a = os.path.join(alteredDir, file)
            filePairs.append([o,a])
    return filePairs

def addBorder(x, width, align=False, borderSize = 8):
    if align:
        gridSpacing = 16
        r = x % gridSpacing
        if borderSize < 0:
            x = x - r
        else:
            x = x + r
    else:
        x = x + borderSize
    if x < 0:
        x = 0
    if x > (width - 1):
        x = width - 1
    return x


def processFilePair(o, a, tidyUp=False):
    dims = getAVIFileDims(o)
    width, height = getDimsFromFileName(dims)
    channels = 3
    frameSize = (width * height * 3) // 2
    print("Dimensions are {} x {}, frame size {}".format(width, height, frameSize))
    oyuvname = convertAVItoYUV(o)
    ayuvname = convertAVItoYUV(a)
    print("files are {} and {}".format(oyuvname, ayuvname))

    oyuv = np.fromfile(oyuvname, 'u1')
    ayuv = np.fromfile(ayuvname, 'u1')

    #print(oyuv.shape)

    oFrame = oyuv[0:frameSize]
    aFrame = ayuv[0:frameSize]

    odata = f.YUV420_2_YUV444(oFrame, height, width)
    adata = f.YUV420_2_YUV444(aFrame, height, width)

    odata = odata.reshape((channels, height, width))
    adata = adata.reshape((channels, height, width))



    diff = abs(odata - adata)
    #diff420 = f.YUV444_2_YUV420(diff, height, width)
    #f.saveToFile(diff420, "temp.yuv")
    diffY = diff[0:(width*height)]
    diffInds1 = np.nonzero(diff)
    minX = np.amin(diffInds1[2])
    maxX = np.amax(diffInds1[2])
    minY = np.amin(diffInds1[1])
    maxY = np.amax(diffInds1[1])
    print("bounding box ({},{}) to ({},{})".format(minX, minY, maxX, maxY))
    # Adjust the cropped region by adding a border
    alignTo16grid = True
    minX = addBorder(minX, width, alignTo16grid, -8)
    maxX = addBorder(maxX, width, alignTo16grid, 8)
    minY = addBorder(minY, height, alignTo16grid, -8)
    maxY = addBorder(maxY, height, alignTo16grid, 8)
    print("adjusted bounding box ({},{}) to ({},{})".format(minX, minY, maxX, maxY))

    croppedWidth = maxX - minX
    croppedHeight = maxY - minY
    dims = "cropped_{}x{}".format(croppedWidth, croppedHeight)
    odata = odata.reshape((channels, height, width))
    adata = adata.reshape((channels, height, width))
    oROI = odata[:, minY:maxY, minX:maxX]
    aROI = adata[:, minY:maxY, minX:maxX]

    # Now save the crops to a file:
    oCropName = o.replace(".avi", "_{}.yuv".format(dims))
    aCropName = a.replace(".avi", "_{}.yuv".format(dims))
    oROI420 = f.YUV444_2_YUV420(oROI, croppedHeight, croppedWidth)
    aROI420 = f.YUV444_2_YUV420(aROI, croppedHeight, croppedWidth)
    f.saveToFile(oROI420, oCropName)
    f.saveToFile(aROI420, aCropName)
    print("Made files {} and {}".format(oCropName, aCropName))

    if tidyUp:
        os.remove(oyuv, 'u1')
        os.remove(ayuv, 'u1')
    quit()


if __name__ == "__main__":

    sourceTopDir = "/Users/pam/Documents/data/FaceForensics/FaceForensics_compressed/"
    destTopDir = "/Volumes/LaCie/data/FaceForensics/SampleDataset"

    filePairs = createFilePairs(sourceTopDir)
    tidyUp = False
    print(filePairs)

    for pair in filePairs:
        # run through the list and do a diff (look at first frame only).
        o,a = pair
        processFilePair(o, a, tidyUp)



