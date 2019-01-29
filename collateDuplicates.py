import numpy as np
import matplotlib.pyplot as plt
import os
import re
import patchIt2 as pi
import functions

def createFileList(srcDir="/Users/pam/Documents/dev/git/ImageTamper/", desiredName="duplicate"):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("csv"):
                baseFileName, ext = os.path.splitext(filename)
                if desiredName in baseFileName:
                    fileList.append(os.path.join(dirName, filename))
    return fileList

if __name__ == "__main__":
    print("Collating duplicate files, all .csv files and all with duplicates in the title")
    fileList = createFileList()
    print(fileList)

    seqresults = np.zeros((52, 52))

    for file in fileList:
        print(file)
        my_data = np.genfromtxt(file, delimiter=',')
        seqresults = seqresults + my_data
        np.savetxt("collatedDuplicates.csv", seqresults, delimiter=",", fmt="%d")

        print(seqresults)