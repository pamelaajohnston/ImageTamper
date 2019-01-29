import numpy as np
import sys
import os
import shutil
import functions

def createFileList(srcDir="/Volumes/LaCie/data/YUV_temp", desiredNames=["tempete_cif"]):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("bin"):
                baseFileName, ext = os.path.splitext(filename)
                for desiredName in desiredNames:
                    if desiredName in baseFileName:
                        fileList.append(os.path.join(dirName, filename))
    return fileList

def createDataset(indexNumer, pathToSource, pathToDst):
    print("Creating dataset")

    if os.path.isdir(pathToDst):
        shutil.rmtree(pathToDst)
    os.mkdir(pathToDst)

    patchDim = 80
    numFiles = 10

    sourceList = [["bus_cif", "flower_cif", "news_cif", "news_qcif", "tempete_cif"],
                 ["akiyo_cif", "akiyo_qcif", "bridge-close_cif", "bridge-close_qcif",],
                 ["coastguard_cif", "coastguard_qcif", "container_cif", "container_qcif",],
                 ["mother-daughter_cif", "mother-daughter_qcif", "paris_cif", ],
                 ["salesman_qcif", "silent_qcif", "suzie_qcif", "waterfall_cif",],
                 ["highway_qcif", "highway_cif", "mobile_cif", "mobile_qcif",],
                 ["bridge-far_qcif", "bridge-far_cif", "foreman_cif", "foreman_qcif",],
                 ["hall_cif", "hall_qcif", "carphone_qcif",],
                 ["grandma_qcif", "miss-america_qcif","claire_qcif",],
                 ["tempete_cif", "bus_cif", "flower_cif","stefan_cif", ]]





    testFiles = []
    trainFiles = []
    for i, list in enumerate(sourceList):
        if i == indexNumer:
            testFiles.extend(list)
        else:
            trainFiles.extend(list)


    allTestFiles = createFileList(pathToSource, testFiles)
    allTrainFiles = createFileList(pathToSource, trainFiles)

    #print("Test files:")
    #print(allTestFiles)
    #print("And the training files:")
    #print(allTrainFiles)


    print("TEST DATA")
    patchList = []
    # print(patchList)
    shuffled = True
    for filename in allTestFiles:
        print("Reading {}".format(filename))
        with open(filename, "rb") as f:
            mybytes = np.fromfile(f, 'u1')
            patchList.extend(mybytes)

    patches_array = np.array(patchList)
    patches_array = patches_array.flatten()
    entrySize = ((patchDim*patchDim*3)+1)
    numPatches = patches_array.shape[0]//entrySize
    print("Total test patches: {}".format(numPatches))
    patches = patches_array.reshape((numPatches, entrySize))

    print("Normalise labels")
    patches[:, 0] = patches[:, 0]/2

    print("Randomise the patches")
    if shuffled:
        np.random.shuffle(patches)

    # Writing out to file
    outfiles = [os.path.join(pathToDst, 'test_%d.bin' % i) for i in xrange(0, numFiles)]

    patchesPerFile = numPatches//numFiles
    print(outfiles)

    for i, file in enumerate(outfiles):
        if os.path.exists(file):
            os.remove(file)
        start = i*patchesPerFile
        end = start + patchesPerFile
        data = patches[start:end,:]
        data = data.flatten()
        functions.saveToFile(data, file)

    numTestPatches = numPatches

    data = []
    outfiles = []
    patches = []
    patches_array = []
    mybytes = []








    #quit()
    numTrainFiles = len(allTrainFiles)
    trainFiles1 = allTrainFiles[:len(allTrainFiles)//2]
    print(trainFiles1)
    trainFiles2 = allTrainFiles[len(allTrainFiles)//2:]
    print("TRAIN DATA 1")
    patchList = []
    # print(patchList)
    shuffled = True
    entrySize = ((patchDim * patchDim * 3) + 1)
    for filename in trainFiles1:
        print("Reading {}".format(filename))
        with open(filename, "rb") as f:
            mybytes = np.fromfile(f, 'u1')
            numPatches = mybytes.shape[0] // entrySize
            #mybytes = mybytes.reshape((numPatches, entrySize))
            patchList.extend(mybytes)
            #print(len(patchList))
            #quit()

    print("Read all the files and there are {} bytes, now for the numpy stuff".format(len(patchList)))

    patches_array = np.array(patchList)
    print("Turned it into an array")
    patches_array = patches_array.flatten()
    print("And now the list is a numpy array")
    patchList = []
    numPatches = patches_array.shape[0]//entrySize
    numTrainPatches = numPatches
    print("Total train patches: {}".format(numPatches))
    patches = patches_array.reshape((numPatches, entrySize))

    print("Normalise labels")
    patches[:, 0] = patches[:, 0]/2

    print("Randomise the patches")
    if shuffled:
        np.random.shuffle(patches)

    # Writing out to file
    outfiles = [os.path.join(pathToDst, 'train_%d.bin' % i) for i in xrange(0, numFiles//2)]
    patchesPerFile = numPatches//numFiles

    for i, file in enumerate(outfiles):
        if os.path.exists(file):
            os.remove(file)
        start = i*patchesPerFile
        end = start + patchesPerFile
        data = patches[start:end,:]
        data = data.flatten()
        functions.saveToFile(data, file)









    print("TRAIN DATA 2")
    patchList = []
    # print(patchList)
    shuffled = True
    for filename in trainFiles2:
        print("Reading {}".format(filename))
        with open(filename, "rb") as f:
            mybytes = np.fromfile(f, 'u1')
            patchList.extend(mybytes)

    print("Read all the files, now for the numpy stuff")

    patches_array = np.array(patchList)
    patches_array = patches_array.flatten()
    entrySize = ((patchDim*patchDim*3)+1)
    numPatches = patches_array.shape[0]//entrySize
    numTrainPatches = numTrainPatches + numPatches
    print("Total train patches: {}".format(numPatches))
    patches = patches_array.reshape((numPatches, entrySize))

    print("Normalise labels")
    patches[:, 0] = patches[:, 0]/2

    print("Randomise the patches")
    if shuffled:
        np.random.shuffle(patches)

    # Writing out to file
    outfiles = [os.path.join(pathToDst, 'train_%d.bin' % i) for i in xrange(numFiles//2, numFiles)]
    patchesPerFile = numPatches//numFiles

    for i, file in enumerate(outfiles):
        if os.path.exists(file):
            os.remove(file)
        start = i*patchesPerFile
        end = start + patchesPerFile
        data = patches[start:end,:]
        data = data.flatten()
        functions.saveToFile(data, file)
    return numTestPatches, numTrainPatches

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("First argument is the index into the source list")
        print("Second argument is the folder where the bin files are")
        print("Third argument is the folder where we're going to store the created dataset")
        print("Try again...")
        quit()

    indexNumer = int(sys.argv[1])
    pathToSource = str(sys.argv[2])
    pathToDst = str(sys.argv[3])

    print("Index number {}".format(indexNumer))

    createDataset(indexNumer, pathToSource, pathToDst)







