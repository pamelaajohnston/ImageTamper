import numpy as np
import matplotlib.pyplot as plt
import os
import re
import patchIt2 as pi
import functions
#import generateHeatmapFromYUV as gh


def createFileList(srcDir="/Volumes/LaCie/data/yuv_testOnly/CompAndReComp", desiredName=".yuv"):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("yuv"):
                baseFileName, ext = os.path.splitext(filename)
                if desiredName in baseFileName:
                    fileList.append(os.path.join(dirName, filename))
    return fileList

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

# This might work but it is *incredibly* slow and is comparing every patch with every other
# There's no need for that as matches are unlikely across sequences
def doingItFromDatasets():

    datasetMajorPath = "/Volumes/LaCie/data/YUV_80x80_datasets"
    datasetMinorPaths = ["intraForQp0to7_cropDim80_spacStep48_tempStep40",
                    "intraForQp0to14_cropDim80_spacStep48_tempStep40",
                    "intraForQp0to21_cropDim80_spacStep48_tempStep40",
                    "intraForQP_cropDim80_spacStep48_tempStep40",
                    "intra0VsInter1_cropDim80_spacStep48_tempStep40",
                    "deblock1_cropDim80_spacStep48_tempStep40"]

    datasetMinorPaths = ["intraForQp0to7_cropDim80_spacStep48_tempStep40/",]
    testTrain = ["test_", "train_"]

    patchHeight = 80
    patchWidth = 80
    channels = 3
    dataSize = (patchHeight*patchWidth*channels)
    labelSize = 1
    entrySize = dataSize + labelSize

    numLabels = 8
    results = np.zeros((numLabels, numLabels))

    for datasetMinorPath in datasetMinorPaths:
        data_dir = os.path.join(datasetMajorPath, datasetMinorPath)
        print(data_dir)

        # read in all the entries
        filenames = [os.path.join(data_dir, 'test_%d.bin' % i) for i in xrange(0, 10)]

        patchList = []
        #print(patchList)
        for filename in filenames:
            print("Reading {}".format(filename))
            with open(filename, "rb") as f:
                mybytes = np.fromfile(f, 'u1')
                #print(mybytes)
                patchList.extend(mybytes)
                #print(patchList)

        patches_array = np.array(patchList)
        patches_array = patches_array.flatten()
        print("Done reading and rearranging")



        numPatches = patches_array.shape[0] // entrySize
        print("There are {} patches in {}".format(numPatches, data_dir))

        patches = patches_array.reshape((numPatches, entrySize))
        data_labels = patches[:, 0].copy()
        data_array = patches[:, 1:].copy()
        data_labels = np.reshape(data_labels, (numPatches))
        print(data_labels.shape)

        for i in range(0, numPatches):
            row = data_array[i, :]
            label = data_labels[i]

            #print(label)
            matches = np.where((data_array == row).all(axis=1))
            #print("The matches for row {}: {}".format(i, matches))
            labelMatches = data_labels[matches]
            #print("The matches for row {} has labels {}".format(i, data_labels[matches]))

            for m in data_labels[matches]:
                results[label, m] = results[label, m] + 1
            if (i % 10) == 0:
                print("{}".format(results))

        print("Final results matrix for {}, test:".format(data_dir))


if __name__ == "__main__":

    compYuvDir = "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_test"
    seqnames = ["tempete_cif", "bus_cif", "flower_cif", "news_cif", "news_qcif"]
    resultsCSVname = "duplicates_test.csv"
    compYuvDir = "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train"
    seqnames1 = ["akiyo_cif", "akiyo_qcif", "bridge-close_cif", "carphone_qcif", "claire_qcif", "coastguard_cif",
                 "coastguard_qcif", "container_cif", "container_qcif", "foreman_cif", "foreman_qcif", "grandma_qcif",
                 "hall_cif", "hall_qcif", "miss-america_qcif", "mobile_cif",
                "mobile_qcif", "mother-daughter_cif", "mother-daughter_qcif", "paris_cif",
                "salesman_qcif", "silent_qcif", "stefan_cif",
                "suzie_qcif", "waterfall_cif", "bridge-close_qcif", "bridge-far_qcif", "highway_qcif",]
    seqnames = ["bridge-far_cif", "highway_cif",
                "ducks_720p50", "in_to_tree_720p50", "mobcal_720p50", "old_town_cross_720p50",
                "parkrun_720p50", "shields_720p50", "stockholm_720p50", "crowd_run_1080p50", ]
    #seqnames = ["bridge-far_cif",]
    
    halveSeqs = False
    #halveSeqs = True

    resultsCSVname = "duplicates_train.csv"
    #seqnames = ["news_cif",]
    tempPatchDir = "/Volumes/LaCie/data/YUV_temp/patchyPatch"
    tempHalvedDir = "/Volumes/LaCie/data/YUV_temp/halfSeqs"
    cropDim = 80
    cropSpacStep = 48
    cropTempStep = 40
    #cropSpacStep = 16
    #cropTempStep = 1

    # These are constant (I haven't tried varying them)
    num_channels = 3
    bit_depth = 8 # please remember to normalise
    entrySize = (cropDim*cropDim*num_channels) +1

    results = np.zeros((52, 52))

    for seqname in seqnames:
        print(seqname)
        fileList = createFileList(compYuvDir, seqname)
        fileList.sort()

        #fileList = ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_test/quant_0/tempete_cif_q0.yuv",
        #            "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_test/quant_1/tempete_cif_q1.yuv"]

        numLabels = len(fileList)
        print(fileList)

        #results = np.zeros((numLabels, numLabels))
        seqresults = np.zeros((52, 52))

        #pairs = []
        #for i, file in enumerate(fileList):
        #    for j in range((i+1),len(fileList)):
        #        tuple = (file, fileList[j])
        #        pairs.append(tuple)

        #print(pairs)

        patchFiles = []

        if halveSeqs:
            # a bit of jiggery pokery creating intermediate sequences....
            firstHalvesList = []
            secondHalvesList = []
            thirdHalvesList = []
            fourthHalvesList = []

            for filename in fileList:
                infile0 = filename
                p0, b0 = os.path.split(infile0)
                b0, e0 = os.path.splitext(b0)
                outfile0 = "{}_first.yuv".format(b0)
                outfile0 = os.path.join(tempHalvedDir, outfile0)
                outfile1 = "{}_second.yuv".format(b0)
                outfile1 = os.path.join(tempHalvedDir, outfile1)
                outfile2 = "{}_third.yuv".format(b0)
                outfile2 = os.path.join(tempHalvedDir, outfile2)
                outfile3 = "{}_fourth.yuv".format(b0)
                outfile3 = os.path.join(tempHalvedDir, outfile3)

                firstHalvesList.append(outfile0)
                secondHalvesList.append(outfile1)
                thirdHalvesList.append(outfile2)
                fourthHalvesList.append(outfile3)

                width, height = pi.getDimsFromFileName(infile0)
                with open(infile0, "rb") as f:
                    mybytes = np.fromfile(f, 'u1')
                frameSize = width * height * 3 // 2 # magic numbers because it's YUV 420.
                num_frames = len(mybytes) / frameSize
                #print(num_frames)
                #print(num_frames//2)
                #print(mybytes.shape)
                firstHalf = mybytes[0:((num_frames//4)*frameSize)].copy()
                functions.appendToFile(firstHalf, outfile0)
                secondHalf = mybytes[((num_frames//4)*frameSize):((num_frames//2)*frameSize)].copy()
                functions.appendToFile(secondHalf, outfile1)
                thirdHalf = mybytes[((num_frames//2)*frameSize):((num_frames//4)*frameSize*3)].copy()
                functions.appendToFile(thirdHalf, outfile2)
                fourthHalf = mybytes[((num_frames//4)*frameSize*3):].copy()
                functions.appendToFile(fourthHalf, outfile3)

            fileList = firstHalvesList
            print(firstHalvesList)
            print(secondHalvesList)
            print(thirdHalvesList)
            print(fourthHalvesList)
        else:
            for filename in fileList:
                infile0 = filename
                p0, b0 = os.path.split(infile0)
                b0, e0 = os.path.splitext(b0)
                outfile = "{}_first.yuv".format(b0)
                outfile = os.path.join(tempHalvedDir, outfile)



        repeatsRange = range(0, 1)
        if halveSeqs:
            repeatsRange = range(0, 4)

        print(repeatsRange)

        detectDuplicates = False
        for halfSeq in repeatsRange:
            for filename in fileList:
                infile0 = filename

                # sort out the names
                p0, b0 = os.path.split(infile0)
                b0, e0 = os.path.splitext(b0)
                outfile0 = "{}.bin".format(b0)
                outfile0 = os.path.join(tempPatchDir, outfile0)
                print(outfile0)
                patchFiles.append(outfile0)


                if os.path.isfile(outfile0):
                    numPatches0 = os.path.getsize(outfile0)/entrySize
                    print("{} patch file exists with {} patches".format(outfile0, numPatches0))
                else:
                    numPatches0 = pi.patchOneFile(fileIn=infile0, fileOut=outfile0, label="qp_only",
                                                 cropDim=cropDim, cropTempStep=cropTempStep, cropSpacStep=cropSpacStep,
                                                 num_channels=num_channels, bit_depth=bit_depth
                                                 )


                qp0 = getQuantFromFileName(infile0)


            #patchList = []
            # print(patchList)
            print(fileList)
            print(patchFiles)

            patchList = np.concatenate([np.fromfile(f, 'u1') for f in patchFiles])
            #for filename in fileList:
            #    print("Reading {}".format(filename))
            #    with open(filename, "rb") as f:
            #        mybytes = np.fromfile(f, 'u1')
            #        # print(mybytes)
            #        patchList.extend(mybytes)
            #        # print(patchList)

            #patches_array = np.array(patchList)
            patches_array = patchList
            print("The shape of the data {}".format(patchList.shape))

            patches_array = patches_array.flatten()

            numPatches = patches_array.shape[0] // entrySize
            print("Done reading and rearranging")
            print("There are {} patches for {} which is {} per file".format(numPatches, seqname, (numPatches/(len(patchFiles)))))

            patches = patches_array.reshape((numPatches, entrySize))
            data_labels = patches[:, 0].copy()
            data_array = patches[:, 1:].copy()
            data_labels = np.reshape(data_labels, (numPatches))
            #print(data_array.shape)
            #print(data_labels.shape)
            #quit()

            if detectDuplicates:
                for i in range(0, numPatches):
                    row = data_array[i, :]
                    label = data_labels[i]

                    #print(label)
                    matches = np.where((data_array == row).all(axis=1))
                    #print("The matches for row {}: {}".format(i, matches))
                    #print(matches)
                    #print(i)
                    #print(np.where(matches[0] == i))
                    #a = np.array([1,2,3])
                    #print(a)
                    #new_matches = np.delete(matches[0], np.where(matches[0] == [i]))
                    #print("The matches for row {}: {}".format(i, new_matches))
                    labelMatches = data_labels[matches]
                    print("The matches for row {} has labels {}".format(i, data_labels[matches]))

                    for m in data_labels[matches]:
                        seqresults[label, m] = seqresults[label, m] + 1
                    #if (i % 10) == 0:
                    #    print("{}".format(results))

            results = results + seqresults
            print("Final results matrix for seq {}".format(seqname))
            np.set_printoptions(threshold=np.nan)
            print(seqresults)

            resultsName = "duplicates_{}_{}_spac{}_temp{}.csv".format(seqname, halfSeq, cropSpacStep, cropTempStep)
            np.savetxt(resultsName, seqresults, delimiter=",")

            with open("dump.txt", "a") as l:
                l.write("Final results matrix for seq {}\n".format(seqname))
                l.write(seqresults)

            if halveSeqs: # prepare to repeat...
                if halfSeq == 0:
                    fileList = secondHalvesList
                elif halfSeq == 1:
                    fileList = thirdHalvesList
                else:
                    fileList = fourthHalvesList

    print("Final results matrix for all seq")
    np.set_printoptions(threshold=np.nan)
    print(results)
    np.savetxt(resultsCSVname, results, delimiter=",")
    with open("dump.txt", "a") as l:
        l.write("Final results matrix for seq {}\n".format(seqname))
        l.write(seqresults)

    quit()




