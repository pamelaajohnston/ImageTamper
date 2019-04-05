import numpy as np
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
#from keras.layers.core import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import tensorflow as tf
import keras
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
import csv
import re


def pickOutKeyFrames(filenames, predsWidth, predsHeight, numPatches, numFrames):
    print("Picking out key frames")
    patchFrame = predsHeight * predsWidth

    keyFrames = [0,]
    for i, filename in enumerate(filenames):
        predVals = np.loadtxt(filename)
        predVals = predVals[0:numPatches]
        predVals = (predVals / np.linalg.norm(predVals))*100 # normalising
        predVals = predVals.reshape(numFrames,patchFrame)

        # discard the last frame because it fades to black and is nonsense
        predVals = np.append(predVals[0:(numFrames-2), :], predVals[(numFrames-4):(numFrames-2), :]).reshape(numFrames,patchFrame)

        avgs = predVals.mean(axis=1)*1000

        if i == 0:
            faTotals = avgs
        else:
            faTotals = np.multiply(avgs, faTotals)

    mean = np.mean(faTotals)
    stdDev = np.std(faTotals)
    mylist = np.where(abs(faTotals) > (mean + 2*stdDev))



    keyFrames = keyFrames + (mylist[0].tolist())
    # Remove "pairs" (because working on deltas means you detect I->P and sometimes P->I?
    remove_indices = []
    for i, k in enumerate(keyFrames[1:]):
        if keyFrames[i-1] == (keyFrames[i] - 1):
            keyFrames[i] = 0



    keyFrames = np.asarray(keyFrames)
    #print(keyFrames)
    keyFrames = keyFrames.flatten()
    #print(keyFrames)
    keyFrames = np.unique(keyFrames)
    return keyFrames



def predictNumPatches(fileSize, cropDim, tempStep, spacStep, height, width):
    frameSize = width * height * 3 // 2
    num_frames = fileSize // frameSize
    patchedFrames = num_frames // tempStep
    patchWidth = (width - cropDim)//spacStep
    patchHeight = (height - cropDim)//spacStep

    # This feels like a hack...
    if (width % 16) != 0:
        patchWidth = patchWidth + 1
    if (height % 16) != 0:
        patchHeight = patchHeight + 1

    numPatches = patchedFrames * patchWidth * patchHeight
    return patchWidth, patchHeight, patchedFrames, numPatches

# Take the data contained in the dataFiles in the dataPath and munge them together into an appropriate numpy array
# return enough stuff that you can use the data array with the labels and
def prepData(dataPath, yuvFile, width, height, cropDim, cropTempStep, cropSpacStep,
             dataFiles, keyFramesOnly, addBorders, predFramesOnly):
    fileSize = os.path.getsize(yuvFile)
    numFrames = fileSize // (width * height * 3 / 2)

    patchWidth, patchHeight, patchedFrames, numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                                                                           tempStep=cropTempStep, spacStep=cropSpacStep,
                                                                           height=height, width=width)

    keyFrameFiles = ["qpPred.csv", "diffs.csv"]
    labelFile = "gt.csv"

    # First construct my data numpy array
    fullDataFiles = [os.path.join(dataPath, s) for s in keyFrameFiles]
    print(fullDataFiles)

    keyFrames = pickOutKeyFrames(fullDataFiles, patchWidth, patchHeight, numPatches, patchedFrames)
    print(keyFrames)
    allFrameNums = range(0,numFrames)
    predFrames = list(set(allFrameNums) - set(keyFrames))
    print(predFrames)
    if predFramesOnly:
        keyFrames = predFrames

    dataList = []

    for file in dataFiles:
        filename = os.path.join(dataPath, file)
        print(filename)
        my_data = np.genfromtxt(filename, delimiter=',')
        my_data = my_data.flatten()
        my_data = my_data[0:numPatches]

        filename = os.path.join(dataPath, labelFile)
        my_labels = np.genfromtxt(filename, delimiter=',')
        my_labels = my_labels.flatten()
        my_labels = my_labels[0:numPatches]

        # labels = labels.reshape((patchedFrames, patchHeight, patchWidth, 1))
        print("The data from the cvs file: {} the labels: {} and numFrames {}".format(my_data.shape, my_labels.shape, numFrames))
        if addBorders:
            predsHeight = patchHeight
            predsWidth = patchWidth

            normPredVals = my_data.reshape((patchedFrames, predsHeight, predsWidth))
            firstCol = normPredVals[:, :, 0].reshape((patchedFrames, predsHeight, 1))
            lastCol = normPredVals[:, :, (predsWidth - 1)].reshape((patchedFrames, predsHeight, 1))
            firstRow = normPredVals[:, 0, :].reshape((patchedFrames, 1, predsWidth))
            lastRow = normPredVals[:, (predsHeight - 1), :].reshape((patchedFrames, 1, predsWidth))

            my_data_left = np.append(firstCol, normPredVals[:, :, :(predsWidth - 1)], axis=2).flatten()
            my_data_right = np.append(normPredVals[:, :, 1:], lastCol, axis=2).flatten()
            my_data_top = np.append(firstRow, normPredVals[:, :(predsHeight - 1), :], axis=1).flatten()
            my_data_bottom = np.append(normPredVals[:, 1:, :], lastRow, axis=1).flatten()

            my_data = np.append(my_data, my_data_left)
            my_data = np.append(my_data, my_data_right)
            my_data = np.append(my_data, my_data_top)
            my_data = np.append(my_data, my_data_bottom)

            my_data = my_data.reshape((5, numPatches))


        else:
            my_data = my_data.reshape((1, my_data.shape[0]))
        #labels = labels.reshape(1, numPatches)

        #print(my_data.shape)
        print("Before key frames")
        print(my_data.shape)
        print(my_labels.shape)

        if keyFramesOnly:
            reducedSetData = []
            reducedSetLabels = []
            patchFrameSize = patchWidth * patchHeight
            for frame in keyFrames:
                start = frame * patchFrameSize
                end = start + patchFrameSize

                for i in range(0, my_data.shape[0]):
                    keyframe_data = my_data[i, start:end].flatten()
                    keyframe_data = keyframe_data.tolist()
                    reducedSetData.extend(keyframe_data)
                    #print("Data length {}".format(len(keyframe_data)))

                keyframe_labels = my_labels[start:end].flatten()
                keyframe_labels = keyframe_labels.tolist()
                reducedSetLabels.extend(keyframe_labels)
                #print("Label length {}".format(len(keyframe_labels)))
            my_data = np.asarray(reducedSetData)
            my_labels = np.asarray(reducedSetLabels)
            print("After key frames")
            print(my_data.shape)
            print(my_labels.shape)
            numKeyFrames = len(keyFrames)

        #print(my_data.shape)
        my_data = my_data.tolist()
        dataList.extend(my_data)
    if keyFramesOnly:
        numFrames = numKeyFrames

    numFeatures = len(dataFiles)
    if addBorders:
        numFeatures = len(dataFiles) * 5  # Because centre plus 4 borders (maybe 8 would work better?).

    numPatches = numFrames * patchHeight * patchWidth

    my_data = np.asarray(dataList)
    my_data = my_data.reshape((numFeatures, numPatches))
    my_data = np.swapaxes(my_data, 0, 1)
    # my_data = np.swapaxes(my_data, 1, 2)
    # my_data = np.swapaxes(my_data, 2, 3)

    #my_labels = my_labels.reshape((numPatches))

    labels = my_labels
    print(my_labels.shape)

    print(my_data.shape)
    data = my_data


    # Analyse the data a bit in here: What is the data balance?
    unique, counts = np.unique(labels, return_counts=True)
    print("How unbalanced: {}".format(dict(zip(unique, counts))))
    labels = labels.reshape((labels.shape[0], 1))
    all_data = np.append(labels, data, axis=1)
    print(all_data.shape)
    return data, labels, numFeatures



def getBalanceCode(balanceType):
    #["None", "SMOTE", "RandomOver", "RandomUnder"]
    code = ""
    if balanceType == "None":
        code = '-'
    if balanceType == "SMOTE":
        code = 's'
    if balanceType == "RandomOver":
        code = 'ro'
    if balanceType == "RandomUnder":
        code = 'ru'
    return code

def balanceData(balanceType, trainX, trainY):
    # ["None", "SMOTE", "RandomOver", "RandomUnder"]
    if balanceType == "None":
        x_train_res = trainX
        y_train_res = trainY
    if balanceType == "SMOTE":
        sm = SMOTE(random_state=12, ratio=1.0)
        x_train_res, y_train_res = sm.fit_sample(trainX, trainY)
    if balanceType == "RandomOver":
        sm = RandomOverSampler(random_state=12)
        x_train_res, y_train_res = sm.fit_sample(trainX, trainY)
    if balanceType == "RandomUnder":
        sm = RandomUnderSampler(random_state=12)
        x_train_res, y_train_res = sm.fit_sample(trainX, trainY)

    return x_train_res, y_train_res

def doNeuralNetAnalysis(trainX, trainY, testX, testY, numFeatures):
    print("Doing analysis with fully connected neural network")
    train_Y = np_utils.to_categorical(trainY)
    test_Y = np_utils.to_categorical(testY)
    inputShape = (numFeatures,)
    chanDim = 1
    num_classes = 2

    model = Sequential()
    model.add(Dense(32, input_shape=inputShape))
    model.add(Activation("sigmoid"))
    model.add(Dense(32))
    model.add(Activation("sigmoid"))
    model.add(Dense(num_classes))
    model.add(Activation(tf.nn.softmax))

    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.01
    EPOCHS = 4

    # compile the model using SGD as our optimizer and categorical
    # cross-entropy loss (you'll want to use binary_crossentropy
    # for 2-class classification)
    print("[INFO] training network...")
    # opt = SGD(lr=INIT_LR)
    # model.compile(loss="categorical_crossentropy", optimizer=opt,
    #              metrics=["accuracy"])

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # opt = keras.optimizers.Adam()
    # opt = RMSprop(0.001)

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy',
    #              optimizer=opt,
    #              metrics=[sensitivity, specificity])

    class_weight1 = {0: 1.,
                     1: 20.}
    from sklearn.utils import class_weight

    y_ints = [y.argmax() for y in train_Y]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_ints),
                                                      y_ints)
    print(class_weights)

    # train the neural network
    model.fit(trainX, train_Y, validation_data=(testX, test_Y), epochs=EPOCHS, batch_size=32,
                  class_weight=class_weights, verbose=0)

    # evaluate the network
    print("[INFO] evaluating network...")
    preds = model.predict(testX, verbose=0)
    predictions = preds.argmax(axis=1)
    #print(classification_report(testY, predictions, target_names=["0", "1"]))
    #print(confusion_matrix(testY, predictions))
    return predictions

def getFeatureCode(dataFiles, addBorders):
    #["qpPred.csv", "deblockPred.csv", "ipPred.csv", "diffs.csv"]
    code = ''
    if addBorders:
        code = 'b'
    for file in dataFiles:
        if file == "qpPred.csv":
            code = "q_{}".format(code)
        if file == "deblockPred.csv":
            code = "k_{}".format(code)
        if file == "ipPred.csv":
            code = "i_{}".format(code)
        if file == "diffs.csv":
            code = "d_{}".format(code)
    return code


def getClassifierCode(classifierType):
    #["ANN", "SVM", "Linear Regression", "Decision Tree", "Random Forest", "Naive Bayes"]
    code = ""
    if classifierType == "ANN":
        return "ann"
    if classifierType == "SVM":
        return "svm"
    if classifierType == "Decision Tree":
        return "dt"
    if classifierType == "Random Forest":
        return "rf"
    if classifierType == "Naive Bayes":
        return "NB"
    if classifierType == "Linear Regression":
        return "lr"

    return code


allTheFilesDavino = [["/Users/pam/Documents/results/Davino/tree/", "/Users/pam/Documents/data/Davino_yuv/08_TREE_f.yuv"],
               ["/Users/pam/Documents/results/Davino/cat/", "/Users/pam/Documents/data/Davino_yuv/03_CAT_f.yuv"],
               ["/Users/pam/Documents/results/Davino/dog/", "/Users/pam/Documents/data/Davino_yuv/10_DOG_f.yuv"],
               ["/Users/pam/Documents/results/Davino/girl/", "/Users/pam/Documents/data/Davino_yuv/09_GIRL_f.yuv"],
               ["/Users/pam/Documents/results/Davino/helicopter/", "/Users/pam/Documents/data/Davino_yuv/04_HELICOPTER_f.yuv"],
               ["/Users/pam/Documents/results/Davino/hen/", "/Users/pam/Documents/data/Davino_yuv/05_HEN_f.yuv"],
               ["/Users/pam/Documents/results/Davino/lion/", "/Users/pam/Documents/data/Davino_yuv/06_LION_f.yuv"],
               ["/Users/pam/Documents/results/Davino/man/", "/Users/pam/Documents/data/Davino_yuv/02_MAN_f.yuv"],
               ["/Users/pam/Documents/results/Davino/tank/", "/Users/pam/Documents/data/Davino_yuv/01_TANK_f.yuv"],
               ["/Users/pam/Documents/results/Davino/ufo/", "/Users/pam/Documents/data/Davino_yuv/07_UFO_f.yuv"],

]

someTheFilesDavino = [["/Users/pam/Documents/results/Davino/tank/", "/Users/pam/Documents/data/Davino_yuv/01_TANK_f.yuv"],
               ["/Users/pam/Documents/results/Davino/cat/", "/Users/pam/Documents/data/Davino_yuv/03_CAT_f.yuv"],
               ["/Users/pam/Documents/results/Davino/dog/", "/Users/pam/Documents/data/Davino_yuv/10_DOG_f.yuv"],
               ["/Users/pam/Documents/results/Davino/girl/", "/Users/pam/Documents/data/Davino_yuv/09_GIRL_f.yuv"],
               ["/Users/pam/Documents/results/Davino/helicopter/", "/Users/pam/Documents/data/Davino_yuv/04_HELICOPTER_f.yuv"],
               ["/Users/pam/Documents/results/Davino/hen/", "/Users/pam/Documents/data/Davino_yuv/05_HEN_f.yuv"],
               ["/Users/pam/Documents/results/Davino/lion/", "/Users/pam/Documents/data/Davino_yuv/06_LION_f.yuv"],
               ["/Users/pam/Documents/results/Davino/man/", "/Users/pam/Documents/data/Davino_yuv/02_MAN_f.yuv"],

               ["/Users/pam/Documents/results/Davino/ufo/", "/Users/pam/Documents/data/Davino_yuv/07_UFO_f.yuv"],

]

def createFileList(srcDir="/Volumes/LaCie/data/YUV_temp", baseNamesOnly=True):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("csv"):
                n = os.path.join(dirName, filename)
                if baseNamesOnly:
                    p, n = os.path.split(filename)
                fileList.append(n)
    return fileList

def predictNumPatches(cropDim, spacStep, height, width):
    num_frames = 1
    patchedFrames = num_frames
    patchWidth = (width - cropDim)//spacStep
    patchHeight = (height - cropDim)//spacStep

    # This feels like a hack...
    if (width % 16) != 0:
        patchWidth = patchWidth + 1
    if (height % 16) != 0:
        patchHeight = patchHeight + 1

    numPatches = patchedFrames * patchWidth * patchHeight
    return numPatches


def calculateNumPatches(vid):
    dims = re.search(r'([0-9]+)[x|X]([0-9]+)', vid)
    if dims:
        width = int(dims.group(1))
        height = int(dims.group(2))
        return predictNumPatches(80, 16, height, width)

def prepData_FaceForensics(resultsDir, dataFiles):
    print("Yay, cropped FaceForensics")
    numFeatures = len(dataFiles)

    #First get the names of all the files
    allFiles = createFileList(resultsDir, baseNamesOnly=False)
    allFiles = sorted(allFiles)

    #print(allFiles)

    allTheFeatures = []
    allTheLabels = []

    for dataFile in dataFiles:
        relevantFiles = [f for f in allFiles if dataFile in f]
        relevantFiles = sorted(relevantFiles)
        print("Relevant files")
        print(relevantFiles)
        features = []
        labels = []
        numEntries = 0
        for filename in relevantFiles:
            print(filename)
            my_data = np.genfromtxt(filename, delimiter=',')
            my_data = my_data.flatten()
            numPatches = calculateNumPatches(filename)
            my_data = my_data[0:numPatches]
            print(my_data.shape[0])
            numEntries = numEntries + my_data.shape[0]
            features.extend(my_data)
            if "altered" in filename:
                bunchOfLabels = [1] * my_data.shape[0]
            else:
                bunchOfLabels = [0] * my_data.shape[0]
            labels.extend(bunchOfLabels)

        numEntries = len(features)
        features = np.asarray(features).flatten()
        labels = np.asarray(labels).flatten()
        allTheFeatures.append(features)
        allTheLabels.append(labels)
        print("There are {} features of {}".format(features.shape[0], dataFile))
        print("There are {} labels of {}".format(labels.shape[0], dataFile))

    mememe = np.asarray(allTheFeatures)
    print(allTheFeatures)
    mememe = np.concatenate(allTheFeatures).ravel()
    print(mememe)
    print(mememe.shape)
    data = np.asarray(allTheFeatures).reshape(numFeatures, numEntries).T
    labels = np.asarray(allTheLabels).reshape(numFeatures, numEntries).T
    if numFeatures > 1:
        # Some jiggery pokery needed with the labels - they should all be the same
        equal = True
        for n in range(0, labels.shape[1]):
            col0 = labels[:, 0]
            coln = labels[:, n]
            equal = equal and np.array_equal(col0, coln)
        if not equal:
            print("WARNING, SOME SCREW UP HERE!!!!!!")
        else:
            labels = labels[:, 0]

    print(data.shape)
    print(labels.shape)

    return data, labels, numFeatures

# From https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
def doTtest(a, b):
    from scipy import stats
    ## Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(a, b)
    print("independent t = {}".format(t2))
    print("independent p = {}".format(2 * p2))
    t, p = stats.ttest_rel(a, b)
    print("related t = {}".format(t))
    print("related p = {}".format(2 * p[0]))

def analyseData(data, labels, numFeatures):
    # data is a 2D numpy array where most significant dimension is number of samples and the second one is numFeatures
    # labels is the labels for the data (also numpy array 1D or "by 1"). Binary classes, 0 is authentic, 1 is tampered
    # numFeatures is the number of features we have

    class0indices = np.where(labels == 0)
    class0data = data[class0indices[0], :]
    class0data = class0data * 7
    class1indices = np.where(labels == 1)
    class1data = data[class1indices[0], :]
    class1data = class1data * 7

    print(class0data)
    print(class1data)

    doTtest(class0data, class1data)

    print("Mean Authentic: {}".format(np.mean(class0data)))
    print("Mean Tampered: {}".format(np.mean(class1data)))


    binrange = range(0,52,7)
    plt.hist(class0data, bins=binrange, label="Authentic", color='b', alpha=0.5)
    plt.hist(class1data, bins=binrange, label="Tampered", color='r', alpha=0.5)
    plt.title("Histogram of QP values")
    plt.legend()
    plt.xlabel("Measured QP")
    #xlabels = range(0, 52, 7)
    #plt.xticks(xlabels)
    plt.savefig("fig_FaceForensicsSubset.png")
    plt.show()




if __name__ == "__main__":
    dataPath = "/Users/pam/Documents/results/FaceForensics/crops"
    f = open("output.csv", "wb")
    writer = csv.writer(f)
    dataPaths = [["/Users/pam/Documents/results/FaceForensics/crops", "-"],]
    dataPaths = [["/Volumes/LaCie/data/FaceForensics/FullResults", "-"],]

    for example in dataPaths:
        dataPath = example[0]
        yuvFile = example[1]
        p,n = os.path.split(yuvFile)
        filebasename,e = os.path.splitext(n)
        print(n)
        #dataPath = "/Users/pam/Documents/results/Davino/tank/"
        #yuvFile = "/Users/pam/Documents/data/Davino_yuv/01_TANK_f.yuv"
        #dataPath = "/Users/pam/Documents/results/VTD/billiards"
        #yuvFile = "/Users/pam/Documents/data/VTD_yuv/billiards_f.yuv"
        cropDim = 80
        cropTempStep = 1
        cropSpacStep = 16
        #num_channels = 3
        #bit_depth = 8
        width = 1280
        height = 720
        doDataAnalysis = False


        dataFilesList = [["qpPred.csv"],
                         ["qpPred.csv", "deblockPred.csv", "ipPred.csv"]
                        ]
        dataFilesList = [
                         ["qpPred.csv", "deblockPred.csv", "ipPred.csv"],
                        ]

        #dataFilesList = [["qpPred.csv", "deblockPred.csv", "ipPred.csv"],]

        keyFramesOnly = True
        predFramesOnly = True
        addBorders = True
        balanceTypes = ["None", "SMOTE", "RandomOver", "RandomUnder"]
        balanceTypes = ["None",]
        #analysisType = "All"  # "ANN" or "SVM" or "Linear Regression" or "Decision Tree" or "Random Forest"
        # or "Naive Bayes" or "All"
        analysisTypes = ["ANN", "SVM", "Linear Regression", "Decision Tree", "Random Forest", "Naive Bayes"]
        analysisTypes = ["SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
        tuple = ["file", "features", "balancing", "classifier", "split", "TN", "FP", "FN", "TP", "accuracy", "f1 sum"]
        resultsList = []


        for dataFiles in dataFilesList:
            #for addBorders in [False, True]:
            for addBorders in [False]:
                #data, labels, numFeatures = prepData(dataPath, yuvFile, width, height, cropDim, cropTempStep, cropSpacStep,
                #                                                dataFiles, keyFramesOnly, addBorders, predFramesOnly)
                data, labels, numFeatures = prepData_FaceForensics(dataPath, dataFiles)

                if doDataAnalysis:
                    analyseData(data, labels, numFeatures)
                    #quit()




                #(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
                #print("trainX shape (from train_test_split):", trainX.shape)
                #print("trainY shape (from train_test_split)", trainY.shape)

                for balanceType in balanceTypes:
                    skf = StratifiedKFold(n_splits=4)
                    split = -1
                    for train, test in skf.split(data, labels):
                        split = split + 1
                        trainX = np.take(data, train, axis=0)
                        trainY = np.take(labels, train, axis=0)
                        testX = np.take(data, test, axis=0)
                        testY = np.take(labels, test, axis=0)
                        print("trainX shape:", trainX.shape)
                        print("trainY shape", trainY.shape)


                        # deal with the imbalance on the training set alone (so duplication doesn't bleed across test/train split)
                        trainX, trainY = balanceData(balanceType, trainX, trainY)



                        print("trainX shape:", trainX.shape)
                        print("trainY shape", trainY.shape)
                        trainY = trainY.reshape((trainY.shape[0],))


                        for analysisType in analysisTypes:
                            if analysisType == "ANN":
                                predictions = doNeuralNetAnalysis(trainX, trainY, testX, testY, numFeatures)
                            else:
                                if analysisType == "SVM":
                                    #svclassifier = SVC(kernel='linear')
                                    #svclassifier = SVC(kernel='poly', degree=8)
                                    model = SVC(kernel='rbf', gamma="auto")
                                    #svclassifier = SVC(kernel='sigmoid')
                                    model.fit(trainX, trainY)
                                    predictions = model.predict(testX)
                                if analysisType == "Decision Tree":
                                    model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
                                    model.fit(trainX, trainY)
                                    predictions = model.predict(testX)
                                if analysisType == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=100)
                                if analysisType == "Naive Bayes":
                                    model = GaussianNB()
                                if analysisType == "Linear Regression":
                                    model = LinearRegression()



                                model.fit(trainX, trainY)
                                predictions = model.predict(testX)
                                #scores = cross_val_score(model, data, labels, cv=5)
                            print("Doing analysis with {}".format(analysisType))
                            print(classification_report(testY, predictions))
                            cm = confusion_matrix(testY, predictions)
                            print(cm)
                            accuracy = accuracy_score(testY, predictions)
                            print(accuracy)
                            f1s = f1_score(testY, predictions, average=None)
                            total_f1s = np.sum(f1s)
                            print("Total f1 over both classes: {}".format(total_f1s))
                            featureCode = getFeatureCode(dataFiles, addBorders)
                            balanceCode = getBalanceCode(balanceType)
                            classifierCode = getClassifierCode(analysisType)
                            tuple = [filebasename, featureCode, balanceCode, classifierCode, split, cm[0,0], cm[0,1], cm[1,0], cm[1,1], accuracy, total_f1s]
                            resultsList.append(tuple)
                            writer.writerow(tuple)
                            f.flush()
                            print(tuple)

    print("****************************************************************")
    print("And now for the grand reveal")
    for tuple in resultsList:
        print(tuple)
    titles = ["File", "data", "balancing", "classifier", "stratification", "TN", "FP", "FN", "TP", "accuracy", "total f1"]
    with open("output_all.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(titles)
        writer.writerows(resultsList)

    quit()

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("plot.png")

