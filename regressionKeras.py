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

from keras import backend as K
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

if __name__ == "__main__":
    dataPath = "/Users/pam/Documents/results/tree/"
    yuvFile = "/Users/pam/Documents/data/Davino_yuv/08_TREE_f.yuv"
    #dataPath = "/Users/pam/Documents/results/Davino/tank/"
    #yuvFile = "/Users/pam/Documents/data/Davino_yuv/01_TANK_f.yuv"
    #dataPath = "/Users/pam/Documents/results/VTD/billiards"
    #yuvFile = "/Users/pam/Documents/data/VTD_yuv/billiards_f.yuv"

    fileSize = os.path.getsize(yuvFile)
    cropDim = 80
    cropTempStep = 1
    cropSpacStep = 16
    num_channels = 3
    bit_depth = 8
    width=1280
    height=720
    numFrames = fileSize // (width * height * 3 / 2)
    analysisType = "All" # "ANN" or "SVM" or "Linear Regression" or "Decision Tree" or "Random Forest"
                    # or "Naive Bayes" or "All"
    patchWidth, patchHeight, patchedFrames, numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                      tempStep=cropTempStep, spacStep=cropSpacStep, height=height, width=width)

    dataFiles = ["qpPred.csv", "deblockPred.csv", "ipPred.csv", "diffs.csv"]
    dataFiles = ["qpPred.csv"]
    keyFrameFiles = ["qpPred.csv", "diffs.csv"]
    qpPred = 0
    #dataFiles = ["qpPred.csv", "ipPred.csv", "diffs.csv"]
    labelFile = "gt.csv"

    #First construct my data numpy array
    fullDataFiles = [os.path.join(dataPath, s) for s in keyFrameFiles]
    print(fullDataFiles)
    
    keyFrames = pickOutKeyFrames(fullDataFiles, patchWidth, patchHeight, numPatches, patchedFrames)
    print(keyFrames)
    keyFramesOnly = True
    #keyFramesOnly = False

    dataList = []
    addBorders = True # because doing "true" will screw things up a little

    for file in dataFiles:
        filename = os.path.join(dataPath, file)
        print(filename)
        my_data = np.genfromtxt(filename, delimiter=',')
        my_data = my_data.flatten()
        my_data = my_data[0:numPatches]
        print("The data from the cvs file: {} and numFrames {}".format(my_data.shape, numFrames))
        if addBorders:
            #numFrames = patchedFrames
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

        print(my_data.shape)

        if keyFramesOnly:
            reducedSet = []
            patchFrameSize = patchWidth * patchHeight
            for frame in keyFrames:
                start = frame * patchFrameSize
                end = start + patchFrameSize

                for i in range(0, my_data.shape[0]):
                    keyframe_data = my_data[i, start:end].flatten()
                    keyframe_data = keyframe_data.tolist()
                    reducedSet.extend(keyframe_data)
            my_data = np.asarray(reducedSet)
            numKeyFrames = len(keyFrames)
            #patchedFrames = numFrames


        print(my_data.shape)
        my_data = my_data.tolist()
        dataList.extend(my_data)
    if keyFramesOnly:
        numFrames = numKeyFrames
        patchedFrames = numKeyFrames


    #
    numFeatures = len(dataFiles)
    if addBorders:
        numFeatures = len(dataFiles) * 5 # Because centre plus 4 borders (maybe 8 would work better?).

    numPatches = numFrames * patchHeight * patchWidth


    # Adding in left, right, top, bottom QP values
    #if addBorders:
    #    filename = os.path.join(dataPath, "qpPred.csv")
    #    normPredVals = np.genfromtxt(filename, delimiter=',')
    #    normPredVals = normPredVals[0:numPatches]



    #    numFrames = patchedFrames
    #    predsHeight = patchHeight
    #    predsWidth = patchWidth
    #    normPredVals = normPredVals.reshape((numFrames, predsHeight, predsWidth))
    #    firstCol = normPredVals[:, :, 0].reshape((numFrames, predsHeight, 1))
    #    lastCol = normPredVals[:, :, (predsWidth - 1)].reshape((numFrames, predsHeight, 1))
    #    firstRow = normPredVals[:, 0, :].reshape((numFrames, 1, predsWidth))
    #    lastRow = normPredVals[:, (predsHeight - 1), :].reshape((numFrames, 1, predsWidth))

    #    normPredVals_left = np.append(firstCol, normPredVals[:, :, :(predsWidth - 1)], axis=2).flatten().tolist()
    #    normPredVals_right = np.append(normPredVals[:, :, 1:], lastCol, axis=2).flatten().tolist()
    #    normPredVals_top = np.append(firstRow, normPredVals[:, :(predsHeight - 1), :], axis=1).flatten().tolist()
    #    normPredVals_bottom = np.append(normPredVals[:, 1:, :], lastRow, axis=1).flatten().tolist()
    #    print(len(normPredVals_left))
    #    dataList.extend(normPredVals_left)
    #    dataList.extend(normPredVals_right)
    #    dataList.extend(normPredVals_top)
    #    dataList.extend(normPredVals_bottom)
    #    numFeatures = len(dataFiles) + 4


    my_data = np.asarray(dataList)
    my_data = my_data.reshape((numFeatures, numPatches))
    my_data = np.swapaxes(my_data, 0, 1)
    #my_data = np.swapaxes(my_data, 1, 2)
    #my_data = np.swapaxes(my_data, 2, 3)


    print(my_data.shape)
    data = my_data
    #data = data.reshape((4, patchedFrames, patchHeight, patchWidth))

    filename = os.path.join(dataPath, labelFile)
    labels = np.genfromtxt(filename, delimiter=',')
    labels = labels[0:numPatches]
    #labels = labels.reshape((patchedFrames, patchHeight, patchWidth, 1))

    # Analyse the data a bit in here: What is the data balance?
    unique, counts = np.unique(labels, return_counts=True)
    print("How unbalanced: {}".format(dict(zip(unique, counts))))
    labels = labels.reshape((labels.shape[0], 1))
    all_data = np.append(labels, data, axis=1)
    #all_data = balanceByOverSampling()
    print(all_data.shape)



    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

    # deal with the imbalance on the training set alone (so duplication doesn't bleed across test/train split)
    balanceData=False
    balanceType = "RandomOver"
    if balanceData:
        if balanceType == "SMOTE":
            sm = SMOTE(random_state=12, ratio=1.0)
            x_train_res, y_train_res = sm.fit_sample(trainX, trainY)
        if balanceType == "RandomOver":
            sm = RandomOverSampler(random_state=12)
            x_train_res, y_train_res = sm.fit_sample(trainX, trainY)
        if balanceType == "RandomUnder":
            sm = RandomUnderSampler(random_state=12)
            x_train_res, y_train_res = sm.fit_sample(trainX, trainY)


        trainX = x_train_res
        trainY = y_train_res


    #lb = LabelBinarizer()
    #trainY = lb.fit_transform(trainY)
    #testY = lb.transform(testY)
    print("trainX shape:", trainX.shape)
    print("trainY shape", trainY.shape)

    #trainX = trainX.reshape((patchedFrames, patchHeight, patchWidth, 4))
    #testX = testX.reshape((patchedFrames, patchHeight, patchWidth, 4))
    if analysisType == "ANN" or analysisType == "All":
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


        #model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
        #model.add(Activation("relu"))
        ##model.add(BatchNormalization(axis=chanDim))
        #model.add(Conv2D(32, (5, 5), padding="same"))
        #model.add(Activation("relu"))
        ##model.add(BatchNormalization(axis=chanDim))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        #model = Sequential()
        #model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        #model.add(Activation('relu'))
        #model.add(Conv2D(32, (3, 3)))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        #model.add(Conv2D(64, (3, 3), padding='same'))
        #model.add(Activation('relu'))
        #model.add(Conv2D(64, (3, 3)))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        #model.add(Flatten())
        #model.add(Dense(512))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(num_classes))
        ##model.add(Activation('softmax'))
        #model.add(Activation(tf.nn.softmax))

        # initialize our initial learning rate and # of epochs to train for
        INIT_LR = 0.01
        EPOCHS = 4

        # compile the model using SGD as our optimizer and categorical
        # cross-entropy loss (you'll want to use binary_crossentropy
        # for 2-class classification)
        print("[INFO] training network...")
        #opt = SGD(lr=INIT_LR)
        #model.compile(loss="categorical_crossentropy", optimizer=opt,
        #              metrics=["accuracy"])

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        #opt = keras.optimizers.Adam()
        #opt = RMSprop(0.001)

        # Let's train the model using RMSprop
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        #model.compile(loss='binary_crossentropy',
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
        H = model.fit(trainX, train_Y, validation_data=(testX, test_Y), epochs=EPOCHS, batch_size=32, class_weight=class_weights)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=32)
        print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=["0","1"]))


        print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
    if analysisType == "SVM"  or analysisType == "All":
        print("Doing analysis with SVM")

        #svclassifier = SVC(kernel='linear')
        #svclassifier = SVC(kernel='poly', degree=8)
        svclassifier = SVC(kernel='rbf')
        #svclassifier = SVC(kernel='sigmoid')
        svclassifier.fit(trainX, trainY)
        predictions = svclassifier.predict(testX)
        print(classification_report(testY, predictions))
        print(confusion_matrix(testY, predictions))

    if analysisType == "Linear Regression"  or analysisType == "All":
        print("Doing analysis with Linear Regression")
        print("But it's not actually implemented...")
    if analysisType == "Decision Tree"  or analysisType == "All":
        print("Decision Tree")
        clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        clf.fit(trainX, trainY)
        predictions = clf.predict(testX)
        print(classification_report(testY, predictions))
        print(confusion_matrix(testY, predictions))

    if analysisType == "Random Forest" or analysisType == "All":
        from sklearn.ensemble import RandomForestClassifier
        print("Random Forest")
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(trainX, trainY)
        predictions = clf.predict(testX)
        print(classification_report(testY, predictions))
        print(confusion_matrix(testY, predictions))
    if analysisType == "Naive Bayes"  or analysisType == "All":
        print("Naive Bayes Model")
        model_naive = GaussianNB()
        model_naive.fit(trainX, trainY)
        predictions = model_naive.predict(testX)
        print(classification_report(testY, predictions))
        print(confusion_matrix(testY, predictions))

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

