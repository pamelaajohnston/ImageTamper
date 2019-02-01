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




if __name__ == "__main__":
    dataPath = "/Users/pam/Documents/results/tree/"
    dataPath = "/Users/pam/Documents/results/Davino/tank/"
    yuvFile = "/Users/pam/Documents/data/Davino_yuv/08_TREE_f.yuv"
    yuvFile = "/Users/pam/Documents/data/Davino_yuv/01_TANK_f.yuv"

    fileSize = os.path.getsize(yuvFile)
    cropDim = 80
    cropTempStep = 1
    cropSpacStep = 16
    num_channels = 3
    bit_depth = 8
    width=1280
    height=720
    patchWidth, patchHeight, patchedFrames, numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                      tempStep=cropTempStep, spacStep=cropSpacStep, height=height, width=width)

    dataFiles = ["qpPred.csv", "deblockPred.csv", "ipPred.csv", "diffs.csv"]
    #dataFiles = ["qpPred.csv", "ipPred.csv", "diffs.csv"]
    labelFile = "gt.csv"

    #First construct my data numpy array

    dataList = []

    for file in dataFiles:
        filename = os.path.join(dataPath, file)
        my_data = np.genfromtxt(filename, delimiter=',')
        my_data = my_data.flatten()
        my_data = my_data[0:numPatches]
        print(my_data.shape)
        my_data = my_data.tolist()
        dataList.extend(my_data)

    #
    my_data = np.asarray(dataList)
    my_data = my_data.reshape((len(dataFiles), numPatches))
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

    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

    #lb = LabelBinarizer()
    #trainY = lb.fit_transform(trainY)
    #testY = lb.transform(testY)
    print("trainY shape", trainY.shape)
    trainY = np_utils.to_categorical(trainY)
    testY = np_utils.to_categorical(testY)

    print("trainX shape:", trainX.shape)
    print("trainY shape", trainY.shape)

    #trainX = trainX.reshape((patchedFrames, patchHeight, patchWidth, 4))
    #testX = testX.reshape((patchedFrames, patchHeight, patchWidth, 4))
    inputShape = (len(dataFiles),)
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
    EPOCHS = 10

    # compile the model using SGD as our optimizer and categorical
    # cross-entropy loss (you'll want to use binary_crossentropy
    # for 2-class classification)
    print("[INFO] training network...")
    #opt = SGD(lr=INIT_LR)
    #model.compile(loss="categorical_crossentropy", optimizer=opt,
    #              metrics=["accuracy"])

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # train the neural network
    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["0","1"]))

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

