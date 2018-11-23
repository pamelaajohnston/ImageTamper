"""Evaluation for Quantisation parameters

Accuracy:
No idea

Speed:
No idea

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

Originally scavenged from:
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import os

import itNet
import itNet_input
# import liftedTFfunctions
import patchIt2 as pi
import functions
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 56392, """Number of examples to run.""")
tf.app.flags.DEFINE_integer('network_architecture', 1,
                            """The number of the network architecture to use (inference function) """)
tf.app.flags.DEFINE_string('mylog_dir_eval', '/Users/pam/Documents/temp/', """Directory where to write my logs """)
tf.app.flags.DEFINE_string('yuvfile', '/Users/pam/Documents/data/testyuv/mobile_cif_q7.yuv',
                           """The input yuv 4:2:0 video file """)
tf.app.flags.DEFINE_string('heatmap', 'heatmap', """The heatmap yuv 4:2:0 video dir """)
tf.app.flags.DEFINE_integer('cropDim', 80, """Patch dimensions (must be good for network architecture)""")
tf.app.flags.DEFINE_integer('cropTempStep', 1, """Temporal step between patches""")
tf.app.flags.DEFINE_integer('cropSpacStep', 16, """Spatical step between patches""")


def eval_once(saver, summary_writer, top_k_op, summary_op, gen_confusionMatrix, predLabels):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found {} and {}'.format(FLAGS.checkpoint_dir, ckpt))
            return 0, None

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions, batchConfusionMatrix, mypreds = sess.run([top_k_op, gen_confusionMatrix, predLabels])
                batchConfusionMatrix = np.asarray(batchConfusionMatrix)
                batchPredictions = np.asarray(predictions)
                batchMypreds = np.asarray(mypreds)
                # print("Step: {} and predictions: \n {}".format(step, batchPredictions))
                # print("Step: {} and logits: \n {}".format(step, batchMypreds))
                batchConfusionMatrix = batchConfusionMatrix.reshape((itNet_input.NUM_CLASSES, itNet_input.NUM_CLASSES))
                if step == 0:
                    # confusionMatrix = np.asarray(tf.unstack(batchConfusionMatrix))
                    confusionMatrix = batchConfusionMatrix
                    allPredLabels = batchMypreds
                else:
                    confusionMatrix = np.add(confusionMatrix, batchConfusionMatrix)
                    allPredLabels = np.append(allPredLabels, batchMypreds)
                # numRight = np.sum(predictions)
                # print("test step {} of {} ".format(step, num_iter))
                # print("We got {} correct".format(numRight))
                # print("Here's the batchCM:\n {}".format(batchConfusionMatrix))
                # print("Here's the cm:\n {}".format(confusionMatrix))
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            # print('For calculating precision: {} correct out of {} samples'.format(true_count, total_sample_count))
            # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            # print("Unstacking the confusion matrix")
            # cm = tf.unstack(confusionMatrix)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            # print("Predicted labels for {}: \n {}".format(step, allPredLabels))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        # return precision
        return precision, confusionMatrix, allPredLabels


def confusion_matrix(labels, predictions):
    tf.confusion_matrix(labels, predictions)


def evaluate(returnConfusionMatrix=True, filename="", numPatches=100, predFilename="pred.csv"):
    num_classes = itNet_input.NUM_CLASSES

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        # print("getting input data {}".format(FLAGS.eval_data))
        eval_data = FLAGS.eval_data == 'test'
        images, labels = itNet.inputs(eval_data=True, singleThreaded=True, filename=filename,
                                      numExamplesToTest=numPatches)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # print("calling inference")
        logits = itNet.inference_switch(images, FLAGS.network_architecture)
        predictions = tf.argmax(logits, 1)
        gen_confusionMatrix = tf.confusion_matrix(labels, predictions, num_classes=num_classes)

        # Calculate predictions.
        # print("calculating predictions")
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        # print("restore moving average version of learned variables")
        variable_averages = tf.train.ExponentialMovingAverage(itNet.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # print("Building a summary")
        summary_op = tf.summary.merge_all()

        # print("And a summary writer")
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        logfile = os.path.join(FLAGS.mylog_dir_eval, "log_evals.txt")
        log = open(logfile, 'w')
        start = datetime.now()

        # print("Calling eval_once")
        precision, confusionMatrix, predLabels = eval_once(saver, summary_writer, top_k_op, summary_op,
                                                           gen_confusionMatrix, predictions)
        # precision = eval_once(saver, summary_writer, top_k_op, summary_op)

        np.set_printoptions(threshold='nan')
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        print('{}: confusionMatrix: \n {}'.format(datetime.now(), confusionMatrix))
        # print('AND predictions (in order provided you only used one thread!!!): \n ')
        # print(np.array2string(predLabels, separator=', '))
        np.savetxt(predFilename, predLabels, delimiter=',', fmt='%1.0f')

        rightNow = datetime.now()
        difference = rightNow - start
        log.write("*******************************************************\n")
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            log.write("The checkpoint was: {} ok now".format(ckpt.model_checkpoint_path))
        else:
            log.write("No checkpoint found yet")
        log.write("time: {} seconds \n".format(difference.total_seconds()))
        log.write('precision @ 1 = %.5f \n' % (precision))
        # log.write('confusionMatrix: \n {} \n'.format(confusionMatrix))
        cmString = np.array2string(confusionMatrix, separator='\t')
        cmString = cmString.replace('[', '')
        cmString = cmString.replace(']', '')
        log.write('confusionMatrix: \n {} \n'.format(cmString))

        # log.write("******************************************************* \n")
        log.flush()


# The network configurations
qpNetwork = {
    "summary": "qp",
    "num_classes": 8,
    "network_architecture": 28,
    "checkpoint_dir": "../trainedNetworks/it_train_bs128_20k/",
    "predFilename": "qpPred.csv"
}

ipNetwork = {
    "summary": "ip",
    "num_classes": 2,
    "network_architecture": 28,
    "checkpoint_dir": "../trainedNetworks/it_train_IP_opt1/",
    "predFilename": "ipPred.csv"
}
deblockNetwork = {
    "summary": "deblock",
    "num_classes": 2,
    "network_architecture": 28,
    "checkpoint_dir": "../trainedNetworks/it_train_deblock_opt1/",
    "predFilename": "deblockPred.csv"
}
clustersNetwork = {
    "summary": "clusters",
    "num_classes": 5,
    "network_architecture": 28,
    "checkpoint_dir": "dummy",
    "predFilename": "clusters.csv"
}

#keyframesNetwork = {
#    "summary": "keyframes",
#    "num_classes": 5,
#    "network_architecture": 28,
#    "checkpoint_dir": "dummy",
#    "predFilename": ""
#}

def predictNumPatches(fileSize, cropDim, tempStep, spacStep, height, width):
    frameSize = width * height * 3 // 2
    num_frames = fileSize // frameSize
    patchedFrames = num_frames // tempStep
    patchWidth = (width - cropDim)//spacStep
    patchHeight = (height - cropDim)//spacStep
    numPatches = patchedFrames * patchWidth * patchHeight
    return numPatches

def deriveMaskFilename(filename):
    if "Davino" in filename or "VTD" in filename or "SULFA" in filename:
        maskfilename = filename.replace("_f.yuv", "_mask.yuv")

    return maskfilename



#def main(argv=None):  # pylint: disable=unused-argument
def doEverything():
    doPatching = False
    doEvaluation = False
    doClustering = False
    doFrameAnalysis = False
    doYUVSummary = False
    doGroundTruthProcessing = False
    doIOU = False
    doHeatmaps = False
    multiTruncatedOutput = False

    #doPatching = True
    #doEvaluation = True
    doClustering = True
    #doFrameAnalysis = True
    doYUVSummary = True
    doGroundTruthProcessing = True
    doIOU = True
    doHeatmaps = True
    #multiTruncatedOutput = True

    numPatches = 0

    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # We'll create the patches locally and save them.
    myDataDirName = "temp"
    myHeatmapFileDir = FLAGS.heatmap
    if doEvaluation:
        if tf.gfile.Exists(myDataDirName):
            tf.gfile.DeleteRecursively(myDataDirName)
        tf.gfile.MakeDirs(myDataDirName)

    #if tf.gfile.Exists(myHeatmapFileDir):
    #    tf.gfile.DeleteRecursively(myHeatmapFileDir)
    tf.gfile.MakeDirs(myHeatmapFileDir)
    #if tf.gfile.Exists(FLAGS.heatmap):
    #    os.remove(FLAGS.heatmap)

    inFilename = os.path.join(FLAGS.data_dir, FLAGS.yuvfile)
    outFilename = os.path.join(myDataDirName, "test.bin")
    maskFilename = deriveMaskFilename(inFilename)
    print(maskFilename)
    clustersCSV = os.path.join(myHeatmapFileDir, "clusters.csv")
    gtCSV = os.path.join(myHeatmapFileDir, "gt.csv")

    width, height = pi.getDimsFromFileName(inFilename)

    cropDim = FLAGS.cropDim
    cropTempStep = FLAGS.cropTempStep
    cropSpacStep = FLAGS.cropSpacStep
    num_channels = 3
    bit_depth = 8

    predsWidth = (width - cropDim) // cropSpacStep
    predsHeight = (height - cropDim) // cropSpacStep

    networks = [qpNetwork, ipNetwork, deblockNetwork]


    fileSize = os.path.getsize(inFilename)
    numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                                   tempStep=cropTempStep, spacStep=cropSpacStep, height=height, width=width)
    frameSize = width * height
    numFrames = int((fileSize // (frameSize * 3 / 2)) // cropTempStep)

    print(numPatches)


    # pi.patchOneFile(fileIn=inFilename, fileOut=outFilename, label="qp",
    #                cropDim=80, cropTempStep=1, cropSpacStep=16, num_channels=3, bit_depth=8)

    if doPatching:
        print("Begin patching")
        numPatches = pi.patchOneFile(fileIn=inFilename, fileOut=outFilename, label="none",
                                     cropDim=cropDim, cropTempStep=cropTempStep, cropSpacStep=cropSpacStep,
                                     num_channels=num_channels, bit_depth=bit_depth
                                     )
        print("End patching")

    if doEvaluation:
        print("Begin evaluation - calculating predictions on the trained networks")
        for network in networks:
            FLAGS.checkpoint_dir = network['checkpoint_dir']
            FLAGS.network_architecture = network['network_architecture']
            predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
            itNet_input.NUM_CLASSES = network['num_classes']
            itNet.NUM_CLASSES = network['num_classes']

            # print("Data dir: {}".format(FLAGS.data_dir))
            # print("Batches dir: {}".format(FLAGS.batches_dir))
            # print("Eval dir: {}".format(FLAGS.eval_dir))
            print("Checkpoint dir: {}".format(FLAGS.checkpoint_dir))
            print("Network Architecture: {}".format(FLAGS.network_architecture))
            print("Num Classes: {}".format(itNet_input.NUM_CLASSES))
            print("YUV file: {}".format(FLAGS.yuvfile))
            print("heatmap file: {}".format(FLAGS.heatmap))
            print("Testing {} patches".format(numPatches))

            FLAGS.num_examples = numPatches
            print("Evaluating on network {}".format(FLAGS.checkpoint_dir))
            evaluate(filename=outFilename, numPatches=numPatches, predFilename=predFilename)
            # precision, confusionMatrix, predLabels = evaluate()
        print("Done evaluation")

    if doClustering:
        print("Begin combination of preds")

        if kmeansClustering:
            allPreds = []
            clusteredNetworks = [qpNetwork, deblockNetwork]
            clusteredNetworks = [qpNetwork,]
            for network in clusteredNetworks:
                predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                predVals = np.loadtxt(predFilename)
                predVals = predVals[0:numPatches]
                normPredVals = predVals / np.linalg.norm(predVals)
                allPreds.extend(normPredVals)

            allPreds = np.asarray(allPreds)
            allPreds = allPreds.flatten()
            allPreds = allPreds.reshape(len(clusteredNetworks), numPatches)
            allPreds = np.swapaxes(allPreds, 0, 1)
            #print(allPreds)
            model = KMeans(n_clusters=2)

            # Fitting Model
            model.fit(allPreds)

            # Prediction on the entire data
            all_predictions = model.predict(allPreds)
            np.savetxt(clustersCSV, all_predictions, delimiter=",", fmt='%1.0f')
            #print(all_predictions)

        print("End combination of preds")








    keyFrames = np.asarray(range(0, numFrames))
    if doFrameAnalysis:
        print("Doing frame analysis")
        patchWidth = (width - cropDim) // cropSpacStep
        patchHeight = (height - cropDim) // cropSpacStep
        patchFrame = patchHeight * patchWidth

        #clusteredNetworks = [qpNetwork, deblockNetwork]
        clusteredNetworks = [qpNetwork,]
        keyFrames = []
        for i, network in enumerate(clusteredNetworks):
            predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
            predVals = np.loadtxt(predFilename)
            predVals = predVals[0:numPatches]
            predVals = (predVals / np.linalg.norm(predVals))*100 # normalising
            predVals = predVals.reshape(numFrames,patchFrame)
            avgs = predVals.mean(axis=1)*1000
            #print(avgs)
            plotDiff = True
            if plotDiff:
                avgs = np.diff(avgs)
                avgs = np.insert(avgs, 0, avgs[0]).reshape(-1, 1)
                frames = range(0, numFrames)
            else:
                avgs = avgs.reshape(-1, 1)
                frames = range(0, numFrames)

            #print("Here's the diffs for {}".format(network['summary']))
            #print(avgs)
            plt.plot(frames, avgs)
            plt.title("Frame average for {}".format(network['predFilename']))
            plt.xlabel("Frame number")
            plt.ylabel("Average value from {}".format(network['predFilename']))
            plt.savefig("a_{}.png".format(network['summary']))
            plt.close()
            model = KMeans(n_clusters=2)
            model.fit(avgs)
            frameTypes = model.predict(avgs)
            # Now make sure the minority class (the key frames) is always 1
            numOnes = len(np.where(frameTypes == 1))
            numZeros = len(np.where(frameTypes == 1))
            if numOnes > numZeros:
                print("Inverting binary matrix, but this might not work!")
                frameTypes = 1 - frameTypes
            mylist = np.where(frameTypes == 1)
            keyFrames.append(mylist[0].tolist())

        keyFrames = np.asarray(keyFrames)
        #print(keyFrames)
        keyFrames = keyFrames.flatten()
        #print(keyFrames)
        keyFrames = np.unique(keyFrames)
        print("End of frame analysis")

    if doYUVSummary:
        print("Summarising YUV and mask files with key frames {}".format(keyFrames))
        # create a cut-down YUV file
        yuvFiles = [inFilename, maskFilename] # and add in ground truth mask or whatever
        for file in yuvFiles:
            keyframesFilename = os.path.join(myHeatmapFileDir, "keyframes.yuv")
            if "mask" in file:
                keyframesFilename = os.path.join(myHeatmapFileDir, "keyframes_mask.yuv")
            if tf.gfile.Exists(keyframesFilename):
                os.remove(keyframesFilename)
            with open(file, "rb") as f:
                mybytes = np.fromfile(f, 'u1')
            fs = int(height * width * 3/2)

            for f in keyFrames:
                start = f * fs
                end = start + fs
                theFrame = mybytes[start:end]
                functions.appendToFile(theFrame, keyframesFilename)
        print("End of YUV summarising")






    if doGroundTruthProcessing:
        print("Turning mask file {} into a pred csv".format(maskFilename))
        fs = int(height * width * 3 / 2)
        with open(maskFilename, "rb") as f:
            mybytes = np.fromfile(f, 'u1')

        mybytes = mybytes.reshape((numFrames, fs))
        mybytes = mybytes[:, 0:frameSize] # Take only Y component
        mybytes = mybytes.reshape((numFrames, height, width))
        patchesList = []
        for f in range(0, numFrames, cropTempStep):
            for y in range((cropDim//2), (height - (cropDim//2)), cropSpacStep):
                yend = y + cropSpacStep
                for x in range((cropDim//2), (width - (cropDim//2)), cropSpacStep):
                    xend = x + cropSpacStep
                    patch = mybytes[f, y:yend, x:xend]
                    patch = patch.flatten()
                    patch = patch - 16
                    #print("The patch is")
                    #print(patch)

                    label = 0
                    if np.sum(patch) != 0:
                        label = 1
                    #print("the label is {}".format(label))

                    patchesList.append(label)
        patches = np.asarray(patchesList)
        #print("There are {} patches from a {} by {} file".format(patches.shape, width, height))
        np.savetxt(gtCSV, patches, delimiter=',', fmt='%1.0f')
        print("Finished turning mask.yuv into gt.csv")

    if doIOU:
        print("Comparing gt.csv and clusters.csv using Intersection over Union")
        iouResult = 0

        predsPerFrame = predsWidth * predsHeight
        totalPreds = predsPerFrame * keyFrames.shape[0]

        gtVals = np.loadtxt(gtCSV)
        gtVals = gtVals[0:totalPreds]
        predVals = np.loadtxt(clustersCSV)
        predVals = predVals[0:totalPreds]
        gtVals = gtVals != 0
        np.set_printoptions(threshold=np.nan)
        #print(gtVals)
        predVals = predVals != 0
        #print(predVals)

        intersect = gtVals & predVals
        intersect = np.count_nonzero(intersect, axis=0)
        #print(intersect)
        union = gtVals | predVals
        union = np.count_nonzero(union, axis=0)
        #print(union)

        if union == 0:
            iouResult = 0
        else:
            iouResult = intersect / union


        print("Done the comparison. Result was IOU={} which is {} over {} for file {} for frames {}".format(iouResult,
                                                                                                intersect,
                                                                                                union,
                                                                                                inFilename,
                                                                                                keyFrames))









    if multiTruncatedOutput:
        keyFrames = range(0, numFrames)



    # Now generate the heatmap from the preds in "pred.csv"
    if doHeatmaps:
        print("Begin generating heatmaps")
        for generateAll in [0, 1, 2]:
            for network in networks:
                if generateAll == 0:
                    predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                    myHeatmapFileName = "{}.yuv".format(network['summary'])
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, myHeatmapFileName)
                    predVals = np.loadtxt(predFilename)
                elif generateAll == 1: # the clusters
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, "clusters.yuv")
                    predVals = np.loadtxt(clustersCSV)
                else: # the ground truth
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, "gt_blocked.yuv")
                    predVals = np.loadtxt(gtCSV)

                print(predVals.shape)
                predVals = predVals[0:numPatches]


                if tf.gfile.Exists(myHeatmapFileName):
                   os.remove(myHeatmapFileName)


                if numFrames == 0:
                    numFrames = 1

                predsPerFrame = predVals.shape[0] // numFrames
                # print(predVals.reshape((predsHeight, predsWidth)))
                print("numFrames {} predsWidth {} predsHeight {} predsPerFrame {}".format(numFrames, predsWidth, predsHeight,
                                                                                          predsPerFrame))

                # This is because we're using 8 labels.
                multiplier = 256 // network['num_classes']
                predVals = predVals * multiplier
                padding = cropDim // 2

                uvValue = 128 # this is for grey
                uv = np.full((frameSize // 2), uvValue)

                for f in keyFrames:
                    predsStart = f * predsPerFrame
                    predsEnd = predsStart + predsPerFrame
                    framePreds = predVals[predsStart:predsEnd]
                    framePreds = framePreds.reshape((predsHeight, predsWidth))
                    framePreds = framePreds.repeat(cropSpacStep, axis=0).repeat(cropSpacStep, axis=1)
                    framePreds = np.pad(framePreds, ((padding, padding), (padding, padding)), 'edge')
                    functions.appendToFile(framePreds, myHeatmapFileName)
                    functions.appendToFile(uv, myHeatmapFileName)


                if generateAll > 0:
                    break

        print("Done generating heatmap(s)")

    #print("The shape of the YUV: {}".format(predVals.shape))
    # uv = np.full(heatmap.shape, 128)
    # heatmap = np.append(heatmap, uv)
    # heatmap = np.append(heatmap, uv)

    #if tf.gfile.Exists(myDataDirName):
    #    tf.gfile.DeleteRecursively(myDataDirName)
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)


yuvfileslist =[
    #["/Users/pam/Documents/data/Davino_yuv/", "01_TANK_f.yuv", "/Users/pam/Documents/results/Davino/tank"],
    #["/Users/pam/Documents/data/Davino_yuv/", "02_MAN_f.yuv", "/Users/pam/Documents/results/Davino/man"],
    #["/Users/pam/Documents/data/Davino_yuv/", "03_CAT_f.yuv", "/Users/pam/Documents/results/Davino/cat"],
    #["/Users/pam/Documents/data/Davino_yuv/", "04_HELICOPTER_f.yuv", "/Users/pam/Documents/results/Davino/helicopter"],
    #["/Users/pam/Documents/data/Davino_yuv/", "05_HEN_f.yuv", "/Users/pam/Documents/results/Davino/hen"],
    #["/Users/pam/Documents/data/Davino_yuv/", "06_LION_f.yuv", "/Users/pam/Documents/results/Davino/lion"],
    #["/Users/pam/Documents/data/Davino_yuv/", "07_UFO_f.yuv", "/Users/pam/Documents/results/Davino/ufo"],
    #["/Users/pam/Documents/data/Davino_yuv/", "08_TREE_f.yuv", "/Users/pam/Documents/results/Davino/tree"],
    ["/Users/pam/Documents/data/Davino_yuv/", "09_GIRL_f.yuv", "/Users/pam/Documents/results/Davino/girl"],
    ["/Users/pam/Documents/data/Davino_yuv/", "10_DOG_f.yuv", "/Users/pam/Documents/results/Davino/dog"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "01_f.yuv", "/Users/pam/Documents/results/SULFA/01"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "02_f.yuv", "/Users/pam/Documents/results/SULFA/02"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "03_f.yuv", "/Users/pam/Documents/results/SULFA/03"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "04_f.yuv", "/Users/pam/Documents/results/SULFA/04"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "05_f.yuv", "/Users/pam/Documents/results/SULFA/05"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "06_f.yuv", "/Users/pam/Documents/results/SULFA/06"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "07_f.yuv", "/Users/pam/Documents/results/SULFA/07"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "08_f.yuv", "/Users/pam/Documents/results/SULFA/08"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "09_f.yuv", "/Users/pam/Documents/results/SULFA/09"],
    ["/Users/pam/Documents/data/SULFA_yuv/", "10_f.yuv", "/Users/pam/Documents/results/SULFA/10"],
]

def main(argv=None):  # pylint: disable=unused-argument
    runAbunch = False
    if runAbunch:
        for entry in yuvfileslist:
            FLAGS.data_dir = entry[0]
            FLAGS.yuvfile = entry[1]
            FLAGS.heatmap = entry[2]
            doEverything()
    else:
        doEverything()

if __name__ == '__main__':
    tf.app.run()
