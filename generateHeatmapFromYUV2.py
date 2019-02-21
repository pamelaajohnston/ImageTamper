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
And then adapted to run with the FaceForensics dataset.
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
import glob
import sys
import shlex
import subprocess
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

        #np.set_printoptions(threshold='nan')
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
frameDiffNetwork = {
    "summary": "frameDiff",
    "num_classes": 2,
    "network_architecture": 28,
    "checkpoint_dir": "dummy",
    "predFilename": "diffs.csv"
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

    # This feels like a hack...
    if (width % 16) != 0:
        patchWidth = patchWidth + 1
    if (height % 16) != 0:
        patchHeight = patchHeight + 1

    numPatches = patchedFrames * patchWidth * patchHeight
    return numPatches

def deriveMaskFilename(filename):
    maskfilename = filename.replace(".yuv", "_mask.yuv")
    if "Davino" in filename or "VTD" in filename or "SULFA" in filename:
        maskfilename = filename.replace("_f.yuv", "_mask.yuv")
    if "realisticTampering" in filename:
        maskfilename = filename.replace("all", "mask")

    return maskfilename

def frameDiffs(infilename, graphname, width, height, cropDim, cropTempStep, cropSpacStep, csvname):
    sigmas = 2.15 # more than 2 sigmas different from average gives a key frame
    with open(infilename, "rb") as f:
        mybytes = np.fromfile(f, 'u1')
    #print("There are {} bytes in the file width: {} height: {}".format(len(mybytes), width, height))
    frameSize = int((width * height) * 3/2) # assuming yuv 420
    num_frames = int(len(mybytes) / frameSize)
    #print("There are {} frames".format(num_frames))

    pixels = mybytes.reshape((num_frames, frameSize))
    frameDeltas = np.diff(pixels, axis=0)
    #frameDeltasNonabs = frameDeltas
    frameDeltas = abs(frameDeltas)
    #if not np.array_equal(frameDeltas, frameDeltasNonabs):
    #    print("YEAH there is a mistake")
    #    print(frameDeltas)
    #    print(frameDeltasNonabs)
    avgs = np.average(frameDeltas, axis=1)
    # normalise
    avgsavg = np.average(avgs)
    avgsstd = np.std(avgs)
    scores = abs(avgs-avgsavg)/avgsstd # or a "standard score"
    avgs = scores
    avgs = np.insert(avgs, 0, sigmas).reshape(-1, 1) # ???!!!insert sigmas since first frame will usually be "key"
    frames = range(0,num_frames)

    plt.plot(frames, avgs)
    #plt.title("Mean Frame Delta")
    #plt.xlabel("Frame number")
    #plt.ylabel("Average delta")
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the left edge are off
        labelleft=False)  # labels along the bottom edge are off
    plt.savefig(graphname, bbox_inches='tight')
    plt.close()

    avgs = avgs.flatten()
    interestingFrames = np.where(avgs >= sigmas)
    #print(interestingFrames)

    # Now convert frameDeltas to a csv, with 1 indicating a change from previous frame and 0 otherwise
    # This doesn't work because it's one less frame - pad it out with a frame and also, add another nested for
    # for frames!!!!!!!!!
    frameDeltas = frameDeltas.flatten()
    # pad it
    zeroFrame = np.zeros(frameSize)
    frameDeltas = np.append(zeroFrame, frameDeltas)
    frameDeltas = frameDeltas.reshape((num_frames, frameSize))
    frameDeltas = frameDeltas[:, 0:(width * height)] # WARNING - taking only the Y (not the U and V)
    frameDeltas = frameDeltas.reshape((num_frames, height, width))

    # The +1's here are to give extra room for frames that aren't a multiple of 16
    mbDiffs = np.zeros((num_frames, ((height//cropSpacStep)+1), ((width//cropSpacStep))+1), dtype="float32")
    border = 0
    #border = int(((cropDim/2)//cropSpacStep)*cropSpacStep)
    #print("The border is {}".format(border))
    print(mbDiffs.shape)
    print(cropSpacStep)

    for f in range(0, num_frames, cropTempStep):
        for j in range(border, (height - border), cropSpacStep):
            for i in range(border, (width - border), cropSpacStep):
                ii = (i-border)//cropSpacStep
                jj = (j-border)//cropSpacStep
                myTile = frameDeltas[f, j:(j+cropSpacStep), i:(i+cropSpacStep)]
                total = np.sum(myTile)
                #print("Frame {}, ({}, {}), total: {} going to csv ({}, {})".format(f, i, j, total, ii, jj))
                if total > 0:
                    #print("Frame {}, ({}, {}), total: {} going to csv ({}, {})".format(f, i, j, total, ii, jj))
                    mbDiffs[f, jj, ii] = int(1)

    # This is where we adjust for the patch/stride stuff)
    bordermbs = int(((cropDim/2)//cropSpacStep))
    #print(bordermbs)
    #print(((height//16) - bordermbs))
    #print(((width//16) - bordermbs))
    mbDiffs = mbDiffs[:, bordermbs:((height//cropSpacStep) - bordermbs)-1, bordermbs:((width//cropSpacStep) - bordermbs)-1]
    print(mbDiffs.shape)


    mbDiffs = mbDiffs.flatten()

    np.savetxt(csvname, mbDiffs, delimiter=",", fmt='%1.0f')


    return interestingFrames

def getVTDselectedframes(filename):
    if "archery" in filename:
        return [0, 140, 160, 236, 279, 320]
    if "audirs7" in filename:
        return [0, 25,  81, 96, 97, 122, 144, 153, 170, 288, 380]
    if "basketball" in filename:
        return [0, 160, 320]
    if "billiards" in filename:
        return [0, 15, 79, 160, 210, 299, 320, 385]
    if "bowling" in filename:
        return [0, 135, 137, 160, 162, 163, 165, 175, 177, 185, 250, 251, 320]
    if "bullet" in filename:
        return [0,   1,   2,   5,  75, 144, 195, 205, 206, 209, 220, 224, 288, 301, 304, 321, 336]
    if "cake" in filename:
        return [0,  93,  95,  96,  97,  98,  99, 101, 102, 156, 160, 300, 320]
    if "camera" in filename:
        return [0, 160, 220, 259, 297, 320, 335, 374, 451]
    if "carpark" in filename:
        return [0,   1,   2,  95,  96,  97,  98, 160, 249, 280, 282, 286, 287, 288, 320, 359]
    if "carplate" in filename:
        return [0,   1,   3,   5,   7,  98, 144, 181, 182, 183, 186, 187, 188, 189,
                190, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
                207, 208, 209, 210, 211, 212, 213, 288, 368, 414, 415]
    if "cctv" in filename:
        return [0,   1,   2,  16, 120, 160, 208, 211, 213, 215, 225, 227, 229,
       239, 270, 272, 300, 320, 384]
    if "clarity" in filename:
        return[0,   1,  16, 127, 160, 193, 240, 320, 352, 403, 416, 464]
    if "cuponk" in filename:
        return [0,  50, 144, 188, 268, 272, 274, 276, 277, 278, 279, 280, 281,
       282, 284, 288, 321]
    if "dahua" in filename:
        return [0, 150, 160, 169, 170, 177, 180, 192, 216, 303, 304, 320, 358, 359]
    if "football" in filename:
        return [0, 160, 197, 320, 332]
    if "highway" in filename:
        return [0, 47, 144, 236, 288, 393, 418]
    if "kitchen" in filename:
        return [0,  95, 123, 126, 127, 128, 129, 130, 132, 133, 160, 255, 256,
       258, 259, 260, 261, 320]
    if "manstreet" in filename:
        return [  0,   1,   4,  43,  55,  65, 160, 192, 209, 210, 235, 240, 261,
       273, 320, 364, 415, 428]
    if "passport" in filename:
        return [0,  34, 144, 154, 168, 286, 288, 296, 298, 300, 355, 371, 372, 428, 430]
    if "plane" in filename:
        return [0,  39, 160, 189, 232, 233, 238, 239, 240, 241, 243, 244, 320, 356, 359]
    if "pong" in filename:
        return [0,  10,  24,  28,  32,  37,  39,  41,  44, 144, 187, 288]
    if "studio" in filename:
        return [0,  34,  69, 144, 229, 240, 288, 354, 366, 406]
    if "swann" in filename:
        return [0,  88,  89, 103, 122, 128, 160, 211, 212, 272, 273, 300, 303,
       320, 331, 332, 422, 452]
    if "swimming" in filename:
        return [0, 114, 115, 116, 117, 120, 121, 122, 123, 124, 160, 320, 361, 417, 418,
       419, 423]
    if "whitecar" in filename:
        return [0,   1,   2,  90, 160, 210, 278, 320, 321, 337, 359, 361]
    if "yellowcar" in filename:
        return [0,  48,  64,  72, 160, 320, 355, 375, 434, 435]




def doEverything(resultsLog, threshold=1):
    doFrameDiffs = False
    doSelectedFramesOnly = False
    doPatching = False
    doEvaluation = False
    doClustering = False
    doFrameAnalysis = False
    doYUVSummary = False
    doGroundTruthProcessing = False
    doAverages = False
    doIOU = False
    doHeatmaps = False
    multiTruncatedOutput = False

    #doFrameDiffs = True
    #doSelectedFramesOnly = True # selects some frame from one of the tables
    #doPatching = True # Patches up the YUV file
    #doEvaluation = True # Evaluates the patches using whatever networks are programmed (takes ages but saves results to file)
    #doClustering = True # Clusters the results somehow - sub options are available
    #doFrameAnalysis = True # need "doFrameAnalysis" if we're to extract the key frames!
    #doYUVSummary = True #extracts only the key frames to a summary file
    #doGroundTruthProcessing = True # takes the ground truth and turns it into a csv (16x16 granularity)
    #doAverages = True # Looks at the averages and plots a profile (for mask=0 and mask=1)
    #doIOU = True # Actually compares clusters to gt using IOU, F1, MCC, TPR, FPR among other things
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
    diffsCSV = os.path.join(myHeatmapFileDir, "diffs.csv")

    width, height = pi.getDimsFromFileName(inFilename)
    print(width,height)

    cropDim = FLAGS.cropDim
    cropTempStep = FLAGS.cropTempStep
    cropSpacStep = FLAGS.cropSpacStep
    num_channels = 3
    bit_depth = 8

    predsWidth = (width - cropDim) // cropSpacStep
    predsHeight = (height - cropDim) // cropSpacStep

    # This feels like a hack...
    if (width % 16) != 0:
        predsWidth = predsWidth + 1
    if (height % 16) != 0:
        predsHeight = predsHeight + 1

    networks = [qpNetwork, ipNetwork, deblockNetwork]


    fileSize = os.path.getsize(inFilename)
    numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                                   tempStep=cropTempStep, spacStep=cropSpacStep, height=height, width=width)
    frameSize = width * height
    numFrames = int((fileSize // (frameSize * 3 / 2)) // cropTempStep)
    print("Predicted {} patches".format(numPatches))

    selectedFrames = range(0, numFrames)
    if doFrameDiffs:
        print("Doing frame diffs")
        # This is just a way of finding potential key frames
        graphName = os.path.join(myHeatmapFileDir, "fig_frameDiffs")
        selectedFrames = frameDiffs(inFilename, graphName, width, height, cropDim, cropTempStep, cropSpacStep, diffsCSV)
        selectedFrames = selectedFrames[0].tolist()
        print("End of frame diffs")

    if doSelectedFramesOnly:
        selectedFrames = getVTDselectedframes(inFilename)
        print("Doing these frames only: {} on file {}".format(selectedFrames, inFilename))
        tempFileName = os.path.join(myHeatmapFileDir, "selectedFramesOnly{}x{}_f.yuv".format(width, height))
        tempMaskFile = deriveMaskFilename(tempFileName)
        # Copy the "selected frames into the temporary file
        yuvFilesIn = [inFilename, maskFilename] # and add in ground truth mask or whatever
        yuvFilesOut = [tempFileName, tempMaskFile]  # and add in ground truth mask or whatever

        fs = int(height * width * 3 / 2)
        for i, file in enumerate(yuvFilesIn):
            outfile = yuvFilesOut[i]
            print("{} going to {}".format(file, outfile))
            if tf.gfile.Exists(outfile):
                os.remove(outfile)
            with open(file, "rb") as f:
                mybytes = np.fromfile(f, 'u1')
            for f in selectedFrames:
                start = f * fs
                end = start + fs
                theFrame = mybytes[start:end]
                functions.appendToFile(theFrame, outfile)
        #Reset the names and continue as normal
        inFilename = tempFileName
        maskFilename = tempMaskFile
        fileSize = os.path.getsize(inFilename)
        numPatches = predictNumPatches(fileSize=fileSize, cropDim=cropDim,
                                       tempStep=cropTempStep, spacStep=cropSpacStep, height=height, width=width)
        frameSize = width * height
        numFrames = int((fileSize // (frameSize * 3 / 2)) // cropTempStep)
        print("Predicted {} patches".format(numPatches))
        print("End of selecting frames, rehashed input yuv and mask files")





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
        print("Predswidth {} predsheigh {} numPatches {}".format(predsWidth, predsHeight, numPatches))
        kmeansClustering, advancedKmeans, thresholding = False, True, False
        perFrame = False
        if kmeansClustering:
            print("Clustering is kmeans")
            allPreds = []
            clusteredNetworks = [qpNetwork, deblockNetwork]
            clusteredNetworks = [qpNetwork, frameDiffNetwork]
            #clusteredNetworks = [qpNetwork]
            #clusteredNetworks = [frameDiffNetwork]
            for network in clusteredNetworks:
                predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                predVals = np.loadtxt(predFilename)
                predVals = predVals[0:numPatches]
                normPredVals = predVals / np.linalg.norm(predVals)
                allPreds.extend(normPredVals)

            allPreds = np.asarray(allPreds)
            allPreds = allPreds.flatten()
            print(len(clusteredNetworks))
            allPreds = allPreds.reshape(len(clusteredNetworks), numPatches)
            allPreds = np.swapaxes(allPreds, 0, 1)
            #print(allPreds)
            if perFrame:
                allPreds = allPreds.reshape((numFrames, (predsHeight*predsWidth),  len(clusteredNetworks)))
                #all_predictions = []
                for f in range(0,numFrames):
                    myPreds = allPreds[f, :, :]
                    model = KMeans(n_clusters=2)
                    # Fitting Model
                    model.fit(myPreds)
                    # Prediction on the entire data
                    if f==0:
                        all_predictions = model.predict(myPreds)
                    else:
                        all_predictions = np.append(all_predictions, model.predict(myPreds))


            else:
                model = KMeans(n_clusters=2)
                # Fitting Model
                model.fit(allPreds)
                # Prediction on the entire data
                all_predictions = model.predict(allPreds)

            # now all_predictions is either 0 or 1. Assume that "tampered"(1) is the minority class
            t = all_predictions.sum()
            print("Sum {} vs length {}".format(t, all_predictions.shape[0]))
            #if t > all_predictions.shape[0]:
            #    all_predictions = np.add(all_predictions, -1)

            np.savetxt(clustersCSV, all_predictions, delimiter=",", fmt='%1.0f')
            #print(all_predictions)
        elif advancedKmeans:
            print("Clustering is advanced kmeans (i.e. using the 8? neighbours as a feature)")
            allPreds = []
            clusteredNetworks = [qpNetwork, deblockNetwork]
            clusteredNetworks = [qpNetwork, frameDiffNetwork]
            #clusteredNetworks = [qpNetwork]
            #clusteredNetworks = [frameDiffNetwork]
            for network in clusteredNetworks:
                predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                predVals = np.loadtxt(predFilename)
                predVals = predVals[0:numPatches]
                normPredVals = predVals / np.linalg.norm(predVals)
                allPreds.extend(normPredVals)

                normPredVals = normPredVals.reshape((numFrames, predsHeight, predsWidth))

                #print(normPredVals[0, 0:5, 0:5])

                firstCol = normPredVals[:, :, 0].reshape((numFrames, predsHeight, 1))
                lastCol = normPredVals[:, :, (predsWidth-1)].reshape((numFrames, predsHeight, 1))
                firstRow = normPredVals[:, 0, :].reshape((numFrames, 1, predsWidth))
                lastRow = normPredVals[:, (predsHeight-1), :].reshape((numFrames, 1, predsWidth))

                normPredVals_left = np.append(firstCol, normPredVals[:, :, :(predsWidth-1)], axis=2)
                #print("left")
                #print(normPredVals_left[0, 0:5, 0:5])

                normPredVals_right = np.append(normPredVals[:, :, 1:], lastCol, axis=2)
                #print("right")
                #print(normPredVals_right[0, 0:5, 0:5])

                normPredVals_top = np.append(firstRow, normPredVals[:, :(predsHeight-1), :], axis=1)
                #print("top")
                #print(normPredVals_top[0, 0:5, 0:5])

                normPredVals_bottom = np.append(normPredVals[:, 1:, :], lastRow, axis=1)
                #print("bottom")
                #print(normPredVals_bottom[0, 0:5, 0:5])
                print(normPredVals_left.shape)
                print(normPredVals_right.shape)
                print(normPredVals_top.shape)
                print(normPredVals_bottom.shape)

                allPreds.extend(normPredVals_left.flatten())
                allPreds.extend(normPredVals_right.flatten())
                allPreds.extend(normPredVals_top.flatten())
                allPreds.extend(normPredVals_bottom.flatten())


            allPreds = np.asarray(allPreds)
            allPreds = allPreds.flatten()
            print(len(clusteredNetworks))
            featuresPerPatch = len(clusteredNetworks) * 5
            allPreds = allPreds.reshape(featuresPerPatch, numPatches)
            allPreds = np.swapaxes(allPreds, 0, 1)
            print(allPreds.shape)
            if perFrame:
                allPreds = allPreds.reshape((numFrames, (predsHeight*predsWidth),  len(clusteredNetworks)))
                #all_predictions = []
                for f in range(0,numFrames):
                    myPreds = allPreds[f, :, :]
                    model = KMeans(n_clusters=2)
                    # Fitting Model
                    model.fit(myPreds)
                    # Prediction on the entire data
                    if f==0:
                        all_predictions = model.predict(myPreds)
                    else:
                        all_predictions = np.append(all_predictions, model.predict(myPreds))


            else:
                model = KMeans(n_clusters=2)
                # Fitting Model
                model.fit(allPreds)
                # Prediction on the entire data
                all_predictions = model.predict(allPreds)

            # now all_predictions is either 0 or 1. Assume that "tampered"(1) is the minority class
            t = all_predictions.sum()
            print("Sum {} vs length {}".format(t, all_predictions.shape[0]))
            #if t > all_predictions.shape[0]:
            #    all_predictions = np.add(all_predictions, -1)

            np.savetxt(clustersCSV, all_predictions, delimiter=",", fmt='%1.0f')
            #print(all_predictions)
        elif thresholding:
            #threshold = 1
            print("Clustering is thresholding, threshold is qp={}".format(threshold))
            allPreds = []
            clusteredNetworks = [qpNetwork, frameDiffNetwork]

            clusteredNetworks = [qpNetwork]
            #clusteredNetworks = [frameDiffNetwork]
            for network in clusteredNetworks:
                predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                predVals = np.loadtxt(predFilename)
                predVals = predVals[0:numPatches]
                #normPredVals = predVals / np.linalg.norm(predVals)
                allPreds = predVals

            allPreds = np.asarray(allPreds)
            allPreds = allPreds.flatten()
            print(len(clusteredNetworks))

            allPreds = allPreds.reshape(len(clusteredNetworks), numPatches)
            allPreds = np.swapaxes(allPreds, 0, 1)
            #print(allPreds[0:20, :])
            all_predictions = np.zeros(allPreds.shape)

            all_predictions[np.where(allPreds < threshold)] = 1

            # now all_predictions is either 0 or 1. Assume that "tampered"(1) is the minority class
            t = all_predictions.sum()
            print("Sum {} vs length {}".format(t, all_predictions.shape[0]))
            #if t > all_predictions.shape[0]:
            #    all_predictions = np.add(all_predictions, -1)

            np.savetxt(clustersCSV, all_predictions, delimiter=",", fmt='%1.0f')
            #print(all_predictions)
        else:
            print("Clustering is custom")







        print("End combination of preds")








    keyFrames = np.asarray(range(0, numFrames))
    interestingFrames = keyFrames
    if doFrameAnalysis:
        print("Doing frame analysis")
        #patchWidth = (width - cropDim) // cropSpacStep
        #patchHeight = (height - cropDim) // cropSpacStep
        #patchFrame = patchHeight * patchWidth
        patchFrame = predsHeight * predsWidth

        #clusteredNetworks = [qpNetwork, deblockNetwork]
        clusteredNetworks = [qpNetwork, deblockNetwork, ipNetwork]
        keyFrames = [0,]
        for i, network in enumerate(clusteredNetworks):
            predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
            predVals = np.loadtxt(predFilename)
            predVals = predVals[0:numPatches]
            predVals = (predVals / np.linalg.norm(predVals))*100 # normalising
            predVals = predVals.reshape(numFrames,patchFrame)

            # discard the last frame because it fades to black and is nonsense
            predVals = np.append(predVals[0:(numFrames-2), :], predVals[(numFrames-4):(numFrames-2), :]).reshape(numFrames,patchFrame)

            avgs = predVals.mean(axis=1)*1000
            #print(avgs)
            #for j, k in enumerate(avgs):
            #    print("{}     {}".format(j, k))

            #print(avgs)
            plotDiff = True
            #if network['summary'] == "ip":
            #    plotDiff = False
            if plotDiff:
                avgs = np.diff(avgs)
                avgs = np.insert(avgs, 0, avgs[0]).reshape(-1, 1) # to align it to frame numbers, insert 0 at front
                frames = range(0, numFrames)
            else:
                avgs = avgs.reshape(-1, 1)
                frames = range(0, numFrames)

            #print("Here's the diffs for {}".format(network['summary']))
            #print(avgs)
            plt.plot(frames, avgs)
            #plt.title("Frame average for predicted {}".format(network['summary']))
            #plt.xlabel("Frame number")
            #plt.ylabel("Average value of {}".format(network['summary']))
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the left edge are off
                labelleft=False)  # labels along the bottom edge are off
            filename = os.path.join(myHeatmapFileDir, "fig_{}".format(network['summary']))
            plt.savefig(filename, bbox_inches='tight')
            plt.close()


            if i == 0:
                faTotals = avgs
            else:
                faTotals = np.multiply(avgs, faTotals)

        plt.plot(frames, faTotals)
        #plt.title("Frame totals for all")
        #plt.xlabel("Frame number")
        #plt.ylabel("Average value from all")
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            labelleft=False)  # labels along the bottom edge are off
        filename = os.path.join(myHeatmapFileDir, "fig_all")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        model = KMeans(n_clusters=2)
        model.fit(faTotals)
        frameTypes = model.predict(faTotals)
        # Now make sure the minority class (the key frames) is always 1
        numOnes = len(np.where(frameTypes == 1))
        numZeros = len(np.where(frameTypes == 1))
        if numOnes > numZeros:
            print("Inverting binary matrix, but this might not work!")
            frameTypes = 1 - frameTypes
        mylist = np.where(frameTypes == 1)
        # To hell with this clustering, lets just go with stats!

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
        interestingFrames = keyFrames
        print("The key frames for {}: {}".format(myHeatmapFileDir, keyFrames))
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
    else:
        keyFrames = np.asarray(range(0, numFrames))

    if doGroundTruthProcessing:
        print("Turning mask file {} into a pred csv".format(maskFilename))
        fs = int(height * width * 3 / 2)
        with open(maskFilename, "rb") as f:
            mybytes = np.fromfile(f, 'u1')

        # check on number of frames:
        maskNumFrames = mybytes.shape[0] // fs
        if maskNumFrames > numFrames:
            mybytes = mybytes[0:(numFrames*fs)]
        if maskNumFrames < numFrames:
            diff = numFrames - maskNumFrames
            # Just append a few frames on the end....
            mybytes = np.append(mybytes, mybytes[0:(diff * fs)])

        #quit()
        print("Numframes {} vs mask frames {}".format(numFrames, maskNumFrames))

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
                    if "Davino" in maskFilename:
                        patch = patch - 16
                    #print("The patch is")
                    #print(patch)

                    label = 0
                    if np.sum(patch) != 0:
                        label = 1
                    #print("the label is {}".format(label))

                    patchesList.append(label)
        patches = np.asarray(patchesList)
        print("There are {} patches from a {} by {} file".format(patches.shape, width, height))
        np.savetxt(gtCSV, patches, delimiter=',', fmt='%1.0f')
        print("Finished turning mask.yuv into gt.csv")

    if doAverages:
        print("Doing Averages")
        # First check that the mask is available:

        filesToCheck = [qpNetwork['predFilename'], diffsCSV]
        predsPerFrame = predsWidth * predsHeight
        totalPreds = predsPerFrame * keyFrames.shape[0]

        if not os.path.isfile(gtCSV):
            print("The ground truth file {} does not exist".format(gtCSV))
            gtVals = np.zeros(totalPreds)
            gtVals[0:predsPerFrame] = 1
        else:
            gtVals = np.loadtxt(gtCSV)
            gtVals = gtVals[0:totalPreds]

        for file in filesToCheck:
            f = os.path.basename(file)
            f,b = os.path.splitext(f)
            predFilename = os.path.join(myHeatmapFileDir, file)
            predVals = np.loadtxt(predFilename)
            predVals = predVals[0:totalPreds]

            a_all = np.average(predVals)
            v_all = np.var(predVals)
            a_mask0 = np.average(predVals[np.where(gtVals == 0)])
            v_mask0 = np.var(predVals[np.where(gtVals == 0)])
            a_mask1 = np.average(predVals[np.where(gtVals == 1)])
            v_mask1 = np.var(predVals[np.where(gtVals == 1)])
            print("Averages: File {} and {}  Overall average: {} {} nomask: {} {} mask: {} {}".format(myHeatmapFileDir, f, a_all, v_all, a_mask0, v_mask0, a_mask1, v_mask1))

            if file == qpNetwork['predFilename']:
                bins = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                xLabel = "Predicted Quantisation Parameter"
                xTicks = range(0, 52, 7)
                xTickPosn = [bin+0.5 for bin in bins]
            else:
                bins = [0,1,2]
                xLabel = "Macroblock comparison in consecutive frames"
                xTicks = ["same", "different", ""]
                xTickPosn = [bin+0.5 for bin in bins]

            mask0 = predVals[np.where(gtVals == 0)]
            mask1 = predVals[np.where(gtVals == 1)]

            mask0_w = np.empty(mask0.shape)
            mask0_w.fill(1 / mask0.shape[0])
            mask1_w = np.empty(mask1.shape)
            mask1_w.fill(1 / mask1.shape[0])

            n0, bins0, patches0 = plt.hist([mask0, mask1],
                                           bins=bins,
                                           color=['#fffb00', '#000000'],
                                           weights=[mask0_w, mask1_w],
                                           label=['authentic', 'masked'],
                                           rwidth=1.0)
            #n1, bins1, patches1 = plt.hist(predVals[np.where(gtVals == 1)], bins=bins, facecolor='yellow', alpha = 0.5, normed=1)
            plt.xticks(xTickPosn, xTicks, horizontalalignment='center')
            plt.title("Mask Histogram")
            plt.xlabel(xLabel)
            plt.ylabel("Frequency")
            plt.legend()
            filename = os.path.join(myHeatmapFileDir, "a_mask_hist_{}".format(f))
            plt.savefig(filename)
            plt.close()

            # And "key frames only"
            if interestingFrames.shape[0] != keyFrames.shape[0]:
                #print("Plus doing the average on just the key frames")
                predVals2 = predVals.reshape((keyFrames.shape[0], predsPerFrame))
                gtVals2 = gtVals.reshape((keyFrames.shape[0], predsPerFrame))
                predVals2 = predVals2[interestingFrames, :]
                gtVals2 = gtVals2[interestingFrames, :]

                a_all = np.average(predVals2)
                v_all = np.var(predVals2)
                a_mask0 = np.average(predVals2[np.where(gtVals2 == 0)])
                v_mask0 = np.var(predVals2[np.where(gtVals2 == 0)])
                a_mask1 = np.average(predVals2[np.where(gtVals2 == 1)])
                v_mask0 = np.var(predVals2[np.where(gtVals2 == 1)])
                print("Averages: File {} and {}  key average: {} {} nomask: {} {} mask: {} {}".format(myHeatmapFileDir, f, a_all, v_all, a_mask0, v_mask0, a_mask1, v_mask1))

                mask0 = predVals2[np.where(gtVals2 == 0)]
                mask1 = predVals2[np.where(gtVals2 == 1)]

                if mask1.shape[0] == 0 or mask0.shape[0] == 0:
                    print("no duality in detected key frames")
                    continue

                mask0_w = np.empty(mask0.shape)
                mask0_w.fill(1 / mask0.shape[0])
                mask1_w = np.empty(mask1.shape)
                mask1_w.fill(1 / mask1.shape[0])

                #n0, bins0, patches0 = plt.hist(predVals2[np.where(gtVals2 == 0)], bins=bins, facecolor='blue', alpha = 0.5, normed=1)
                n0, bins0, patches0 = plt.hist([mask0, mask1],
                                               bins=bins,
                                               color=['#fffb00', '#000000'],
                                               weights=[mask0_w, mask1_w],
                                               label=['authentic', 'masked'],
                                               rwidth=1.0)
                #n1, bins1, patches1 = plt.hist(predVals2[np.where(gtVals2 == 1)], bins=bins, facecolor='yellow', alpha = 0.5, normed=1)
                plt.xticks(xTickPosn, xTicks, horizontalalignment='center')
                plt.title("Mask Histogram")
                plt.xlabel(xLabel)
                plt.ylabel("Frequency")
                plt.legend()
                filename = os.path.join(myHeatmapFileDir, "a_mask_key_hist_{}".format(f))
                plt.savefig(filename)
                plt.close()
            else:
                print("Not doing key frames")


        print("Finished doing averages")

    if doIOU:
        print("Comparing gt.csv and clusters.csv using Intersection over Union")
        iouResult = 0

        predsPerFrame = predsWidth * predsHeight
        totalPreds = predsPerFrame * keyFrames.shape[0]

        gtVals = np.loadtxt(gtCSV)
        gtVals = gtVals[0:totalPreds]
        predVals = np.loadtxt(clustersCSV)
        predVals = predVals[0:totalPreds]

        if not doYUVSummary:
            print("We didn't summarise earlier so doing it now")
            if not doFrameAnalysis or (keyFrames.shape[0] == interestingFrames.shape[0]):
                print("WARNING:!!!! Your results will be awful as you're not isolating key frames....!!!!!")
            gtKeyVals = np.zeros((interestingFrames.shape[0], predsPerFrame))
            predKeyVals = np.zeros((interestingFrames.shape[0], predsPerFrame))
            gtVals = gtVals.reshape(keyFrames.shape[0], predsPerFrame)
            predVals = predVals.reshape(keyFrames.shape[0], predsPerFrame)

            for i,f in enumerate(interestingFrames):
                gtKeyVals[i, :] = gtVals[f, :]
                predKeyVals[i, :] = predVals[f, :]
            gtVals = gtKeyVals.flatten()
            predVals = predKeyVals.flatten()
            print("Reducto!")

        #print(predVals.shape)

        gtVals = gtVals != 0
        #np.set_printoptions(threshold=np.nan)
        #print(gtVals)
        predVals = predVals != 0
        #print(predVals)

        intersect = gtVals & predVals
        #print(intersect)
        intersect = np.count_nonzero(intersect, axis=0)
        #print(intersect)
        union = gtVals | predVals
        union = np.count_nonzero(union, axis=0)
        #print(union)

        if union.all() == 0:
            iouResult = 0
        else:
            iouResult = intersect / union

        # This is lazy and you should use your brain
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        #print(gtVals)
        for i in range(0, gtVals.shape[0]):
            gtVal = gtVals[i]
            predVal = predVals[i]
            if [gtVal, predVal] == [True, True]:
                tp = tp +1
            if [gtVal, predVal] == [False, False]:
                tn = tn +1
            if [gtVal, predVal] == [True, False]:
                fn = fn + 1
            if [gtVal, predVal] == [False, True]:
                fp = fp +1

        asum = (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)
        if asum == 0:
            mcc = -2
        else:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        asum = (2*tp) + fn + fp
        if asum == 0:
            f1 = 0
        else:
            f1 = (2*tp)/asum




        print("IOU Results IOU={} which is {} over {}; tp:{}, tn:{}, fp:{}, fn:{}, mcc:{}, f1:{} "
              "for file {} for frames {}".format(iouResult,
                                                 intersect,
                                                 union,
                                                 tp,
                                                 tn,
                                                 fp,
                                                 fn,
                                                 mcc,
                                                 f1,
                                                 inFilename,
                                                 interestingFrames))
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)
        print("tpr:{}, fpr:{}".format(tpr, fpr))
        resultsLog.write("{}; {}; {}; {}; {}; {}; {}; {}; {}\n".format(inFilename, tp, tn, fp, fn, mcc, f1, iouResult, interestingFrames))









    if multiTruncatedOutput:
        keyFrames = range(0, numFrames)



    # Now generate the heatmap from the preds in "pred.csv"
    if doHeatmaps:
        heatmapNetworks = [qpNetwork, ipNetwork, deblockNetwork]
        print("Begin generating heatmaps")
        # 0 is all the networks, 1 is the clusters, 2 is the diffs, 3 is the ground truth
        #for generateAll in [0, 1, 2, 3]:
        for generateAll in [0, 2]:
            for network in heatmapNetworks:
                doBorders = True
                predsWidth = (width - cropDim) // cropSpacStep
                predsHeight = (height - cropDim) // cropSpacStep
                # This is because we're using 8 labels.
                multiplier = 256 // network['num_classes']

                if generateAll == 0:
                    predFilename = os.path.join(myHeatmapFileDir, network['predFilename'])
                    myHeatmapFileName = "{}.yuv".format(network['summary'])
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, myHeatmapFileName)
                    predVals = np.loadtxt(predFilename)
                elif generateAll == 1: # the clusters
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, "clusters.yuv")
                    predVals = np.loadtxt(clustersCSV)
                    multiplier = 255
                elif generateAll == 2:  # the frameDiffs
                    myHeatmapFileName = os.path.join(myHeatmapFileDir, "frameDiffs.yuv")
                    predVals = np.loadtxt(diffsCSV)
                    #doBorders = False
                    multiplier = 255
                else: # the ground truth
                    myHeatmapFileName =os.path.join(myHeatmapFileDir, "gt_blocked.yuv")
                    predVals = np.loadtxt(gtCSV)
                    multiplier = 255

                #print(predVals.shape)
                if doBorders:
                    predVals = predVals[0:numPatches]
                else:
                    predsWidth = (width) // cropSpacStep
                    predsHeight = (height) // cropSpacStep


                if tf.gfile.Exists(myHeatmapFileName):
                   os.remove(myHeatmapFileName)


                if numFrames == 0:
                    numFrames = 1

                predsPerFrame = predVals.shape[0] // numFrames
                # print(predVals.reshape((predsHeight, predsWidth)))
                print(myHeatmapFileName)
                print("numFrames {} predsWidth {} predsHeight {} predsPerFrame {}".format(numFrames,
                                                                                          predsWidth,
                                                                                          predsHeight,
                                                                                          predsPerFrame))


                predVals = predVals * multiplier
                padding = cropDim // 2
                if not doBorders:
                    padding = 0

                uvValue = 128 # this is for grey
                uv = np.full((frameSize // 2), uvValue)

                for f in keyFrames:
                    predsStart = f * predsPerFrame
                    predsEnd = predsStart + predsPerFrame
                    framePreds = predVals[predsStart:predsEnd]
                    framePreds = framePreds.reshape((predsHeight, predsWidth))
                    framePreds = framePreds.repeat(cropSpacStep, axis=0).repeat(cropSpacStep, axis=1)
                    if doBorders:
                        framePreds = np.pad(framePreds, ((padding, padding), (padding, padding)), 'edge')
                    # Caution - this only works if it's the height that's a non-16 multiple
                    framePreds = framePreds[:height, :width]
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

def getAverageQPinCSVs(dir):
    print(dir)
    csvlist = glob.glob(os.path.join(dir, "*", "qpPred.csv"))
    print(csvlist)
    avgs = []
    avgTotal = 0
    for csv in csvlist:
        vals = np.loadtxt(csv)
        avg = np.average(vals)
        avgTotal = avgTotal + avg
        avgs = np.append(avgs, avg)
    finalAverage = avgTotal/len(csvlist)
    avgs = np.asarray(avgs)

    a = np.average(avgs)
    std = np.std(avgs)
    return a, std





yuvfileslist_video =[
    ["/Users/pam/Documents/data/Davino_yuv/", "01_TANK_f.yuv", "/Users/pam/Documents/results/Davino/tank"],
    ["/Users/pam/Documents/data/Davino_yuv/", "02_MAN_f.yuv", "/Users/pam/Documents/results/Davino/man"],
    ["/Users/pam/Documents/data/Davino_yuv/", "03_CAT_f.yuv", "/Users/pam/Documents/results/Davino/cat"],
    ["/Users/pam/Documents/data/Davino_yuv/", "04_HELICOPTER_f.yuv", "/Users/pam/Documents/results/Davino/helicopter"],
    ["/Users/pam/Documents/data/Davino_yuv/", "05_HEN_f.yuv", "/Users/pam/Documents/results/Davino/hen"],
    ["/Users/pam/Documents/data/Davino_yuv/", "06_LION_f.yuv", "/Users/pam/Documents/results/Davino/lion"],
    ["/Users/pam/Documents/data/Davino_yuv/", "07_UFO_f.yuv", "/Users/pam/Documents/results/Davino/ufo"],
    ["/Users/pam/Documents/data/Davino_yuv/", "08_TREE_f.yuv", "/Users/pam/Documents/results/Davino/tree"],
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

yuvfileslist_davino =[
    ["/Users/pam/Documents/data/Davino_yuv/", "01_TANK_f.yuv", "/Users/pam/Documents/results/Davino/tank"],
    ["/Users/pam/Documents/data/Davino_yuv/", "02_MAN_f.yuv", "/Users/pam/Documents/results/Davino/man"],
    ["/Users/pam/Documents/data/Davino_yuv/", "03_CAT_f.yuv", "/Users/pam/Documents/results/Davino/cat"],
    ["/Users/pam/Documents/data/Davino_yuv/", "04_HELICOPTER_f.yuv", "/Users/pam/Documents/results/Davino/helicopter"],
    ["/Users/pam/Documents/data/Davino_yuv/", "05_HEN_f.yuv", "/Users/pam/Documents/results/Davino/hen"],
    ["/Users/pam/Documents/data/Davino_yuv/", "06_LION_f.yuv", "/Users/pam/Documents/results/Davino/lion"],
    ["/Users/pam/Documents/data/Davino_yuv/", "07_UFO_f.yuv", "/Users/pam/Documents/results/Davino/ufo"],
    ["/Users/pam/Documents/data/Davino_yuv/", "08_TREE_f.yuv", "/Users/pam/Documents/results/Davino/tree"],
    ["/Users/pam/Documents/data/Davino_yuv/", "09_GIRL_f.yuv", "/Users/pam/Documents/results/Davino/girl"],
    ["/Users/pam/Documents/data/Davino_yuv/", "10_DOG_f.yuv", "/Users/pam/Documents/results/Davino/dog"],
]


yuvfileslist_realisticImages =[
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_0.yuv", "/Users/pam/Documents/results/realisticTampering/0"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_1.yuv", "/Users/pam/Documents/results/realisticTampering/1"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_2.yuv", "/Users/pam/Documents/results/realisticTampering/2"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_3.yuv", "/Users/pam/Documents/results/realisticTampering/3"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_4.yuv", "/Users/pam/Documents/results/realisticTampering/4"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_5.yuv", "/Users/pam/Documents/results/realisticTampering/5"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_6.yuv", "/Users/pam/Documents/results/realisticTampering/6"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_7.yuv", "/Users/pam/Documents/results/realisticTampering/7"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_8.yuv", "/Users/pam/Documents/results/realisticTampering/8"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_9.yuv", "/Users/pam/Documents/results/realisticTampering/9"],
    ["/Users/pam/Documents/data/realisticTampering/", "all_1080p_10.yuv", "/Users/pam/Documents/results/realisticTampering/10"],
]

yuvfileslist_VTD=[
    ["/Users/pam/Documents/data/VTD_yuv", "archery_f.yuv", "/Users/pam/Documents/results/VTD/archery"],
    ["/Users/pam/Documents/data/VTD_yuv", "cctv_f.yuv", "/Users/pam/Documents/results/VTD/cctv"],
    ["/Users/pam/Documents/data/VTD_yuv", "studio_f.yuv", "/Users/pam/Documents/results/VTD/studio"],
    ["/Users/pam/Documents/data/VTD_yuv", "swann_f.yuv", "/Users/pam/Documents/results/VTD/swann"],
    ["/Users/pam/Documents/data/VTD_yuv", "carpark_f.yuv", "/Users/pam/Documents/results/VTD/carpark"],
    ["/Users/pam/Documents/data/VTD_yuv", "bowling_f.yuv", "/Users/pam/Documents/results/VTD/bowling"],
    ["/Users/pam/Documents/data/VTD_yuv", "dahua_f.yuv", "/Users/pam/Documents/results/VTD/dahua"],
    ["/Users/pam/Documents/data/VTD_yuv", "clarity_f.yuv", "/Users/pam/Documents/results/VTD/clarity"],
    ["/Users/pam/Documents/data/VTD_yuv", "kitchen_f.yuv", "/Users/pam/Documents/results/VTD/kitchen"],
    ["/Users/pam/Documents/data/VTD_yuv", "basketball_f.yuv", "/Users/pam/Documents/results/VTD/basketball"],
    ["/Users/pam/Documents/data/VTD_yuv", "billiards_f.yuv", "/Users/pam/Documents/results/VTD/billiards"],
    ["/Users/pam/Documents/data/VTD_yuv", "bullet_f.yuv", "/Users/pam/Documents/results/VTD/bullet"],
    ["/Users/pam/Documents/data/VTD_yuv", "cake_f.yuv", "/Users/pam/Documents/results/VTD/cake"],
    ["/Users/pam/Documents/data/VTD_yuv", "camera_f.yuv", "/Users/pam/Documents/results/VTD/camera"],
    ["/Users/pam/Documents/data/VTD_yuv", "carplate_f.yuv", "/Users/pam/Documents/results/VTD/carplate"],
    ["/Users/pam/Documents/data/VTD_yuv", "cuponk_f.yuv", "/Users/pam/Documents/results/VTD/cuponk"],
    ["/Users/pam/Documents/data/VTD_yuv", "football_f.yuv", "/Users/pam/Documents/results/VTD/football"],
    ["/Users/pam/Documents/data/VTD_yuv", "highway_f.yuv", "/Users/pam/Documents/results/VTD/highway"],
    ["/Users/pam/Documents/data/VTD_yuv", "manstreet_f.yuv", "/Users/pam/Documents/results/VTD/manstreet"],
    ["/Users/pam/Documents/data/VTD_yuv", "passport_f.yuv", "/Users/pam/Documents/results/VTD/passport"],
    ["/Users/pam/Documents/data/VTD_yuv", "plane_f.yuv", "/Users/pam/Documents/results/VTD/plane"],
    ["/Users/pam/Documents/data/VTD_yuv", "pong_f.yuv", "/Users/pam/Documents/results/VTD/pong"],
    ["/Users/pam/Documents/data/VTD_yuv", "swimming_f.yuv", "/Users/pam/Documents/results/VTD/swimming"],
    ["/Users/pam/Documents/data/VTD_yuv", "whitecar_f.yuv", "/Users/pam/Documents/results/VTD/whitecar"],
    ["/Users/pam/Documents/data/VTD_yuv", "yellowcar_f.yuv", "/Users/pam/Documents/results/VTD/yellowcar"],
]

yuvfileslist_VTD2 = [
    ["/Users/pam/Documents/data/VTD_yuv", "audirs7_f.yuv", "/Users/pam/Documents/results/VTD/audirs7"],
]

yuvfileslist_theTestSet = [
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_35", "flower_cif_q35.yuv", "/Users/pam/Documents/results/testSet/flower_q35"],
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_35", "tempete_cif_q35.yuv", "/Users/pam/Documents/results/testSet/tempete_q35"],
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_14", "flower_cif_q14.yuv", "/Users/pam/Documents/results/testSet/flower_q14"],
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_14", "tempete_cif_q14.yuv", "/Users/pam/Documents/results/testSet/tempete_q14"],
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_0", "flower_cif_q0.yuv", "/Users/pam/Documents/results/testSet/flower_q0"],
    ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_0", "tempete_cif_q0.yuv", "/Users/pam/Documents/results/testSet/tempete_q0"],
]
justOne = [
    ["/Users/pam/Documents/data/VTD_yuv", "basketball_f.yuv", "/Users/pam/Documents/results/VTD/basketball"],
    ["/Users/pam/Documents/data/VTD_yuv", "cctv_f.yuv", "/Users/pam/Documents/results/VTD/cctv"],

]
justOne = [
    ["/Users/pam/Documents/data/Davino_yuv/", "08_TREE_f.yuv", "/Users/pam/Documents/results/Davino/tree"]
]

def createFileList(srcDir="/Volumes/LaCie/data/yuv_testOnly/CompAndReComp", resultsDir="/Users/pam/Documents/results/Comp"):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith("yuv") or filename.endswith("avi"):
                baseFileName, ext = os.path.splitext(filename)
                r = os.path.join(resultsDir, baseFileName)
                tuple = [srcDir,filename,r]
                fileList.append(tuple)
    return fileList

# One specifically for FaceForensics where the same file name is shared across altered, mask, original folders
def createFileList2(srcDir="/Volumes/LaCie/data/yuv_testOnly/CompAndReComp", resultsDir="/Users/pam/Documents/results/Comp"):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            #print(os.path.join(dirName, filename))
            #if filename.endswith("yuv") or filename.endswith("avi"):
            if filename.endswith("avi"):
                baseFileName, ext = os.path.splitext(filename)
                p, set = os.path.split(dirName)
                p, state = os.path.split(p)
                if set == "mask":
                    continue
                resultsFolder = "{}_{}_{}".format(baseFileName, state, set)
                r = os.path.join(resultsDir, resultsFolder)
                tuple = [dirName,filename,r]
                fileList.append(tuple)
    return fileList

recomp_1 = [
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.05_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.05_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.1_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.1_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.2_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.2_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q0.5_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q0.5_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0_q0.05.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0_q0.1.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0_q0.2.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0_q0.5.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'bus_cif_q1.0_q1.0.yuv', '/Users/pam/Documents/results/Comp/bus_cif_q1.0_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.05_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.05_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.1_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.1_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.2_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.2_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q0.5_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q0.5_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0_q0.05.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0_q0.1.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0_q0.2.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0_q0.5.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'flower_cif_q1.0_q1.0.yuv', '/Users/pam/Documents/results/Comp/flower_cif_q1.0_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.05_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.05_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.1_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.1_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.2_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.2_q1.0']
        ]
recomp = [
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q0.5_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q0.5_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_cif_q1.0_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_cif_q1.0_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.05_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.05_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.1_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.1_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.2_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.2_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q0.5_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q0.5_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0_q0.05.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0_q0.1.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0_q0.2.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0_q0.5.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'news_qcif_q1.0_q1.0.yuv', '/Users/pam/Documents/results/Comp/news_qcif_q1.0_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.05_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.05_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.1_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.1_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.2_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.2_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q0.5_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q0.5_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0_q0.05.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0_q0.05'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0_q0.1.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0_q0.1'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0_q0.2.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0_q0.2'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0_q0.5.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0_q0.5'],
          ['/Volumes/LaCie/data/yuv_testOnly/CompAndReComp', 'tempete_cif_q1.0_q1.0.yuv', '/Users/pam/Documents/results/Comp/tempete_cif_q1.0_q1.0']
          ]

def convertAVItoYUV(infilename):
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
    print("The dimensions are {}".format(dims))

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



def main(argv=None):  # pylint: disable=unused-argument


    resultsLog = open("resultsLog.txt", "w")
    resultsLog.write("file; tp; tn; fp; fn; mcc; f1; iouResult; frames")

    runAbunch = True
    faceForensics = createFileList2("/Users/pam/Documents/data/FaceForensics/FaceForensics_compressed",
                                   "/Users/pam/Documents/results/FaceForensics")
    print(faceForensics)

    if runAbunch:
        #for entry in yuvfileslist:
        for entry in faceForensics:
            FLAGS.data_dir = entry[0]
            FLAGS.yuvfile = entry[1]
            FLAGS.heatmap = entry[2]
            removeYUV = False
            if FLAGS.yuvfile.endswith(".avi"):
                completeFilename = os.path.join(FLAGS.data_dir, FLAGS.yuvfile)
                tempfilename = convertAVItoYUV(completeFilename)
                print("Converted to avi: {}".format(tempfilename))
                FLAGS.data_dir = "."
                FLAGS.yuvfile = tempfilename
                #removeYUV = True

            #for threshold in range(0, 8, 1):
            doEverything(resultsLog, threshold=0)
            if removeYUV:
                os.remove(FLAGS.yuvfile)
    else:
        doEverything(resultsLog, threshold=0)

if __name__ == '__main__':
    tf.app.run()
