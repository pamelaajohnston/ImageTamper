
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
#import liftedTFfunctions
import patchIt2 as pi
import functions

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 56392, """Number of examples to run.""")
tf.app.flags.DEFINE_integer('network_architecture', 1, """The number of the network architecture to use (inference function) """)
tf.app.flags.DEFINE_string('mylog_dir_eval', '/Users/pam/Documents/temp/',  """Directory where to write my logs """)
tf.app.flags.DEFINE_string('yuvfile', '/Users/pam/Documents/data/testyuv/mobile_cif_q7.yuv',  """The input yuv 4:2:0 video file """)
tf.app.flags.DEFINE_string('heatmap', 'heatmap.yuv',  """The heatmap yuv 4:2:0 video file """)
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
                #print("Step: {} and predictions: \n {}".format(step, batchPredictions))
                #print("Step: {} and logits: \n {}".format(step, batchMypreds))
                batchConfusionMatrix = batchConfusionMatrix.reshape((itNet_input.NUM_CLASSES, itNet_input.NUM_CLASSES))
                if step == 0:
                    #confusionMatrix = np.asarray(tf.unstack(batchConfusionMatrix))
                    confusionMatrix = batchConfusionMatrix
                    allPredLabels = batchMypreds
                else:
                    confusionMatrix = np.add(confusionMatrix, batchConfusionMatrix)
                    allPredLabels = np.append(allPredLabels, batchMypreds)
                #numRight = np.sum(predictions)
                #print("test step {} of {} ".format(step, num_iter))
                #print("We got {} correct".format(numRight))
                #print("Here's the batchCM:\n {}".format(batchConfusionMatrix))
                #print("Here's the cm:\n {}".format(confusionMatrix))
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            #print('For calculating precision: {} correct out of {} samples'.format(true_count, total_sample_count))
            #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            #print("Unstacking the confusion matrix")
            #cm = tf.unstack(confusionMatrix)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            #print("Predicted labels for {}: \n {}".format(step, allPredLabels))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        #return precision
        return precision, confusionMatrix, allPredLabels



def confusion_matrix(labels, predictions):
    tf.confusion_matrix(labels, predictions)


def evaluate(returnConfusionMatrix=True, filename="", numPatches=100, predFilename="pred.csv"):
    num_classes = itNet_input.NUM_CLASSES

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        #print("getting input data {}".format(FLAGS.eval_data))
        eval_data = FLAGS.eval_data == 'test'
        images, labels = itNet.inputs(eval_data=True, singleThreaded=True, filename=filename, numExamplesToTest=numPatches)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #print("calling inference")
        logits = itNet.inference_switch(images, FLAGS.network_architecture)
        predictions = tf.argmax(logits, 1)
        gen_confusionMatrix = tf.confusion_matrix(labels, predictions, num_classes=num_classes)
    
        # Calculate predictions.
        #print("calculating predictions")
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
        # Restore the moving average version of the learned variables for eval.
        #print("restore moving average version of learned variables")
        variable_averages = tf.train.ExponentialMovingAverage(itNet.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        # Build the summary operation based on the TF collection of Summaries.
        #print("Building a summary")
        summary_op = tf.summary.merge_all()
                                                          
        #print("And a summary writer")
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        logfile = os.path.join(FLAGS.mylog_dir_eval, "log_evals.txt")
        log = open(logfile, 'w')
        start = datetime.now()

        #print("Calling eval_once")
        precision, confusionMatrix, predLabels = eval_once(saver, summary_writer, top_k_op, summary_op, gen_confusionMatrix, predictions)
        #precision = eval_once(saver, summary_writer, top_k_op, summary_op)

        np.set_printoptions(threshold='nan')
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        print('{}: confusionMatrix: \n {}'.format(datetime.now(), confusionMatrix))
        #print('AND predictions (in order provided you only used one thread!!!): \n ')
        #print(np.array2string(predLabels, separator=', '))
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
        #log.write('confusionMatrix: \n {} \n'.format(confusionMatrix))
        cmString = np.array2string(confusionMatrix, separator='\t')
        cmString = cmString.replace('[', '')
        cmString = cmString.replace(']', '')
        log.write('confusionMatrix: \n {} \n'.format(cmString))

        #log.write("******************************************************* \n")
        log.flush()

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # We'll create the patches locally and save them.
    myDataDirName = "temp"
    myHeatmapFileName = FLAGS.heatmap
    if tf.gfile.Exists(myDataDirName):
        tf.gfile.DeleteRecursively(myDataDirName)
    tf.gfile.MakeDirs(myDataDirName)
    if tf.gfile.Exists(FLAGS.heatmap):
        os.remove(FLAGS.heatmap)

    inFilename = os.path.join(FLAGS.data_dir, FLAGS.yuvfile)
    outFilename = os.path.join(myDataDirName, "test.bin")
    predFilename = os.path.join(myDataDirName, "pred.csv")

    width, height = pi.getDimsFromFileName(inFilename)

    cropDim=FLAGS.cropDim
    cropTempStep=FLAGS.cropTempStep
    cropSpacStep=FLAGS.cropSpacStep
    num_channels=3
    bit_depth=8
    label="qp"

    #pi.patchOneFile(fileIn=inFilename, fileOut=outFilename, label="qp",
    #                cropDim=80, cropTempStep=1, cropSpacStep=16, num_channels=3, bit_depth=8)

    numPatches = pi.patchOneFile(fileIn=inFilename, fileOut=outFilename, label=label,
                                 cropDim=cropDim, cropTempStep=cropTempStep, cropSpacStep=cropSpacStep,
                                 num_channels=num_channels, bit_depth=bit_depth
                                 )



    #print("Data dir: {}".format(FLAGS.data_dir))
    #print("Batches dir: {}".format(FLAGS.batches_dir))
    #print("Eval dir: {}".format(FLAGS.eval_dir))
    print("Checkpoint dir: {}".format(FLAGS.checkpoint_dir))
    print("Network Architecture: {}".format(FLAGS.network_architecture))
    print("YUV file: {}".format(FLAGS.yuvfile))
    print("heatmap file: {}".format(FLAGS.heatmap))
    print("Testing {} patches".format(numPatches))



    FLAGS.num_examples = numPatches
    print("Evaluating on network {}".format(FLAGS.checkpoint_dir))
    evaluate(filename=outFilename, numPatches=numPatches, predFilename=predFilename)
    print("Done evaluation")
    #precision, confusionMatrix, predLabels = evaluate()

    # Now generate the heatmap from the preds in "pred.csv"
    print("Generating heatmap")
    predVals = np.loadtxt(predFilename)
    predVals = predVals[0:numPatches]

    # This is ok for a single frame but not for multiples
    frameSize = width * height
    fileBytes = os.path.getsize(inFilename)
    numFrames = int((fileBytes // (frameSize*3/2))//cropTempStep)
    predsWidth = (width - cropDim) // cropSpacStep
    predsHeight = (height - cropDim) // cropSpacStep
    predsPerFrame = predVals.shape[0] // numFrames
    #print(predVals.reshape((predsHeight, predsWidth)))
    print("numFrames {} predsWidth {} predsHeight {} predsPerFrame {}".format(numFrames, predsWidth, predsHeight, predsPerFrame))

    # This is because we're using 8 labels.
    multiplier = 256//8
    predVals = predVals*multiplier
    padding = cropDim//2
    uv = np.full((frameSize//2), 128)

    heatmap = []
    for f in range(0, numFrames):
        predsStart = f*predsPerFrame
        predsEnd = predsStart + predsPerFrame
        framePreds = predVals[predsStart:predsEnd]
        framePreds = framePreds.reshape((predsHeight,predsWidth))
        framePreds = framePreds.repeat(cropSpacStep, axis=0).repeat(cropSpacStep, axis=1)
        framePreds = np.pad(framePreds, ((padding, padding), (padding, padding)), 'edge')
        functions.appendToFile(framePreds, myHeatmapFileName)
        functions.appendToFile(uv, myHeatmapFileName)

    print("Done generating heatmap")

    print("The shape of the YUV: {}".format(predVals.shape))
    #uv = np.full(heatmap.shape, 128)
    #heatmap = np.append(heatmap, uv)
    #heatmap = np.append(heatmap, uv)
    if tf.gfile.Exists(myDataDirName):
        tf.gfile.DeleteRecursively(myDataDirName)
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)


if __name__ == '__main__':
  tf.app.run()
