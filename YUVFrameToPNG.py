from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import functions
import time
import patchIt2 as pi

def savePic(saveName, rgbpic, height, width, border=False):
    test = np.asarray(rgbpic, 'u1')
    pictureA = test.reshape(3, height, width)
    pictureA = np.swapaxes(pictureA,0,1)
    pictureA = np.swapaxes(pictureA,1,2)
    pictureA = np.ndarray.flatten(pictureA)

    imageA = Image.frombytes('RGB', (height, width), pictureA)
    #imageB = imageA.resize(( (height*4), (width*4)), PIL.Image.LANCZOS)
    #imageB = imageA.resize(( (height*4), (width*4)))

    if border:
        imageA = imageA.crop((-4, -4, height+4, width+4))
        imageB = imageB.crop((-4, -4, (height*4)+4, (width*4)+4))

    #display(imageA)
    imageA.save(saveName, "PNG")
    #bigSaveName = "{}_big.png".format(saveName.replace('.png', ''))
    #imageB.save(bigSaveName, "PNG")

if __name__ == "__main__":

    print("Converting a .yuv and a file number to PNG")

    infilename = "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_noDeblock_test/quant_35/flower_cif_q35.yuv"
    infilename = "/Users/pam/Documents/results/testSet/flower_cif_q35/qp.yuv"
    infilename = "/Users/pam/Documents/data/Davino_yuv/08_TREE_r.yuv"
    infilename = "/Users/pam/Documents/results/Davino/tree/clusters_1280x720.yuv"
    infilename = "/Users/pam/Documents/results/Davino/tree/gt_blocked_1280x720.yuv"
    infilename = "/Users/pam/Documents/results/Davino/tree/qp_1280x720.yuv"
    frameNumber = 0
    outfilename = "tree_qp.png"

    entries = [
        ["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
        ["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_159.png", 159, 1280, 720],
        ["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_160.png", 160, 1280, 720],
        ["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_161.png", 161, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/qp.yuv", "fig_basketball_qp_158.png", 158, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/qp.yuv", "fig_basketball_qp_159.png", 159, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/qp.yuv", "fig_basketball_qp_160.png", 160, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/qp.yuv", "fig_basketball_qp_161.png", 161, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/ip.yuv", "fig_basketball_ip_158.png", 158, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/ip.yuv", "fig_basketball_ip_159.png", 159, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/ip.yuv", "fig_basketball_ip_160.png", 160, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/ip.yuv", "fig_basketball_ip_161.png", 161, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/deblock.yuv", "fig_basketball_deblock_158.png", 158, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/deblock.yuv", "fig_basketball_deblock_159.png", 159, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/deblock.yuv", "fig_basketball_deblock_160.png", 160, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/deblock.yuv", "fig_basketball_deblock_161.png", 161, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/frameDiffs.yuv", "fig_basketball_frameDiffs_158.png", 158, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/frameDiffs.yuv", "fig_basketball_frameDiffs_159.png", 159, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/frameDiffs.yuv", "fig_basketball_frameDiffs_160.png", 160, 1280, 720],
        ["/Users/pam/Documents/results/VTD/basketball/frameDiffs.yuv", "fig_basketball_frameDiffs_161.png", 161, 1280, 720],

    ]

    entries = [
        ["/Users/pam/Documents/data/Davino_yuv/05_HEN_r.yuv", "fig_hen_real.png", 0, 1280, 720],
        ["/Users/pam/Documents/data/Davino_yuv/05_HEN_f.yuv", "fig_hen_fake.png", 0, 1280, 720],
        ["/Users/pam/Documents/results/Davino/hen/gt_blocked.yuv", "fig_hen_gt.png", 0, 1280, 720],
        ["/Users/pam/Documents/results/Davino/hen/clusters.yuv", "fig_hen_clusters.png", 0, 1280, 720],
        ["/Users/pam/Documents/results/Davino/hen/qp.yuv", "fig_hen_qp.png", 0, 1280, 720],
        ["/Users/pam/Documents/data/Davino_yuv/06_LION_r.yuv", "fig_lion_real.png", 187, 1280, 720],
        ["/Users/pam/Documents/data/Davino_yuv/06_LION_f.yuv", "fig_lion_fake.png", 187, 1280, 720],
        ["/Users/pam/Documents/results/Davino/lion/gt_blocked.yuv", "fig_lion_gt.png", 187, 1280, 720],
        ["/Users/pam/Documents/results/Davino/lion/clusters.yuv", "fig_lion_clusters.png", 187, 1280, 720],
        ["/Users/pam/Documents/results/Davino/lion/qp.yuv", "fig_lion_qp.png", 187, 1280, 720],
    ]

    entries = [
        ["/Users/pam/Documents/results/Davino/hen/clusters.yuv", "fig_hen_clusters.png", 0, 1280, 720],
    ]
    #invert=True
    entries = [
        ["/Users/pam/Documents/data/Davino_yuv/08_TREE_r.yuv", "fig_tree_r.png", 10, 1280, 720],
        ["/Users/pam/Documents/data/Davino_yuv/08_TREE_f.yuv", "fig_tree_f.png", 10, 1280, 720],
        ["/Users/pam/Documents/data/Davino_yuv/08_TREE_mask.yuv", "fig_tree_mask.png", 10, 1280, 720],
    ]

    entries = [
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/mobile_cif_qp0.yuv", "fig_mobile_qp0.png", 0, 352, 288],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/mobile_cif_qp14.yuv", "fig_mobile_qp14.png", 0, 352, 288],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/mobile_cif_qp28.yuv", "fig_mobile_qp28.png", 0, 352, 288],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/mobile_cif_qp42.yuv", "fig_mobile_qp42.png", 0, 352, 288],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/tempete_qp42.yuv", "fig_tempete_qp42_194.png", 194, 352, 288],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/tempete_cif.yuv", "fig_tempete_cif_194.png", 194, 352, 288],
    ]
    entries = [
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/pretty_352x352_qp0.yuv", "pretty_qp0.png", 0, 352, 352],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/pretty_352x352_qp14.yuv", "pretty_qp14.png", 0, 352, 352],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/pretty_352x352_qp28.yuv", "pretty_qp28.png", 0, 352, 352],
        ["/Users/pam/Documents/MyWriting/thesisDraw.io/compression/pretty_352x352_qp42.yuv", "pretty_qp42.png", 0, 352, 352],
    ]

    entries = [
        ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed_firstSampleDownload/test/altered/1aJO2VkfZiY_2_EMLALfhSftA_0_640x480.yuv", "fig_ff_full_alt.png", 0, 640, 480],
        ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed_firstSampleDownload/test/original/1aJO2VkfZiY_2_EMLALfhSftA_0_640x480.yuv", "fig_ff_full_ori.png", 0, 640, 480],
        ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed_firstSampleDownload/test/altered/1aJO2VkfZiY_2_EMLALfhSftA_0_cropped_146x178.yuv", "fig_ff_crop_alt.png", 0, 146, 178],
        ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed_firstSampleDownload/test/original/1aJO2VkfZiY_2_EMLALfhSftA_0_cropped_146x178.yuv", "fig_ff_crop_ori.png", 0, 146, 178],
    ]

    invert=False

    for entry in entries:
        infilename = entry[0]
        outfilename = entry[1]
        frameNumber = entry[2]
        width, height = pi.getDimsFromFileName(infilename)
        if width == 0:
            width = entry[3]
            height = entry[4]
        frameSize = int(width * height * 3/2)
        print("Dimensions {} by {}".format(width, height))

        with open(infilename, "rb") as f:
            allbytes = np.fromfile(f, 'u1')

        numFrames = int(allbytes.shape[0] // frameSize)
        allbytes = allbytes.reshape((numFrames, frameSize))

        myFrame = allbytes[frameNumber, :]

        if invert:
            invertedFrame = np.zeros(myFrame.shape)
            invertedFrame[np.where(myFrame==0)] = 255
            invertedFrame[(width*height):] = 128
            myFrame = invertedFrame


        frame = functions.YUV420_2_YUV444(myFrame, height, width)
        rgbframe = functions.planarYUV_2_planarRGB(frame, height, width)
        savePic(outfilename, rgbframe, width, height)


    quit()
