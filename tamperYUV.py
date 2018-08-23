# Written by Pam to try to tamper a YUV image

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import cv2 as cv
import sys
import functions
import yuvview
from PIL import Image















if __name__ == "__main__":
    filename_mask_source = "/Volumes/LaCie/data/yuv/cif/news_cif.yuv"
    filename_background_source = "/Volumes/LaCie/data/yuv/cif/mobile_cif.yuv"
    #filename_mask_source = "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_withDeblock/quant_7/news_cif_q7.yuv"
    #filename_background_source = "/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_withDeblock/quant_7/mobile_cif_q7.yuv"
    filename_mask = "mask.yuv"
    filename_output = "munge.yuv"
    filename_output_ori = "original_q7.yuv"
    width = 352
    height = 288
    upperThresh = 255
    lowerThresh = 0
    kernelSize = 7
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    createMask = True



    if len(sys.argv) > 0:
        if len(sys.argv) < 3:
            print "***** Usage syntax Error!!!! *****\n"
            print "Usage:"
            print "filename_mask filename_bg width height filename_op"
            print "Using default:"
            print "tamperYUV {} {} {} {} {}".format(filename_mask_source, filename_background_source, width, height, filename_output)
            print " By the way, sequences must have equal heights and widths"
        else:
            filename_mask_source = sys.argv[1]
            filename_background_source = sys.argv[2]
            width = int(sys.argv[3])
            height = int(sys.argv[4])
            filename_output = sys.argv[5]

    # assume YUV4:2:0
    frameSize = int(width * height * 3 / 2)

    with open(filename_mask_source, "rb") as f:
        a_mask_source = np.fromfile(f, 'u1')

    with open(filename_background_source, "rb") as f:
        a_bg_source = np.fromfile(f, 'u1')



    numFrames = a_mask_source.shape[0] / frameSize
    a_mask_source = a_mask_source.reshape(numFrames, frameSize)

    #discard the first frames:
    a_mask_source = a_mask_source[1:, :]
    numFrames = numFrames-1



    numFrames_bg = a_bg_source.shape[0] / frameSize
    a_bg_source = a_bg_source.reshape(numFrames_bg, frameSize)
    if numFrames_bg > numFrames:
        a_bg_source = a_bg_source[0:numFrames, :]
    else:
        while numFrames_bg < numFrames:
            a_bg_source_mirror = np.flip(a_bg_source, 0)
            a_bg_source = np.append(a_bg_source, a_bg_source_mirror)
            numFrames_bg = a_bg_source.shape[0] / frameSize
        a_bg_source = a_bg_source[0:numFrames, :]



    if createMask:
        # frame diffs
        mask = np.diff(a_mask_source, axis=0)
        mask = np.diff(mask)

        diffMask = np.copy(a_mask_source)

        # clean up the mask

        #mask = mask[:,0:(height*width)] # Look at Y-channel only
        Uoffset = height*width
        uv = np.full((((width * height)/4)), 128)
        #greyuv444 = np.full(((width * height)), 128)
        uvheight = int(height/2)
        uvwidth = int(width/2)
        laplacian = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="int")  # from PyImageSearch

        fatten = np.full((5, 5), 1)

        mask[ mask[:, :] > upperThresh ] = 0
        mask[ mask[:, :] < lowerThresh ] = 0

        for i in range(0, (numFrames-1)):
            # single Y channel
            diff = mask[i, :Uoffset]
            diff = diff.reshape(height, width)
            diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel)
            diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
            ret, diff = cv.threshold(diff, 1, 255, cv.THRESH_BINARY)
            # Now apply watershed but only on the boundary regions
            # might need to do this as a convolution
            # mark the background as 1
            markers = diff
            forEdges = diff
            # find and fatten the boundary
            edges = cv.filter2D(forEdges, -1, laplacian)
            edges = cv.filter2D(edges, -1, fatten)
            ret, edges = cv.threshold(edges, 1, 255, cv.THRESH_BINARY)
            markers[markers[:, :] == 255] = 2
            markers[markers[:, :] == 0] = 1
            markers[edges[:, :] != 0] = 0

            # Here's a stupid piece of code - save it as a file then read back in
            img = a_mask_source[i, :Uoffset]
            img = img.reshape(height, width)
            img = np.append(img, np.append(img, img))
            rgbimg = functions.planarYUV_2_planarRGB(img, width, height)
            img = img.reshape((3, height, width)).astype(np.uint8)
            pictureA = img
            pictureA = np.swapaxes(pictureA, 0, 1)
            pictureA = np.swapaxes(pictureA, 1, 2)
            pictureA = np.ndarray.flatten(pictureA)
            imageA = Image.frombytes('RGB', (width, height), pictureA)
            imageA.save("img.png", "PNG")
            img = cv.imread('img.png') #read in as [h,w,c]

            # changing the type of markers is important
            markers = np.ascontiguousarray(markers, dtype=np.int32)
            img = np.ascontiguousarray(img, dtype=np.uint8)
            markers = cv.watershed(img, markers)
            markers[markers[:, :] == 255] = 0 # edges become background
            markers[markers[:, :] == 1] = 0 # 1 is background (but maybe this is different
            markers[markers[:, :] != 0] = 255 # 1 is background (but maybe this is different
            #print markers



            # and the UV - this might not be accurate....
            #uv = cv.resize(markers, (uvwidth, uvheight))
            uv = markers[::2, ::2]
            frame = np.append(markers, uv)
            frame = np.append(frame, uv)
            diffMask[i, :] = frame
            #quit()








        # save the mask as yuv
        datayuv = np.asarray(diffMask, 'u1')
        datayuv = datayuv.flatten()
        yuvByteArray = bytearray(datayuv)
        with open(filename_mask, "wb") as yuvFile:
            yuvFile.write(yuvByteArray)
    else: # read the mask from the file
        with open(filename_mask, "rb") as f:
            diffMask = np.fromfile(f, 'u1')
        diffMask = diffMask.reshape((numFrames, frameSize))

    # And now on to munging together the foreground and the background
    print "Mungey munge: apply the mask and munge together the foreground and background"

    mask = diffMask

    munge = a_mask_source.copy()
    munge[mask[:,:] == 0] = 0
    munge[mask[:,:] == 0] = a_bg_source[mask[:,:] == 0]

    diffdata = np.subtract(munge, a_mask_source)
    ret, diffdata = cv.threshold(diffdata, 1, 255, cv.THRESH_BINARY)


    # save the output as yuv
    datayuv = np.asarray(munge, 'u1')
    datayuv = datayuv.flatten()
    yuvByteArray = bytearray(datayuv)
    with open(filename_output, "wb") as yuvFile:
        yuvFile.write(yuvByteArray)


    # save the original as yuv (because you dumped that first frame...)
    datayuv = np.asarray(a_mask_source, 'u1')
    datayuv = datayuv.flatten()
    yuvByteArray = bytearray(datayuv)
    with open(filename_output_ori, "wb") as yuvFile:
        yuvFile.write(yuvByteArray)


    # save the diffdata
    datayuv = np.asarray(diffdata, 'u1')
    datayuv = datayuv.flatten()
    yuvByteArray = bytearray(datayuv)
    with open("diff.yuv", "wb") as yuvFile:
        yuvFile.write(yuvByteArray)

    savedFrameNo = 23
    # now save off a single frame
    yuvview.yuvFileTobmpFile(filename_output, width, height, savedFrameNo, "i420", "fig_uncomp_munge.bmp")
    yuvview.yuvFileTobmpFile(filename_output_ori, width, height, savedFrameNo, "i420", "fig_uncomp_ori.bmp")
    yuvview.yuvFileTobmpFile(filename_mask, width, height, savedFrameNo, "i420", "fig_uncomp_mask.bmp", bw=True)
    yuvview.yuvFileTobmpFile("diff.yuv", width, height, savedFrameNo, "i420", "fig_uncomp_diff.bmp", bw=True)

    #compress the munged file
    qp=384000
    filename_outcomp = "c_munge.264"
    filename_outdecomp = "c_munge_qp{}.yuv".format(qp)
    functions.compressFile("x264", filename_output, width, height, qp, filename_outcomp, filename_outdecomp)
    # and the original file
    filename_outcomp_ori = "c_ori.264"
    filename_outdecomp_ori = "c_ori_qp{}.yuv".format(qp)
    functions.compressFile("x264", filename_output_ori, width, height, qp, filename_outcomp_ori, filename_outdecomp_ori)
    # and the mask
    filename_outcomp_mask = "c_mask.264"
    filename_outdecomp_mask = "c_mask_qp{}.yuv".format(qp)
    functions.compressFile("x264", filename_mask, width, height, qp, filename_outcomp_mask, filename_outdecomp_mask)

    with open(filename_outdecomp, "rb") as f:
        munge_c = np.fromfile(f, 'u1')
    munge_c = munge_c.reshape((numFrames, frameSize))

    with open(filename_outdecomp_ori, "rb") as f:
        a_mask_source_c = np.fromfile(f, 'u1')
    a_mask_source_c = a_mask_source_c.reshape((numFrames, frameSize))

    diffdata_c = np.subtract(munge_c, a_mask_source_c)
    ret, diffdata_c = cv.threshold(diffdata_c, 1, 255, cv.THRESH_BINARY)
    # save the diffdata
    datayuv = np.asarray(diffdata_c, 'u1')
    datayuv = datayuv.flatten()
    yuvByteArray = bytearray(datayuv)
    with open("diff_c.yuv", "wb") as yuvFile:
        yuvFile.write(yuvByteArray)

    # now save off a single frame (the compressed images
    yuvview.yuvFileTobmpFile(filename_outdecomp, width, height, savedFrameNo, "i420", "fig_comp_munge_q{}.bmp".format(qp))
    yuvview.yuvFileTobmpFile(filename_outdecomp_ori, width, height, savedFrameNo, "i420", "fig_comp_oriq{}.bmp".format(qp))
    yuvview.yuvFileTobmpFile(filename_outdecomp_mask, width, height, savedFrameNo, "i420", "fig_comp_maskq{}.bmp".format(qp), bw=True)
    yuvview.yuvFileTobmpFile("diff_c.yuv", width, height, savedFrameNo, "i420", "fig_comp_diffq{}.bmp".format(qp), bw=True)
