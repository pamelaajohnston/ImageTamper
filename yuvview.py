from PIL import Image
import sys
from struct import *
import array

# import the necessary packages
from skimage.measure import structural_similarity as structural_similarity
import matplotlib.pyplot as plt
import numpy as np
#import cv2

from ssim.ssimlib import SSIM
from ssim.ssimlib import SSIMImage
from ssim.ssimlib import get_gaussian_kernel
from ssim.compat import Image


def yuvFileTobmpFile (yuvFileName, width, height, framenum, format, bmpFileName, bw=False):
    y = array.array('B')
    u = array.array('B')
    v = array.array('B')

    f_y = open(yuvFileName, "rb")
    f_uv = open(yuvFileName, "rb")

    image_out = Image.new("RGB", (width, height))
    pix = image_out.load()

    #print "width=", width, "height=", height, "framenum=", framenum

    if (format == "nv12"): #nv12 - who uses this?
        framesize = height * width * 1.5
        bytes = framesize * framenum
        f_y.seek(bytes, 1)
        f_uv.seek(bytes + (width*height), 1)
        for i in range(0, height/2):
            for j in range(0, width/2):
                u.append(ord(f_uv.read(1)));
                v.append(ord(f_uv.read(1)));
    else: #i420 - it rocks
        framesize = height * width * 1.5
        bytes = framesize * framenum
        f_y.seek(bytes, 1)
        f_uv.seek(bytes + (width*height), 1)
        for i in range(0, height/2):
            for j in range(0, width/2):
                u.append(ord(f_uv.read(1)));
        for i in range(0, height/2):
            for j in range(0, width/2):
                v.append(ord(f_uv.read(1)));

    for i in range(0,height):
        for j in range(0, width):
            y.append(ord(f_y.read(1)));
            #print "i=", i, "j=", j , (i*width), ((i*width) +j)
            #pix[j, i] = y[(i*width) +j], y[(i*width) +j], y[(i*width) +j]
            Y_val = y[(i*width)+j]
            U_val = u[((i/2)*(width/2))+(j/2)]
            V_val = v[((i/2)*(width/2))+(j/2)]
            if bw:
                U_val = 128
                V_val = 128
            #B = 1.164 * (Y_val-16) + 2.018 * (U_val - 128)
            #G = 1.164 * (Y_val-16) - 0.813 * (V_val - 128) - 0.391 * (U_val - 128)
            #R = 1.164 * (Y_val-16) + 1.596*(V_val - 128)
            #R = Y_val + 1.4075 * (V_val - 128)
            #G = Y_val - 0.3455 * (U_val - 128) - (0.7169 * (V_val - 128))
            #B = Y_val + 1.7790 * (U_val - 128)
            
            ## These values match those used in "videoCreator" and are SD YUV from wikipedia(!)
            R = Y_val + 1.13983 * (V_val - 128)
            G = Y_val - 0.39465 * (U_val - 128) - (0.58060 * (V_val - 128))
            B = Y_val + 2.03211 * (U_val - 128)
            pix[j, i] = int(R), int(G), int(B)

    if bmpFileName:
        image_out.save(bmpFileName)
    return image_out

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    #print "the shape "
    #print np.shape(imageA)
    #print np.size(imageA)
    
    divider = np.prod(np.shape(imageA))
    #err /= float(np.shape(imageA)[0] * np.shape(imageA)[1])
    err /= float(divider)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def psnr(imageA, imageB):
    myMSE = mse(imageA, imageB)
    #print ("mse = {}".format(myMSE))
    temp = 255 / (np.sqrt(myMSE))
    psnr = 20 * np.log10(temp)
    return psnr

def ssim(imageA, imageB):
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    return SSIM(imageA, gaussian_kernel_1d).ssim_value(imageB)

def compareImages(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    dataA = np.array(list(imageA.getdata()))
    dataB = np.array(list(imageB.getdata()))
    #m = mse(dataA, dataB)
    m = psnr(dataA, dataB)
    s = ssim(imageA, imageB)
    
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("PSNR: %.2f, SSIM: %.2f" % (m, s))
    
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    
    # show the images
    plt.show()

def display_image_rgb(data, width, height):
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    myshape = picture.shape
    picture = np.swapaxes(picture,0,1)
    picture = np.swapaxes(picture,1,2)
    plt.imshow(picture)
    plt.axis("off")
    plt.show()

def display_image_yuv(data, width, height):
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    myshape = picture.shape
    picture = picture[0]
    #plt.imshow(picture, cmap="hot")
    plt.imshow(picture, cmap="gray")
    plt.axis("off")
    plt.show()




