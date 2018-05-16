# Written by Pam to try to get a test/train split

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import cv2 as cv
#import sys
#sys.path.append('/Users/pam/Documents/dev/git/cifar10/')
import functions

# these only work when the main directory has a 2-letter name.
au_index = [6, 9, 10, 14]
background_index = [14, 21]
foreground_index = [22, 29]

def find_background(Au_pic, Sp_pic_list):
    ### find spliced images with Au_pic as background
    # Au_pic: the path of an authentic image
    # Sp_pic_list: all paths of spliced images
    # result(return): a list of paths with Au_pic as background
    au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
    #print("The au_name: {}".format(au_name))
    backgrounds = []
    for spliced in Sp_pic_list:
        sp_name = spliced[background_index[0]:background_index[1]]
        #print("The sp_name: {}".format(sp_name))
        if au_name == sp_name:
            backgrounds.append(spliced)
    return backgrounds

def find_foreground(Au_pic, Sp_pic_list):
    ### find spliced images with Au_pic as background
    # Au_pic: the path of an authentic image
    # Sp_pic_list: all paths of spliced images
    # result(return): a list of paths with Au_pic as background
    au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
    foregrounds = []
    for spliced in Sp_pic_list:
        sp_name = spliced[foreground_index[0]:foreground_index[1]]
        if au_name == sp_name:
            foregrounds.append(spliced)
    return foregrounds

def find_donor_background(Sp_pic, Au_pic_list):
    # result(return): the name of the background file
    sp_name = Sp_pic[background_index[0]:background_index[1]]
    #print("sp_name: {}".format(sp_name))
    for Au_pic in Au_pic_list:
        au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
        #print("au_name: {}".format(au_name))
        if au_name == sp_name:
            return Au_pic
    return "null"

def find_donor_foreground(Sp_pic, Au_pic_list):
    # result(return): the name of the background file
    sp_name = Sp_pic[foreground_index[0]:foreground_index[1]]
    #print("sp_name: {}".format(sp_name))
    for Au_pic in Au_pic_list:
        au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
        #print("au_name: {}".format(au_name))
        if au_name == sp_name:
            return Au_pic
    return "null"

def splice_save(Au_pic, backgrounds, save_dir):
    # splice together Au_pic and each of backgrounds, and save it/them.
    # Au_pic: the path of an authentic image
    # backgrounds: list returned by `find_background`
    # save_dir: path to save the combined image
    au_image = plt.imread(Au_pic)
    for each in backgrounds:
        sp_image = plt.imread(each)
        if au_image.shape == sp_image.shape:
            result = np.concatenate((au_image, sp_image), 1)
            plt.imsave(save_dir+os.sep+each[14:], result)




#for Au_pic in Au_pic_list:
#    backgrounds = find_background(Au_pic, Sp_pic_list)
#    splice_save(Au_pic, backgrounds, save_dir)

if __name__ == "__main__":
    myDir = 'test_jpg'
    cropDir = 'test_crop'
    myDir = 'train_jpg'
    cropDir = 'train_crop'
    cropDim = 256
    cropStep = [0,64,128]

    datasetName = "{}.bin".format(cropDir)
    datasetList = []
    numPatches = 0
    failedList = []

    kernel = np.ones((5, 5), np.uint8)
    #kernel2 = np.ones((8, 8), np.uint8)
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))

    if len(myDir) > 2:
        adder = len(myDir) - 2
        au_index = [x + adder for x in au_index]
        background_index = [x + adder for x in background_index]
        foreground_index = [x + adder for x in foreground_index]

    Sp_pic_list = glob(myDir + os.sep + 'Sp' + '*')
    Au_pic_list = glob(myDir + os.sep + 'Au' + '*')
    #print(Sp_pic_list)
    #np.set_printoptions(threshold=np.nan)

    for Sp_pic in Sp_pic_list:
        Au_pic_bg = find_donor_background(Sp_pic, Au_pic_list)
        Au_pic_fg = find_donor_foreground(Sp_pic, Au_pic_list)
        #if Au_pic == "null":
        #    Au_pic = find_donor_foreground(Sp_pic, Au_pic_list)
        print("Spliced: {} from: {} and {}".format(Sp_pic, Au_pic_bg, Au_pic_fg))
        sp_image = plt.imread(Sp_pic)

        n, e = os.path.splitext(Sp_pic)
        maskName = "{}.mask.png".format(n)

        diff1 = 'null'
        if Au_pic_bg != 'null':
            au_image_bg = plt.imread(Au_pic_bg)
            if au_image_bg.shape == sp_image.shape:
                diff1 = sp_image - au_image_bg

        diff2 = 'null'
        if Au_pic_fg != 'null':
            au_image_fg = plt.imread(Au_pic_fg)
            if au_image_fg.shape == sp_image.shape:
                diff2 = sp_image - au_image_fg

        if diff1 == 'null' and diff2 == 'null':
            print("Odd, {} doesn't have a corresponding fg/bg in the same folder?".format(Sp_pic))
            continue
        elif diff1 == 'null':
            diff = diff2
            au_image = au_image_fg
        elif diff2 == 'null':
            diff = diff1
            au_image = au_image_bg
        else:
            # pick the smaller mask
            diff1Tot = np.sum(np.absolute(diff1))
            diff2Tot = np.sum(np.absolute(diff2))
            if diff1Tot > diff2Tot:
                diff = diff2
                au_image = au_image_fg
            else:
                diff = diff1
                au_image = au_image_bg



        #threshold to get the mask
        idx = diff[:, :, :] > 245
        diff[idx] = 0
        idx = diff[:, :, :] < 11
        diff[idx] = 0
        diff = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
        diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel)
        diff = cv.morphologyEx(diff, cv.MORPH_CLOSE, kernel2)
        ret, diff = cv.threshold(diff, 1, 255, cv.THRESH_BINARY)
        idxc = np.logical_or(diff[:, :, 1], diff[:, :, 0])
        idxc = np.logical_or(idxc, diff[:, :, 2]).astype(int)
        mask_1C = np.asarray(idxc)
        idxc = np.dstack((idxc,idxc,idxc)) #np.concatenate((idxc, idxc, idxc), axis=2).astype(int)
        plt.imsave(maskName, idxc)


        # Find a suitable crop: cropping to 256x256, crop longest dimension at 0 or 64 or 128 to give maximum mask
        print(mask_1C.shape)
        h,w,c = au_image.shape
        mask_1C = mask_1C.reshape(h,w)
        bestcS = cropStep[0]
        highestRes = 0
        for cS in cropStep:
            if h < w:  # landscape
                cropped = mask_1C[:,cS:(cropDim+cS)]
            else: # portrait
                cropped = mask_1C[cS:(cropDim+cS),:]
            res = np.count_nonzero(cropped != 0)
            print("For {} crop: shape {} result{}".format(cS, cropped.shape, res))
            if res > highestRes:
                highestRes = res
                bestcS = cS
        print(bestcS)

        #Do the crop and save the file
        p, n = os.path.split(Sp_pic)
        n, e = os.path.splitext(n)
        n = "{}.png".format(n)
        cropName = os.path.join(cropDir, n)
        p, n = os.path.split(maskName)
        cropMaskName = os.path.join(cropDir, n)
        print("Names: {}, {}".format(cropName, cropMaskName))
        if h < w:
            croppedImg = sp_image[:, bestcS:(cropDim+bestcS), :]
            croppedMask = idxc[:, bestcS:(cropDim+bestcS), :]
        else:
            croppedImg = sp_image[bestcS:(cropDim+bestcS), :, :]
            croppedMask = idxc[bestcS:(cropDim+bestcS), :, :]
        plt.imsave(cropName, croppedImg)
        plt.imsave(cropMaskName, croppedMask)

        th, tw, tc = croppedImg.shape
        if th != cropDim or tw != cropDim:
            print("Failed: {} vs {} or {} vs {}, ({})".format(th, cropDim, tw, cropDim, croppedImg.shape))
            failedList.append(cropName)
            continue

        # convert to YUV444 because that's my thing
        print(croppedImg.shape)
        croppedImg = np.swapaxes(croppedImg, 1,2)
        croppedImg = np.swapaxes(croppedImg, 0,1)
        #croppedImg = np.swapaxes(croppedImg, 1,2)
        print(croppedImg.shape)
        datargb = croppedImg.flatten()
        print(datargb.shape)
        datayuv = functions.planarRGB_2_planarYUV(datargb, cropDim, cropDim)
        print(datayuv.shape)
        label = 1
        datayuv = np.divide(datayuv, 8) # normalise
        datayuv = np.concatenate((np.array([label]), datayuv), axis=0)
        datayuv = datayuv.flatten()
        datasetList.append(datayuv)
        numPatches = numPatches + 1

    for Au_pic in Au_pic_list:
        print("Authentic: {} ".format(Au_pic))
        au_image = plt.imread(Au_pic)
        n, e = os.path.splitext(Au_pic)
        p, n = os.path.split(n)
        n = "{}.png".format(n)
        cropName = os.path.join(cropDir, n)

        h,w,c = au_image.shape
        bestcS = random.choice(cropStep)

        if h < w:
            croppedImg = au_image[:, bestcS:(cropDim+bestcS), :]
        else:
            croppedImg = au_image[bestcS:(cropDim+bestcS), :, :]
        plt.imsave(cropName, croppedImg)

        th, tw, tc = croppedImg.shape
        if th != cropDim or tw != cropDim:
            print("Failed: {} vs {} or {} vs {}, ({})".format(th, cropDim, tw, cropDim, croppedImg.shape))
            failedList.append(cropName)
            continue

        # convert to YUV444 because that's my thing
        croppedImg = np.swapaxes(croppedImg, 1,2)
        croppedImg = np.swapaxes(croppedImg, 0,1)
        #croppedImg = np.swapaxes(croppedImg, 1,2)
        datargb = croppedImg.flatten()
        datayuv = functions.planarRGB_2_planarYUV(datargb, 256, 256)
        label = 0
        datayuv = np.divide(datayuv, 8) # normalise
        datayuv = np.concatenate((np.array([label]), datayuv), axis=0)
        datayuv = datayuv.flatten()
        datasetList.append(datayuv)
        numPatches = numPatches + 1

        #quit()
    print("failed list {}".format(failedList))
    dataset_array = np.array(datasetList)
    print("Size of Dataset: {}".format(dataset_array.shape))
    np.random.shuffle(dataset_array)
    print("Size of Dataset: {}".format(dataset_array.shape))
    functions.appendToFile(dataset_array, datasetName)

    quit()
