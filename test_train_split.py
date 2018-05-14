# Written by Pam to try to get a test/train split

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil

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


def main_fromAU():
    test_dir = 'test'
    train_dir = 'train'
    testProportion = 0.1 # 10 percent for test
    testFilesList = []

    print('looking for file names like this:')
    print('Au' + os.sep + '*')
    print('looking.......')
    Au_pic_list = glob('Au' + os.sep + '*')
    Sp_pic_list = glob('Sp' + os.sep + '*')

    totalFiles = len(Au_pic_list) + len(Sp_pic_list)
    testFiles = int(totalFiles*testProportion)
    trainFiles = totalFiles - testFiles
    print('Splitting into test/train: {}/{}'.format(testFiles, trainFiles))

    while len(testFilesList) < testFiles:
        fn = random.choice(Au_pic_list)
        while fn in testFilesList:
            print("Already got {} selecting another".format(fn))
            fn = random.choice(Au_pic_list)
        #print(fn)
        testFilesList.append(fn)

        assocBgs = find_background(fn, Sp_pic_list)
        assocFgs = find_foreground(fn, Sp_pic_list)
        # don't copy duplicates
        for bg in assocBgs:
            if bg not in testFilesList:
                testFilesList.append(bg)
        for fg in assocFgs:
            if fg not in testFilesList:
                testFilesList.append(fg)

    #print(testFilesList)
    #Copy the listed files into test and the others into train
    for fn in testFilesList:
        d, n = os.path.split(fn)
        newName = os.path.join(test_dir, n)
        shutil.move(fn, newName)
        print("moved {}".format(fn))

    quit()


#for Au_pic in Au_pic_list:
#    backgrounds = find_background(Au_pic, Sp_pic_list)
#    splice_save(Au_pic, backgrounds, save_dir)

if __name__ == "__main__":
    test_dir = 'test'
    train_dir = 'train'
    testProportion = 0.1 # 10 percent for test
    testFilesList = []

    print('looking for file names like this:')
    print('Au' + os.sep + '*')
    print('looking.......')
    Au_pic_list = glob('Au' + os.sep + '*')
    Sp_pic_list = glob('Sp' + os.sep + '*')

    totalFiles = len(Au_pic_list) + len(Sp_pic_list)
    testFiles = int(totalFiles*testProportion)
    trainFiles = totalFiles - testFiles
    print('Splitting into test/train: {}/{}'.format(testFiles, trainFiles))

    while len(testFilesList) < testFiles:
        fn = random.choice(Sp_pic_list)
        while fn in testFilesList:
            print("Already got {} selecting another".format(fn))
            fn = random.choice(Sp_pic_list)
        #print(fn)
        testFilesList.append(fn)

        bg = find_donor_background(fn, Au_pic_list)
        fg = find_donor_foreground(fn, Au_pic_list)
        # don't copy duplicates
        if bg not in testFilesList:
            testFilesList.append(bg)
        if fg not in testFilesList:
            testFilesList.append(fg)

    #print(testFilesList)
    #Copy the listed files into test and the others into train
    for fn in testFilesList:
        d, n = os.path.split(fn)
        newName = os.path.join(test_dir, n)
        shutil.move(fn, newName)
        print("moved {}".format(fn))

    quit()

