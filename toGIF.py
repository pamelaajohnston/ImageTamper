import shutil
import os
import functions
import sys
from PIL import Image
import numpy as np





def prepareTempDir(dirName):
    if os.path.isdir(dirName):
        shutil.rmtree(dirName)
    os.makedirs(dirName)

def cleanUp(dirName):
    if os.path.isdir(dirName):
        shutil.rmtree(dirName)

def convertAVItoYUV(infilename):
    # first get the dimensions so they can go in the file name
    probeCmd = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {}".format(
        infilename)
    if sys.platform == 'win32':
        args = probeCmd
    else:
        args = shlex.split(probeCmd)

    # subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()
    dims = out.rstrip()
    #print("The dimensions are {}".format(dims))

    outfilename = "temp_{}.yuv".format(dims)
    #outfilename = infilename.replace(".avi", "_{}.yuv".format(dims))
    if os.path.isfile(outfilename):
        os.remove(outfilename)

    app = "ffmpeg"
    appargs = "-i {} -pix_fmt yuv420p {}".format(infilename, outfilename)

    exe = app + " " + appargs
    # print exe

    if sys.platform == 'win32':
        args = exe
    else:
        args = shlex.split(exe)

    # subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()

    return outfilename

def savePic(saveName, rgbpic, height, width, half=False, border=False):
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
    if half:
        halfh = height // 2
        halfw = width // 2
        imageA.resize((halfh, halfw), Image.LANCZOS)

    #display(imageA)
    imageA.save(saveName, "PNG")
    #bigSaveName = "{}_big.png".format(saveName.replace('.png', ''))
    #imageB.save(bigSaveName, "PNG")


def framesToPNGs(srcFile, srcNumber, width, height, start, end, dstDir, resize=False):
    print(srcFile)
    f, ext = os.path.splitext(srcFile)
    print("{}".format(ext))
    if ext == ".yuv":
        print("Processing YUV file")
        pngNames = YUVFramesToPNGs(srcFile, srcNumber, width, height, start, end, dstDir, resize)
    if ext == ".avi":
        print("AVI: convert to YUV first")
        yuvFile = convertAVItoYUV(srcFile)
        pngNames = YUVFramesToPNGs(yuvFile, srcNumber, width, height, start, end, dstDir, resize)
        os.remove(yuvFile)

    return pngNames


def YUVFramesToPNGs(srcFile, srcNumber, width, height, start, end, dstDir, resize=False):
    pngNames = []
    frameSize = int(width * height * 3 / 2)

    with open(srcFile, "rb") as f:
        allbytes = np.fromfile(f, 'u1')

    numFrames = int(allbytes.shape[0] // frameSize)
    allbytes = allbytes.reshape((numFrames, frameSize))
    print("Converting YUV to PNG files: {} frames".format(numFrames))
    print(allbytes.shape)


    pngNumber = 0
    for frameNumber in range(start, end):
        pngName = os.path.join(dstDir, "{}_{}.png".format(srcNumber, pngNumber))
        print("Save {}".format(pngName))

        myFrame = allbytes[frameNumber, :]
        frame = functions.YUV420_2_YUV444(myFrame, height, width)
        rgbframe = functions.planarYUV_2_planarRGB(frame, height, width)
        savePic(pngName, rgbframe, width, height, resize)
        pngNames.append(pngName)
        pngNumber = pngNumber + 1
    return pngNames

def joinImagesHorizontally(image1, image2, dstDir, srcNo, imgNo):
    images = map(Image.open, [image1, image2])
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    pngName = os.path.join(dstDir, "{}_{}.png".format(srcNo, imgNo))
    new_im.save(pngName)
    return pngName

def joinImagesVertically(image1, image2, dstDir, srcNo, imgNo):
    images = map(Image.open, [image1, image2])
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    pngName = os.path.join(dstDir, "{}_{}.png".format(srcNo, imgNo))
    new_im.save(pngName)
    return pngName

def joinImagesWithResize(list_im, outname, horizontal=True):
    # from https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    # Not actually used, but it was such cool code, I wanted a copy!
    #list_im = ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']
    imgs = [PIL.Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

    if horizontal:
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    else:
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))

    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(outname)

def binDiffYUV(srcFile, srcNumber, width, height, start, end, dstDir):
    pngNames = []
    pngNumber = 0
    frameSize = int(width * height * 3 / 2)
    ysize = width*height
    usize = (width*height)//4

    with open(srcFile, "rb") as f:
        allbytes = np.fromfile(f, 'u1')

    numFrames = int(allbytes.shape[0] // frameSize)
    allbytes = allbytes.reshape((numFrames, frameSize))

    for frameNumber in range(start, end-1):
        pngName = os.path.join(dstDir, "{}_{}.png".format(srcNumber, pngNumber))
        one = allbytes[frameNumber, :]
        two = allbytes[(frameNumber+1), :]
        diff = abs(one-two)

        #Here we do the binarising...!
        myFrame = np.copy(diff)
        myFrame[(width*height):] = 128
        #y = myFrame[0:(width*height)]
        #u = myFrame[ysize:(ysize+usize)]
        #v = myFrame[(ysize+usize):]


        frame = functions.YUV420_2_YUV444(myFrame, height, width)
        rgbframe = functions.planarYUV_2_planarRGB(frame, height, width)
        savePic(pngName, rgbframe, width, height)
        pngNames.append(pngName)
        pngNumber = pngNumber + 1

    return pngNames

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/dev/git/ImageTamper/z_predictedQP_640x480.yuv", 640, 480, 0, 65],
            ["/Users/pam/Documents/dev/git/ImageTamper/z_predictedQP_640x480.yuv", 640, 480, 66, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_qp_640x480.gif"
    ],
    [
        [
            ["/Users/pam/Documents/dev/git/ImageTamper/z_YUV_all_640x480.yuv", 640, 480, 0, 65],
            ["/Users/pam/Documents/dev/git/ImageTamper/z_YUV_all_640x480.yuv", 640, 480, 66, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_yuv_640x480.gif"
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/dev/git/ImageTamper/z_YUV_all_640x480.yuv", 640, 480, 0, 65],
            ["/Users/pam/Documents/dev/git/ImageTamper/z_predictedQP_640x480.yuv", 640, 480, 0, 65],
            ["/Users/pam/Documents/dev/git/ImageTamper/z_YUV_all_640x480.yuv", 640, 480, 66, 65],
            ["/Users/pam/Documents/dev/git/ImageTamper/z_predictedQP_640x480.yuv", 640, 480, 66, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_yuv_640x480.gif",
        "200x100"
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_alt_640x480.yuv", 640, 480, 0, 65],
            ["diff", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_blueDude_640x480.gif",
        "10x100"
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_alt_640x480.yuv", 640, 480, 0, 65],
            ["diff", 640, 480, 0, 65],
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_ori_640x480.yuv", 640, 480, 0, 65],
            ["diff", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_blueDude_640x480.gif",
        "10x100"
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_0/mobile_cif_q0.yuv", 352, 288, 65, 155],
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_14/mobile_cif_q14.yuv", 352, 288, 65, 155],
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_28/mobile_cif_q28.yuv", 352, 288, 65, 155],
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_42/mobile_cif_q42.yuv", 352, 288, 65, 155],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/mobile0_14_28_42_cif.gif",
        ""
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_0/mobile_cif_q0.yuv", 352, 288, 0, 60],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/mobile0_cif.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_14/mobile_cif_q14.yuv", 352,
             288, 0, 60],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/mobile14_cif.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_28/mobile_cif_q28.yuv", 352,
             288, 0, 60],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/mobile28_cif.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/YUV_x264_encoded/yuv_quant_intraOnly_noDeblock_train/quant_42/mobile_cif_q42.yuv", 352,
             288, 0, 60],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/mobile42_cif.gif",
        ""
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Volumes/LaCie/data/VTD_yuv/manstreet_r.yuv", 1280, 720, 0, 75],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/manstreet_r.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/VTD_yuv/manstreet_f.yuv", 1280, 720, 0, 75],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/manstreet_f.gif",
        ""
    ],
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Volumes/LaCie/data/Davino_yuv/05_HEN_f.yuv", 1280, 720, 0, 60],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/hen_f.gif",
        ""
    ],

]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_alt_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_blueDude_640x480_1.gif",
        ""
    ],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_alt_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_blueDude_640x480_1.gif",
        ""
    ],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/chineseWoman_alt_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_chineseWoman_640x480_1.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/altered/-PMjPTgYiuE_0_JKsfXX792AU_3_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_-PMjPTgYiuE_0_JKsfXX792AU_3_640x480.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/altered/-tutmgfU4k4_3_QLpd-5DxgGs_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_-tutmgfU4k4_3_QLpd-5DxgGs_0_640x480.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/altered/0r4uhJdcIQA_1_cpywXpZVP6o_6_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_0r4uhJdcIQA_1_cpywXpZVP6o_6_640x480.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/altered/1ouIl61HXpE_0_EkBGHtN3o34_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_1ouIl61HXpE_0_EkBGHtN3o34_0_640x480.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/altered/1vpfHf42UuI_0_bj6TvzHTczc_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/1vpfHf42UuI_0_bj6TvzHTczc_0_640x480.gif",
        ""
    ]
]

entries = [
    #["/Users/pam/Documents/data/VTD_yuv/basketball_f.yuv", "fig_basketball_f_158.png", 158, 1280, 720],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/blueDude_ori_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_blueDude_640x480_1_ori.gif",
        ""
    ],
    [
        [
            ["/Users/pam/Documents/data/FaceForensics/onesIlike/chineseWoman_ori_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_chineseWoman_640x480_1_ori.gif",
        ""
    ],
    [
        [
            ["/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/original/-PMjPTgYiuE_0_JKsfXX792AU_3_640x480.yuv", 640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_-PMjPTgYiuE_0_JKsfXX792AU_3_640x480_ori.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/original/-tutmgfU4k4_3_QLpd-5DxgGs_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_-tutmgfU4k4_3_QLpd-5DxgGs_0_640x480_ori.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/original/0r4uhJdcIQA_1_cpywXpZVP6o_6_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_0r4uhJdcIQA_1_cpywXpZVP6o_6_640x480_ori.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/original/1ouIl61HXpE_0_EkBGHtN3o34_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/z_ff_1ouIl61HXpE_0_EkBGHtN3o34_0_640x480_ori.gif",
        ""
    ],
    [
        [
            [
                "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test/original/1vpfHf42UuI_0_bj6TvzHTczc_0_640x480.yuv",
                640, 480, 0, 65],
        ],
        "/Users/pam/Documents/dev/git/ImageTamper/1vpfHf42UuI_0_bj6TvzHTczc_0_640x480_ori.gif",
        ""
    ]
]


if __name__ == "__main__":
    print("This is for converting files to a GIF, look at the code for inputs because I don't want to do a UI")
    myTempDir = "temp_gifs"
    andReverse = True
    half = True

    for entry in entries:
        # first set up
        prepareTempDir(myTempDir)
        sourceList = entry[0]
        gifName = entry[1]
        frameRate = entry[2]

        print("The source list:")
        print(sourceList)
        print("The gif name: {}".format(gifName))

        # first, extract all the frames from the source list as pngs
        allPNGs = []
        for srcNo, src in enumerate(sourceList):
            inputFile = src[0]
            width = src[1]  # type: int
            height = src[2]
            startFrame = src[3]
            numFrames = src[4]
            endFrame = startFrame + numFrames
            if inputFile == "diff":
                diffFile = sourceList[srcNo-1][0]
                print("Doing a binary frame diff for {}".format(diffFile))
                pngNames = binDiffYUV(diffFile, srcNo, width, height, startFrame, endFrame, myTempDir)
            else:
                pngNames = framesToPNGs(inputFile, srcNo, width, height, startFrame, endFrame, myTempDir, half)
            allPNGs.append(pngNames)

        if half:
            width = width//2
            height = height//2

        #How many sources do we have?
        numSrcs = len(allPNGs)
        gifImages = []
        if numSrcs == 2:
            joiningList = zip(allPNGs[0], allPNGs[1])
            gifImages = []
            for i, j in enumerate(joiningList):
                pngName = joinImagesHorizontally(j[0], j[1], myTempDir, "end", i)
                gifImages.append(pngName)
        if numSrcs == 1:
            gifImages = allPNGs[0]
        if numSrcs == 4:
            # tl, tr, bl, br is the order of the images
            joiningList = zip(allPNGs[0], allPNGs[1])
            topImages = []
            for i, j in enumerate(joiningList):
                pngName = joinImagesHorizontally(j[0], j[1], myTempDir, "top", i)
                topImages.append(pngName)
            joiningList = zip(allPNGs[2], allPNGs[3])
            botImages = []
            for i, j in enumerate(joiningList):
                pngName = joinImagesHorizontally(j[0], j[1], myTempDir, "bot", i)
                botImages.append(pngName)
            joiningList = zip(topImages, botImages)
            gifImages = []
            for i, j in enumerate(joiningList):
                pngName = joinImagesVertically(j[0], j[1], myTempDir, "end", i)
                gifImages.append(pngName)

        with open('image_list.txt', 'w') as file:
            for item in gifImages:
                file.write("%s\n" % item)
            if andReverse:
                print("Smooth forward and reverse")
                for item in reversed(gifImages):
                    file.write("%s\n" % item)

        if frameRate=="":
            os.system('convert @image_list.txt {}'.format(gifName))
        else:
            os.system('convert @image_list.txt -set delay {} {}'.format(frameRate, gifName))
            