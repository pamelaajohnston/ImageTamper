from PIL import Image
import random
import os
import shlex, subprocess
import yuvview
import numpy as np
import sys
import socket
#import readConfig

def convertTifToYUV(filename):
    im = Image.open(filename)

    # print("The baseFileName is: {}".format(baseFileName))
    width, height = im.size

    pixels = list(im.getdata())
    pixels = np.asarray(pixels, 'u1')
    pixels = pixels.reshape(height, width, 3)
    pixels = np.swapaxes(pixels, 2, 1)
    pixels = np.swapaxes(pixels, 1, 0)
    pixels = pixels.flatten()
    yuvpixels = planarRGB_2_planarYUV(pixels, height, width)
    return width, height, yuvpixels

def convertTifToYUV420(filename):
    im = Image.open(filename)

    # print("The baseFileName is: {}".format(baseFileName))
    width, height = im.size

    pixels = list(im.getdata())
    pixels = np.asarray(pixels, 'u1')
    pixels = pixels.reshape(height, width, 3)
    pixels = np.swapaxes(pixels, 2, 1)
    pixels = np.swapaxes(pixels, 1, 0)
    pixels = pixels.flatten()
    yuvpixels = planarRGB_2_planarYUV(pixels, height=height, width=width)
    yuvpixels420 = YUV444_2_YUV420_alt(yuvpixels, width=width, height=height)
    return width, height, yuvpixels420


def doubleImage(data, width, height):
    doubleW = width * 2
    doubleH = height * 2
    import PIL
    rgbframe = planarYUV_2_planarRGB(data, height, width)
    test = np.asarray(rgbframe, 'u1')
    pictureA = test.reshape(3, height, width)
    pictureA = np.swapaxes(pictureA, 0, 1)
    pictureA = np.swapaxes(pictureA, 1, 2)
    pictureA = np.ndarray.flatten(pictureA)
    imageA = Image.frombytes('RGB', (height, width), pictureA)

    imageA = imageA.resize((doubleH, doubleW), PIL.Image.LANCZOS)

    pixels = list(imageA.getdata())
    pixels = np.asarray(pixels, 'u1')
    pixels = pixels.reshape(doubleH, doubleW, 3)
    pixels = np.swapaxes(pixels, 2, 1)
    pixels = np.swapaxes(pixels, 1, 0)
    pixels = pixels.flatten()
    yuvpixels = planarRGB_2_planarYUV(pixels, doubleH, doubleW)
    return yuvpixels



def createVideoFromFrame(data, filename, numframes, width, height, offset = 8):
    #print("length of data is: " + str(len(data)))
    #shape = datayuv444.shape
    #print(str(shape))

    pic = data.reshape(3, width, height)
    shape = (3, (width+(offset*2)), (height+(offset*2)))
    #print(str(shape))

    bg_pic = np.zeros(shape)

    #add the stripes
    for channel in range(0, shape[0]):
        for row in range(0, shape[1]):
            bg_pic[channel, row,:] = random.randint(0,255)

    #datargb = planarYUV_2_planarRGB(bg_pic, width=40, height=40)
    #display_image_rgb(datargb, 40, 40)


    # remove any existing file
    if os.path.exists(filename):
        os.remove(filename)

    for x in range(0, numframes):
        #print(str(x))
        comb1 = np.array(bg_pic[:,(x*2):,:].copy())
        comb2 = np.array(bg_pic[:,0:(x*2),:].copy())
        
        #print("The shape of the background is: "+str(bg_pic.shape))
        #print("The shape of the comb1 is: "+str(comb1.shape))
        #print("The shape of the comb2 is: "+str(comb2.shape))
        comb = np.concatenate((comb1, comb2), axis=1)
        #print("The shape of the comb is: "+str(comb.shape))
        
        # Now slap the picture in the middle
        comb_pic = comb.copy()
        comb_pic[:, offset:(offset+width), offset:(offset+height)] = pic[:,:,:]
        
        #datargb = planarYUV_2_planarRGB(comb_pic, width=(width+(offset*2)), height=(height+(offset*2)))
        #display_image_rgb(datargb, (width+(offset*2)), (height+(offset*2)))
        
        datayuv = YUV444_2_YUV420(comb_pic, width=(width+(offset*2)), height=(height+(offset*2)))
        appendToFile(datayuv, filename)



def compressFile(app, yuvfilename, w, h, qp, outcomp, outdecomp, deblock = True, intraOnly = False, verbose = False):
    if verbose:
        print("************Compressing the yuv************")
    inputres = '{}x{}'.format(w,h)

    #app = "../x264/x264"
    #if sys.platform == 'win32':
        #app = "..\\x264\\x264.exe"
        #appargs = '-o {} -q {} --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(outcomp, qp, inputres, outdecomp, yuvfilename)
    appargs = '-o {} -q {} --ipratio 1.0 --pbratio 1.0 --no-psy --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(outcomp, qp, inputres, outdecomp, yuvfilename)
    if deblock == False:
        appargs = appargs + ' --no-deblock'
    if intraOnly:
        appargs = appargs + ' -I 1'

    ####### WARNING WARNING constant bitrate is totally different!!!) #####################
    if qp > 100:
        #it's bitrate, not qp, set up args accordingly (bitrate in kbps)
        kbps = int((qp+512)/1024)
        appargs = '-o {} -B {} --input-csp i420 --output-csp i420 --input-res {} --dump-yuv {} {}'.format(outcomp, kbps, inputres, outdecomp, yuvfilename)
    # IBBP: 2 b-frames
    #appargs += ' -b 2 --b-adapt 0'
    
    print appargs

    exe = app + " " + appargs
    #print exe

    if sys.platform == 'win32':
        args = exe
    else:
        args = shlex.split(exe)

    #subprocess.call(args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()

    try:
        outlines = out.splitlines()
        iline = outlines[5].split()
        pline = outlines[6].split()
        bline = outlines[7].split()

        isize = iline[-1]
        psize = pline[-1]
        bsize = bline[-1]
    except:
        isize = 0
        psize = 0
        bsize = 0
        print out
    #print ("iframe average size {}".format(isize))

    return isize, psize, bsize


    if verbose:
        print err
        print out

def comparisonMetrics(imageA, imageB, verbose = False):
    if verbose:
        print("************Comparison Metrics************")
    dataA = np.array(list(imageA.getdata()))
    dataB = np.array(list(imageB.getdata()))
    #m = mse(dataA, dataB)
    m = yuvview.psnr(dataA, dataB)
    s = yuvview.ssim(imageA, imageB)
    return (m, s)

def comparisonMetrics_yuvBuffers(dataA, dataB, width, height, verbose = False):
    if verbose:
        print("************Comparison Metrics************")
            #print("The shape of dataA {}".format(dataA.shape))
    test = np.asarray(np.ndarray.flatten(dataA), 'u1')
    pictureA = test.reshape(3, width, height)
    pictureA = np.swapaxes(pictureA,0,1)
    pictureA = np.swapaxes(pictureA,1,2)
    pictureA = np.ndarray.flatten(pictureA)
    imageA = Image.frombytes('RGB', (width, height), pictureA)
    test = np.array(list(imageA.getdata()))
    #print("The shape of the test {} and the data {}".format(test.shape, test))


    #print("The shape of dataB {}".format(dataB.shape))
    test = np.asarray(np.ndarray.flatten(dataB), 'u1')
    pictureB = test.reshape(3, width, height)
    pictureB = np.swapaxes(pictureB,0,1)
    pictureB = np.swapaxes(pictureB,1,2)
    pictureB = np.ndarray.flatten(pictureB)
    imageB = Image.frombytes('RGB', (width, height), pictureB)
    test = np.array(list(imageB.getdata()))
    #print("The shape of the test {} and the data {}".format(test.shape, test))

    #m = mse(dataA, dataB)
    m = yuvview.psnr(dataA, dataB)
    s = yuvview.ssim(imageA, imageB)
    return (m, s)

def cropImageFromYuvFile(yuvFileName, yuvW, yuvH, bmpFileName, bmpW, bmpH, framenum, yuvformat='i420', bmpformat='L', verbose = False):
    if verbose:
        print("************Crop image from yuv************")
    offsetx = (yuvW - bmpW)/2
    offsety = (yuvH - bmpH)/2
    #decompbmp ="{}_{}x{}.bmp".format('decomp', yuvW, yuvH)
    
    decompImg = yuvview.yuvFileTobmpFile (yuvFileName=decompfilename, width=yuvW, height=yuvH, framenum=framenum, format=yuvformat, bmpFileName="")
    w, h = decompImg.size
    #print "decompImg dimensions {} by {}".format(w,h)
    #print "offsetx {} offsety {} sw {} sh {}".format(offsetx, offsety, sw, sh)
    image_out = decompImg.crop((offsetx, offsety, offsetx+bmpW, offsety+bmpH))
    image_out = image_out.convert(bmpformat)
    #print(image_in.format, image_in.size, image_in.mode)
    #print(image_out.format, image_out.size, image_out.mode)
    #img.rotate(45, expand=True)
    image_out.save(bmpFileName)
    return image_out

##########################################################################################
## These come from the CIFAR python notebook that I've been messing about with
##########################################################################################
import cPickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def thresholdAndRound(y):
    maxn = 255
    minn = 0
    y[y > maxn] = maxn
    y[y < minn] = minn
    y = np.around(y,0)
    return y

def convertToBytes(y):
    y = np.asarray(y, 'u1')
    return y


#Note that this function takes as input a planar RGB image
# It returns planar YUV4:4:4 (it's not common but it can be downsampled to 4:2:0)
def planarRGB_2_planarYUV(data, width, height):
    #print("in planarRGB_2_planarYUV")
    delta = 128.0
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    r = np.array(picture[0])
    g = np.array(picture[1])
    b = np.array(picture[2])
    #print("orig R:" + str(r[0]))
    #print("orig G:" + str(g[0]))
    #print("orig B:" + str(b[0]))
    
    y = np.array(0.299*r + 0.587*g + 0.114*b)
    y = thresholdAndRound(y)
    u = ((b-y)*0.564) + delta
    v = ((r-y)*0.713) + delta
    
    #print("orig Y:" + str(y[0]))
    #print("orig U:" + str(u[0]))
    #print("orig V:" + str(v[0]))
    
    y = thresholdAndRound(y)
    u = thresholdAndRound(u)
    v = thresholdAndRound(v)
    y = convertToBytes(y)
    u = convertToBytes(u)
    v = convertToBytes(v)
    
    yuv = np.concatenate((y,u,v), axis = 0)
    yuv = yuv.reshape((width*height*3), )
    
    #print(y)
    #print(v)
    
    return yuv

def YUV444_2_YUV420(data, width, height):
    from scipy import signal
    #print("YUV444_2_YUV420")
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    #shape = u.shape
    #print("The old shape of u: "+ str(shape))
    #print(u)
    
    kernel = np.array([[1,1,0],
                       [1,1,0],
                       [0,0,0]])
        
    # Perform 2D convolution with input data and kernel
    u = signal.convolve2d(u, kernel, mode='same')/kernel.sum()
    u = u[::2, ::2].copy()
    v = signal.convolve2d(v, kernel, mode='same')/kernel.sum()
    v = v[::2, ::2].copy()

    y = y.flatten()
    u = u.flatten()
    v = v.flatten()

    #shape = u.shape
    #print("The new shape of u: "+ str(shape))
    yuv = np.concatenate([y,u,v])
    return yuv


def YUV444_2_YUV420_alt(data, width, height):
    from scipy import signal
    # print("YUV444_2_YUV420_alt")
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, height, width)
    y = picture[0]
    u = picture[1]
    v = picture[2]

    # shape = u.shape
    # print("The old shape of u: "+ str(shape))
    # print(u)

    kernel = np.array([[1, 1, 0],
                       [1, 1, 0],
                       [0, 0, 0]])

    # Perform 2D convolution with input data and kernel
    u = signal.convolve2d(u, kernel, mode='same') / kernel.sum()
    u = u[::2, ::2].copy()
    v = signal.convolve2d(v, kernel, mode='same') / kernel.sum()
    v = v[::2, ::2].copy()

    y = y.flatten()
    u = u.flatten()
    v = v.flatten()

    # shape = u.shape
    # print("The new shape of u: "+ str(shape))
    yuv = np.concatenate([y, u, v])
    return yuv

def YUV420_2_YUV444(data, width, height):
    from scipy import signal
    #print("YUV420_2_YUV444")
    picture = np.array(data)
    picSize = width*height
    #picture = pic_planar.reshape(3, width, height)
    y = np.array(picture[0:picSize])
    
    u = np.array(picture[picSize:(picSize*5/4)])
    u = u.reshape((width/2), (height/2))
    #print("The old shape of u: "+ str(u.shape))
    #print(u)
    u = np.repeat(u, 2, axis=0)
    #print("The new shape of u: "+ str(u.shape))
    #print(u)
    u = np.repeat(u, 2, axis=1)
    #print("The new shape of u: "+ str(u.shape))
    #print(u)
    
    v = np.array(picture[(picSize*5/4):])
    v = v.reshape((width/2), (height/2))
    #print("The old shape of v: "+ str(v.shape))
    #print(v)
    v = np.repeat(v, 2, axis=0)
    #print("The new shape of v: "+ str(v.shape))
    #print(u)
    v = np.repeat(v, 2, axis=1)
    #print("The new shape of v: "+ str(v.shape))
    #print(v)
    
    
    y = y.flatten()
    u = u.flatten()
    v = v.flatten()
    
    #shape = u.shape
    #print("The new shape of u: "+ str(shape))
    yuv = np.concatenate([y,u,v])
    return yuv





# planar YUV 4:4:4 to rgb
def planarYUV_2_planarRGB(data, width, height):
    #print("in planarYUV_2_planarRGB")
    maxn = 255
    minn = 0
    delta = 128.0
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    #print("recon Y:" + str(y[0]))
    #print("recon U:" + str(u[0]))
    #print("recon V:" + str(v[0]))
    
    
    r = y + 1.403 * (v-delta)
    g = y - (0.714 * (v-delta)) - (0.344 * (u-delta))
    b = y + 1.773 * (u-delta)
    
    #r = y + 1.13983 * v
    #g = y - (0.58060 * v) - (0.39465 * u)
    #b = y + (2.03211 * u)
    
    
    r = thresholdAndRound(r)
    r = convertToBytes(r)
    g = thresholdAndRound(g)
    g = convertToBytes(g)
    b = thresholdAndRound(b)
    b = convertToBytes(b)
    #print("Reconstructed r:" + str(r[0]))
    #print("Reconstructed g:" + str(g[0]))
    #print("Reconstructed b:" + str(b[0]))
    
    rgb = np.concatenate((r,g,b), axis = 0)
    rgb = rgb.reshape((width*height*3), )
    return rgb

def planarYUV_2_planarRGB_areadyshaped(data):
    #print("in planarYUV_2_planarRGB")
    maxn = 255
    minn = 0
    delta = 128.0
    y = data[0]
    u = data[1]
    v = data[2]
    
    #print("recon Y:" + str(y[0]))
    #print("recon U:" + str(u[0]))
    #print("recon V:" + str(v[0]))
    
    
    r = y + 1.403 * (v-delta)
    g = y - (0.714 * (v-delta)) - (0.344 * (u-delta))
    b = y + 1.773 * (u-delta)
    
    #r = y + 1.13983 * v
    #g = y - (0.58060 * v) - (0.39465 * u)
    #b = y + (2.03211 * u)
    
    
    r = thresholdAndRound(r)
    r = convertToBytes(r)
    g = thresholdAndRound(g)
    g = convertToBytes(g)
    b = thresholdAndRound(b)
    b = convertToBytes(b)
    #print("Reconstructed r:" + str(r[0]))
    #print("Reconstructed g:" + str(g[0]))
    #print("Reconstructed b:" + str(b[0]))
    
    rgb = np.concatenate((r,g,b), axis = 0)
    return rgb


def planarYUV_2_planarBGR(data, width, height):
    # print("in planarYUV_2_planarRGB")
    maxn = 255
    minn = 0
    delta = 128.0
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]

    # print("recon Y:" + str(y[0]))
    # print("recon U:" + str(u[0]))
    # print("recon V:" + str(v[0]))

    r = y + 1.403 * (v - delta)
    g = y - (0.714 * (v - delta)) - (0.344 * (u - delta))
    b = y + 1.773 * (u - delta)

    # r = y + 1.13983 * v
    # g = y - (0.58060 * v) - (0.39465 * u)
    # b = y + (2.03211 * u)

    r = thresholdAndRound(r)
    r = convertToBytes(r)
    g = thresholdAndRound(g)
    g = convertToBytes(g)
    b = thresholdAndRound(b)
    b = convertToBytes(b)
    # print("Reconstructed r:" + str(r[0]))
    # print("Reconstructed g:" + str(g[0]))
    # print("Reconstructed b:" + str(b[0]))

    rgb = np.concatenate((b, g, r), axis=0)
    rgb = rgb.reshape((width * height * 3), )
    return rgb


def quantiseUV(data, width, height):
    numLevels = 16
    q = 256/numLevels
    x = np.linspace(0, 10, 1000)
    
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    y = picture[0]
    u = picture[1]
    v = picture[2]
    
    u = q * np.round(u/q)
    v = q * np.round(v/q)
    
    yuv = np.concatenate((y,u,v), axis = 0)
    yuv = yuv.reshape((width*height*3), )
    return yuv


def saveToFile(data, filename):
    datayuv = np.asarray(data, 'u1')
    yuvByteArray = bytearray(datayuv)
    mylen = len(yuvByteArray)
    yuvFile = open(filename, "wb")
    yuvFile.write(yuvByteArray)
    yuvFile.close()

def appendToFile(data, filename):
    datayuv = np.asarray(data, 'u1')
    yuvByteArray = bytearray(datayuv)
    mylen = len(yuvByteArray)
    #print("Adding bytes to file: "+str(mylen))
    yuvFile = open(filename, "ab")
    yuvFile.write(yuvByteArray)
    yuvFile.close()

def cropImagesFromYUVfile(filename, width, height, frameNos):
    yuvFile = open(filename, "rb")
    data = np.fromfile(yuvFile, dtype=np.uint8)
    #data = yuvFile.read()
    #mylen = len(data)
    #print("Length of data: "+ str(mylen))
    # ASSUME YUV420 TODO: YUV444
    frameSize = width*height*3/2
    frames = []
    for frameNo in frameNos:
        start = frameSize*frameNo
        finish = (frameSize*(frameNo+1))
        #print("Start: "+ str(start)+ " Finish: " + str(finish))
        frame = data[start:finish]
        frames.append(frame)
    yuvFile.close()
    return frames

def cropROIfromYUV444(data, c, wsrc, hsrc, w, h, x, y):
    dst = np.zeros((c,w,h), dtype=np.uint)
    data = np.array(data)
    data = data.reshape(c,wsrc,hsrc)
    dst[:,:,:] = data[:,x:(x+w), y:(y+h)].copy()
    return dst

def interlace(data, width, height):
    pic_planar = np.array(data)
    picture = pic_planar.reshape(3, width, height)
    yt = picture[0, ::2, :]
    yb = picture[0, 1::2, :]
    ut = picture[1, ::2, :]
    ub = picture[1, 1::2, :]
    vt = picture[2, ::2, :]
    vb = picture[2, 1::2, :]
    
    
    # offset the bottom by 2 pixels of (0, 128, 128)
    a = np.full((height/2, 2), 128)
    b = np.zeros((height/2, 2))
    
    yb = np.concatenate((b,yb), axis = 1)
    ub = np.concatenate((a,ub), axis = 1)
    vb = np.concatenate((a,vb), axis = 1)
    
    yb = yb[:, 0:width]
    ub = ub[:, 0:width]
    vb = vb[:, 0:width]
    
    y = np.empty((width, height), dtype=yt.dtype)
    y[0::2, :] = yt
    y[1::2, :] = yb
    u = np.empty((width, height), dtype=ut.dtype)
    u[0::2, :] = ut
    u[1::2, :] = ub
    v = np.empty((width, height), dtype=vt.dtype)
    v[0::2, :] = vt
    v[1::2, :] = vb
    
    yuv = np.concatenate((y,u,v), axis = 0)
    yuv = yuv.reshape((width*height*3))
    
    return yuv


# Nearest neighbour filter
def nearestNeighbourFilter(data, width, height, initMinSad = 256):
    origType = data.dtype
    pic_planar = np.array(data).astype(int)
    picture = pic_planar.reshape(3, width, height)
    picture_dash = picture.copy()
    # mask = mask.reshape(3, width, height)


    kw = 3
    kh = 3
    half_kw = (kw - 1) / 2
    half_kh = (kh - 1) / 2

    search_h = 3
    search_w = 3
    half_sw = (search_w - 1) / 2
    half_sh = (search_h - 1) / 2

    limit = 3
    count = 0
    for k in range(0, 3):
        c = picture[k, :, :]
        c_dash = picture_dash[k, :, :]
        # c = picture[0, :, :]
        # c_dash = picture_dash[0, :, :]

        for j in range(half_kh, height - half_kh):
            for i in range(half_kw, width - half_kw):
                roi_t = j - half_kh
                roi_b = j + half_kh + 1
                roi_l = i - half_kw
                roi_r = i + half_kw + 1
                roi = c[roi_t:roi_b, roi_l:roi_r]

                # motion search
                minError = initMinSad
                minjj = j
                minii = i
                hSearch = range(j - half_sh, (j + half_sh + 1))
                wSearch = range(i - half_sw, (i + half_sw + 1))

                # print("Search radius: {}; {}".format(half_sh, hSearch))
                # print("Search radius: {}; {}".format(half_sw, wSearch))



                minjj = j
                minii = i
                for jj in hSearch:
                    if jj < half_kh or jj > height - (half_kh + 1):
                        continue
                    for ii in wSearch:
                        # print("Point ({}, {}, {}); cf: ({}, {}, {});".format(k, i, j, k, ii, jj))
                        if ii < half_kw or ii > width - (half_kw + 1):
                            continue
                        if jj == j and ii == i:
                            continue
                        search_area = c[(jj - half_kh):(jj + half_kh + 1), (ii - half_kw):(ii + half_kw + 1)]
                        error = np.sum(np.absolute(np.subtract(roi, search_area)))
                        # print("Error: {}; minError {}".format(error, minError))
                        if error < minError:
                            minError = error
                            selectedArea = search_area
                            minjj = jj
                            minii = ii

                c_dash[j, i] = (c[j, i] + c[minjj, minii]) / 2
                # print("c({}, {}): {}; cf({}, {}): {}; result: {}".format(i, j, c[j,i], ii, jj, c[jj, ii], c_dash[j,i]))

                # picture_dash[1, j, i] = (picture[1, j, i] + picture[1, jj, ii])/2
                # picture_dash[2, j, i] = (picture[2, j, i] + picture[2, jj, ii])/2
    return picture_dash


