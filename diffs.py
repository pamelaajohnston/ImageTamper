from PIL import Image
import numpy as np


def savePic(saveName, rgbpic, height, width, border=False):
    test = np.asarray(rgbpic, 'u1')
    pictureA = test
    #pictureA = test.reshape(3, height, width)
    #pictureA = np.swapaxes(pictureA,0,1)
    #pictureA = np.swapaxes(pictureA,1,2)
    #pictureA = np.ndarray.flatten(pictureA)

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

pic1 = "fig_mobile_qp0.png"
pic2 = "fig_mobile_qp42.png"

entries = [
    ["fig_mobile_qp0.png", "fig_mobile_qp14.png", "fig_mobile_0diff14.png", 352, 288],
    ["fig_mobile_qp0.png", "fig_mobile_qp28.png", "fig_mobile_0diff28.png", 352, 288],
    ["fig_mobile_qp0.png", "fig_mobile_qp42.png", "fig_mobile_0diff42.png", 352, 288],
]

entries = [
    ["fig_ff_full_ori.png", "fig_ff_full_alt.png", "fig_ff_full_diff.png", 640, 480],
    ["fig_ff_crop_ori.png", "fig_ff_crop_alt.png", "fig_ff_crop_diff.png", 146, 178],
]

for entry in entries:
    pic1 = entry[0]
    pic2 = entry[1]
    outname = entry[2]
    width = entry[3]
    height = entry[4]

    p1 = Image.open(pic1)
    p1a = np.array(p1.getdata())
    print(p1a.shape)
    p2 = Image.open(pic2)
    p2a = np.array(p2.getdata())

    diff = np.subtract(p1a,p2a)
    diff = abs(diff)

    s = 255.0/diff.max()
    diff = diff * s

    diff = diff.tolist()
    print(len(diff))
    binarised = False

    if binarised:
        diff = [[250, 250, 250] if i != [0,0,0] else [0,0,0] for i in diff]

    print(len(diff))


    savePic(outname, diff, width, height)
