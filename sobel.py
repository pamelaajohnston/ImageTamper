from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt



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


entries = [
    ["fig_mobile_qp0.png", "fig_mobile_sobel.png"],
]

for entry in entries:
    pic1 = entry[0]

    img = cv2.imread(pic1, 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig("sobel_img.png", bbox_inches='tight')

    plt.imshow(sobely, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig("sobel_y.png", bbox_inches='tight')

    plt.imshow(sobelx, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig("sobel_x.png", bbox_inches='tight')

    #p1 = Image.open(pic1)
    #p1a = np.array(p1.getdata())
    #print(p1a.shape)


    #savePic(outname, diff.tolist(), 352, 288)
