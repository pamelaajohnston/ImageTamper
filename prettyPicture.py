import numpy as np
import functions as fn
import math

#math.sqrt(x)

dim = 352 # because everything is square
width = dim
height = dim

uw = width//2
uh = height//2




cent = dim//2
maxDist = math.sqrt(2*cent*cent) - 20

for f in range(0, 80):
    centx = (dim // 2) + (f*7)
    centy = (dim // 2) - (f * 7)
    centx2 = (dim // 2) + (f*7)
    centy2 = (dim // 2) - (f * 7)
    centx = (dim // 2)
    centy = (dim // 2)
    centx2 = (dim // 2)
    centy2 = (dim // 2)
    y = np.zeros((width * height))
    u = np.full((uw * uh), 200)
    v = np.full((uw * uh), 0)

    y = y.reshape((height, width))
    u = u.reshape((uh, uw))
    v = v.reshape((uh, uw))
    for i in range(0, dim):
        for j in range(0, dim):
            xdist = ((i-centx)*(i-centx))
            ydist = ((j-centy)*(j-centy))
            dist = math.sqrt(xdist+ydist) - (2*f)
            if (dist < 0):
                dist = 0
            if dist % 16:
                dist = dist - (dist %3)

            value = ((maxDist - dist)/maxDist)*192
            value = round(value)
            #print(value)
            y[j,i] = value

    doUV = False
    if doUV:
        for i in range(0, uw):
            for j in range(0, uh):
                xdist = abs(i - centx2)
                ydist = abs(j - centy2)
                value1 = (xdist*255/uw)
                value1 = round(value1)
                value2 = (ydist*255/uw)
                value2 = round(value2)
                print(value1)
                print(value2)
                #print(i, j)
                u[j,i] = value1
                v[j,i] = value2


    y = y.flatten()
    u = u.flatten()
    v = v.flatten()

    frame = np.concatenate((y,u,v), axis = 0)
    filename = "aMeh_{}x{}.yuv".format(width, height)

    fn.appendToFile(frame, filename)


