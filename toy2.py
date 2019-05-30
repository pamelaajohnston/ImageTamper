import numpy as np
import matplotlib.pyplot as plt
% matplotlib
inline


def generateConfusionMatrixPicture(conf_arr, pictureName, xlabels="", ylabels=""):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if xlabels == "":
        xlabels = alphabet[:width]
    if ylabels == "":
        ylabels = alphabet[:height]

    plt.xticks(range(width), xlabels)
    plt.yticks(range(height), ylabels)
    plt.savefig(pictureName, format='png')
    plt.show()


conf_arr = np.array([[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
                     [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                     [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                     [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                     [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                     [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]])

print("The example")
generateConfusionMatrixPicture(conf_arr, "cm_example.png")