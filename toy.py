import numpy as np
import matplotlib.pyplot as plt


def numpyWhere():
     a = x = np.arange(25).reshape(5, 5)
     print(a)
     b = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,0,0,0],
          [1,1,1,1,1],
          [1,1,1,1,1]]
     b = np.array(b)
     print(b)


     indices = np.where(b==0)
     print(indices)


     a_mask0 = np.average(a[np.where(b == 0)])
     print(a_mask0)

     a_mask1 = np.average(a[np.where(b == 1)])
     print(a_mask1)

     rows = [0, 2, 4]

     a_red = a[rows, :]
     print(a_red)


def plotIframeMcc():
    print("Plotting the i-frame matthews correlation coefficient")

    # values obtained from spreadsheed resultsLog_ipDeltas.xlsx

    bpp = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    bpp2 = [0.01, 0.02, 0.05, 0.1, 0.2]
    # These were averaged over all sequences, MCC considers 1st comp i-frame frequency correct
    singleCompMCCs = [0.681319075, 0.834363118, 0.865844626, 0.858136807, 0.69174728, 0.330742273, 0.200047523]
    singleCompF1s = [0.665396825, 0.82879085, 0.860915033, 0.85664488, 0.70000406, 0.352401205, 0.226571429]
    singleCompF1s2 = [0.665396825, 0.82879085, 0.860915033, 0.85664488, 0.70000406]

    # These were averaged over all sequences
    singleCompAvgQPs = [5.323540277, 4.429319163, 3.66343589, 3.239011691, 2.700049134, 1.994843104, 1.440642199]
    singleCompVarQPs = [2.05631947, 1.294984978, 0.851119993, 0.887996208, 1.064710981, 1.243244577, 1.134829649]
    from operator import add
    from operator import sub
    lower = list( map(add, singleCompAvgQPs, singleCompVarQPs) )
    upper = list( map(sub, singleCompAvgQPs, singleCompVarQPs) )

    # These were averaged over all sequences and over all 2nd comp bitrates, MCC considers 1st comp i-frame frequency correct
    firstCompMCCs = [0.5486493, 0.590613373, 0.41160071, 0.318173812, 0.231379242, 0.165864074, 0.168522311]
    # These were averaged over all sequences and over all 1st comp bitrates, MCC considers 2nd comp i-frame frequency correct
    secondCompMCCs = [0.213666433, 0.312235281, 0.501431234, 0.554784572, 0.570761966, 0.588660427, 0.597734196]

    #firstF1s_const2ndComp0_05 = [0.221111111, 0.227046784, 0.192586547, 0.31585695, 0.405010352, 0.733759398, 0.837839697]
    #secondF1s_const2ndComp0_05 = [0.704516595, 0.768197946, 0.824880383, 0.517879767, 0.298142857, 0.225483738, 0.160162726]
    firstF1s_const2ndComp0_05 = [0.221111111, 0.227046784, 0.192586547, 0.31585695, 0.405010352]
    secondF1s_const2ndComp0_05 = [0.704516595, 0.768197946, 0.824880383, 0.517879767, 0.298142857]

    firstF1s_const2ndaveraged = [0.216850092, 0.260063033, 0.293021818, 0.329570695, 0.407435442, 0.456171274, 0.538337664]
    secondF1s_const2ndaveraged = [0.579270665, 0.677681045, 0.664364073, 0.566135323, 0.435947215, 0.262426589, 0.173076418]

    plt.rcParams.update({'font.size': 16})
    plt.plot(bpp2, firstF1s_const2ndComp0_05, label="first comp.", color='#c0c000')
    plt.plot(bpp2, secondF1s_const2ndComp0_05, label="second comp.", color='#000000')
    plt.title("Re-compression (first compression = 0.05 bpp)")
    plt.legend()
    plt.xlabel("Bits per pixel (second compression)")
    plt.ylabel("Key frame F1 score")
    plt.savefig("fig_doubleCompressionf1s_2ndCompConst0_05")
    plt.show()

    plt.plot(bpp, firstF1s_const2ndaveraged, label="first compression")
    plt.plot(bpp, secondF1s_const2ndaveraged, label="second compression")
    plt.title("Re-compression (first compression averaged)")
    plt.legend()
    plt.xlabel("Bits per pixel (second compression)")
    plt.ylabel("Key frame F1 score")
    plt.savefig("fig_doubleCompressionf1s_2ndCompConstAveraged")
    plt.show()

    plt.plot(bpp, singleCompMCCs)
    plt.title("Single Compression")
    plt.xlabel("Bits per pixel")
    plt.ylabel("Key frame MCC")
    plt.show()

    plt.plot(bpp2, singleCompF1s2, color='#000000')
    plt.title("Single Compression")
    plt.xlabel("Bits per pixel")
    plt.ylabel("Key frame F1 score")
    plt.savefig("fig_singleCompressionf1s")
    plt.show()


    plt.plot(bpp, singleCompAvgQPs, 'r-')
    plt.plot(bpp, upper, 'b--')
    plt.plot(bpp, lower, 'b--')
    plt.title("Single Compression")
    plt.xlabel("Bits per pixel")
    plt.ylabel("Average predicted QP")
    plt.show()

    plt.plot(singleCompAvgQPs, singleCompMCCs)
    plt.title("Single Compression")
    plt.xlabel("Average QP (predicted)")
    plt.ylabel("Key frame MCC")
    plt.show()

    plt.plot(bpp, firstCompMCCs)
    plt.title("Double Compression")
    plt.xlabel("Bits per pixel")
    plt.ylabel("Key frame MCC (1st comp)")
    plt.show()

    plt.plot(bpp, secondCompMCCs)
    plt.title("Double Compression")
    plt.xlabel("Bits per pixel")
    plt.ylabel("Key frame MCC (2nd comp)")
    plt.show()

if __name__ == "__main__":
    #numpyWhere()
    plotIframeMcc()