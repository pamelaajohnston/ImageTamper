import os
import patchIt2 as pi

def createFileList(srcDir="/Volumes/LaCie/data/YUV_temp", ext="yuv", baseNamesOnly=True):
    fileList = []
    index = 0
    # First, create a list of the files to encode, along with dimensions
    for (dirName, subdirList, filenames) in os.walk(srcDir):
        for filename in filenames:
            if filename.endswith(ext):
                n = os.path.join(dirName, filename)
                if baseNamesOnly:
                    p, n = os.path.split(filename)
                fileList.append(n)
    return fileList


def getFileList(dir, ext, mustContains=[]):
    #First get the names of all the files
    print("Getting filenames")
    allFiles = createFileList(dir, ext, baseNamesOnly=False)
    allFiles = sorted(allFiles)
    relevantFiles = allFiles
    for mustContain in mustContains:
        relevantFiles = [f for f in relevantFiles if mustContain in f]

    return relevantFiles

def fileNameToResultsFilename(filename):
    f, ext = os.path.splitext(filename)
    p, f = os.path.split(f)
    p, set = os.path.split(p)
    p, state = os.path.split(p)

    resultsFolder = "{}_{}_{}".format(f, state, set)
    return resultsFolder


if __name__ == '__main__':
    baseDir = "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test"
    mustContains = ["oneFrame", "altered"]
    collatedFiles = [
                        ["base", "z_YUV_all", "/Volumes/LaCie/data/FaceForensics/FaceForensics_compressed/test"],
                        ["qp.yuv", "z_predictedQP", "/Volumes/LaCie/data/FaceForensics/FullResults_fullFrame"],
                        ["predictions.yuv", "z_predictions", "/Volumes/LaCie/data/FaceForensics/FullResults_fullFrame"],
    ]

    yuvFiles = getFileList(baseDir, "yuv", mustContains)

    for file in yuvFiles:
        width, height = pi.getDimsFromFileName(file)
        for collatedFile in collatedFiles:
            type = collatedFile[0]
            outFileRoot = collatedFile[1]
            dir = collatedFile[2]
            outputFile = "{}_{}x{}.yuv".format(outFileRoot, width, height)
            print(outputFile)
            exists = os.path.isfile(outputFile)


            if type == "base":
                inputFile = file
            else:
                f = fileNameToResultsFilename(file)
                inputFile = os.path.join(dir, f)
                inputFile = os.path.join(inputFile, type)

            with open(inputFile, "rb") as f:
                data1 = f.read()
            if exists:
                with open(outputFile, "rb") as f:
                    data2 = f.read()
                combined_data = data1 + data2
                with open(outputFile, "wb") as f:
                    f.write(combined_data)
            else:
                with open(outputFile, "wb") as f:
                    f.write(data1)




