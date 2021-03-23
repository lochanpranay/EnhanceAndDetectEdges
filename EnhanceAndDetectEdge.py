import numpy as np
from matplotlib import pyplot as plt
import math
import os
from os import path


# Accepts file object and returns a 2D image array if the file is P2(.pgm), otherwise returns an empty list.
def getImgArr(imgFile):
    rows = None
    cols = None
    maxIntensity = None
    pixArr = None

    if imgFile.readline() != "P2\n":
        return []

    for line in imgFile:
        if line.startswith("#"):
            continue
        elif rows == None and cols == None:
            cols, rows = int(line.split()[0]), int(line.split()[1])
        elif maxIntensity == None:
            maxIntensity = int(line)
        else:
            pixArr = np.array(line.split(), dtype="uint8")
            break
    return pixArr.reshape(rows, cols)


# Quick utility function for displaying an image. Accepts a 3D image array.
def showImg(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.show()


# Utility function to convert 2D array to 3D, useful when working with library functions that accept only 3D image array.
def convert2dTo3d(arr):
    arr3d = np.zeros((len(arr), len(arr[0]), 3), dtype="uint8")
    for i in range(len(arr)):
        for k in range(len(arr[0])):
            val = arr[i][k]
            arr3d[i][k] = [val, val, val]
    return arr3d

# Utility method for creating P2 image in the local file system from 2D pixel array.
def createP2Image(pixels, rows, cols, maxIntensity, fileName, comment=None):
    newimg = open(fileName + ".pgm", "w+")
    newimg.write("P2\n")
    if comment != None:
        newimg.write("#" + comment + "\n")
    newimg.write(str(cols) + " " + str(rows) + "\n")
    newimg.write(str(maxIntensity) + "\n")
    for arr in pixels:
        for val in arr:
            newimg.write(str(val) + " ")

    newimg.close()


# Performs a linear transformation for enhancing the contrast of the image.
# S = ar + b, range[10, 255]
# Returns enhanced image.
def contrastStretching(imgArr):
    Imin = imgArr.min()
    Imax = imgArr.max()
    a = 245/(Imax - Imin)
    b = 10 - (a * Imin)
    newImgArr = np.zeros((len(imgArr), len(imgArr[0])), dtype="uint8")
    for index, pixVal in np.ndenumerate(imgArr):
        newImgArr[index[0], index[1]] = a * pixVal + b

    return newImgArr


# Normalizes the pixel values and applies power law transformation on the input image array.
# Default gamma value = 1
# Returns modified image.
def applyPowerLaw(imgArr, gamma=1):
    newImgArr = np.zeros((len(imgArr), len(imgArr[0])), dtype="uint8")
    for index, pixVal in np.ndenumerate(imgArr):
        newPixVal = 255 * math.pow(pixVal/255, gamma)
        newImgArr[index[0], index[1]] = newPixVal
    return newImgArr


# Applies averaging mask on the input image array to reduce the image noise.
# Default kernel size = 3, which generates a 3*3 (N X N) kernel.
# Returns the processed image array.
def applyLinearFilter(imgArr, kernelSize=3):
    newImgArr = np.zeros((len(imgArr), len(imgArr[0])), dtype="uint8")
    kernel = np.ones([kernelSize, kernelSize], dtype="uint8")

    edgeVal = int((kernelSize - 1) / 2)
    rows = len(imgArr)
    cols = len(imgArr[0])
    rowStart = colStart = edgeVal
    rowEnd = rows - edgeVal
    colEnd = cols - edgeVal
    div = kernelSize * kernelSize

    for i in range(rows):
        for k in range(cols):
            if (i >= rowStart and i < rowEnd) and (k >= colStart and k < colEnd):
                newPixVal = 0
                for x in range(kernelSize):
                    for y in range(kernelSize):
                        newPixVal += imgArr[i - edgeVal + x,
                                            k - edgeVal + y] * kernel[x, y]
                newImgArr[i, k] = newPixVal / div
            else:
                newImgArr[i, k] = imgArr[i, k]

    return newImgArr


# Performs median filter neighborhood operations on the input image array to reduce the image noise.
# Default kernel size = 3, which generates a 3*3 (N X N) kernel.
# Returns the processed image array.
def applyMedianFilter(imgArr, kernelSize=3):
    newImgArr = np.zeros((len(imgArr), len(imgArr[0])), dtype="uint8")

    edgeVal = int((kernelSize - 1) / 2)
    rows = len(imgArr)
    cols = len(imgArr[0])
    rowStart = colStart = edgeVal
    rowEnd = len(imgArr) - edgeVal
    colEnd = len(imgArr[0]) - edgeVal
    div = kernelSize * kernelSize

    for i in range(rows):
        for k in range(cols):
            if (i >= rowStart and i < rowEnd) and (k >= colStart and k < colEnd):
                pxlist = []
                for x in range(kernelSize):
                    for y in range(kernelSize):
                        pxlist.append(imgArr[i - edgeVal + x, k - edgeVal + y])

                pxlist.sort()
                newImgArr[i, k] = pxlist[int((div - 1) / 2)]
            else:
                newImgArr[i, k] = imgArr[i, k]

    return newImgArr


# Performs edge detection on the input image array using Prewitt filter.
# Returns magnitude of the image. Modify return statement to return magnitudes of horizontal and vertical edges.
# Pixel values may exceed 255, perform linear transform before displaying the image.
def getEdgesUsingPrewitt(imgArr):
    verticalFilter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontalFilter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    imgEdges, horEdges, vertEdges = getImgEdges(
        imgArr, verticalFilter, horizontalFilter)
    return contrastStretching(imgEdges)


# Performs edge detection on the input image array using Sobel filter.
# Returns magnitude of the image. Modify return statement to return magnitudes of horizontal and vertical edges.
# Pixel values may exceed 255, perform linear transform before displaying the image.
def getEdgesUsingSobel(imgArr):
    verticalFilter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontalFilter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imgEdges, horEdges, vertEdges = getImgEdges(
        imgArr, verticalFilter, horizontalFilter)
    return contrastStretching(imgEdges)


# Helper method for Prewitt and Sobel methods.
# Calculates magnitudes based on the input filters.
# Do not call this method directly.
def getImgEdges(imgArr, verticalFilter, horizontalFilter):
    filterSize = len(verticalFilter)
    imgHorEdges = np.zeros((len(imgArr), len(imgArr[0])), dtype=int)
    imgVerEdges = np.zeros((len(imgArr), len(imgArr[0])), dtype=int)
    imgEdges = np.zeros((len(imgArr), len(imgArr[0])), dtype=int)

    edgeVal = int((filterSize - 1) / 2)
    rows = len(imgArr)
    cols = len(imgArr[0])
    rowStart = colStart = edgeVal
    rowEnd = len(imgArr) - edgeVal
    colEnd = len(imgArr[0]) - edgeVal

    for i in range(rows):
        for k in range(cols):
            if (i >= rowStart and i < rowEnd) and (k >= colStart and k < colEnd):
                vertPixVal = 0
                horPixVal = 0
                for x in range(filterSize):
                    for y in range(filterSize):
                        pixVal = imgArr[i - edgeVal + x, k - edgeVal + y]
                        vertPixVal += pixVal * verticalFilter[x, y]
                        horPixVal += pixVal * horizontalFilter[x, y]
                imgVerEdges[i, k] = abs(vertPixVal)
                imgHorEdges[i, k] = abs(horPixVal)
                imgEdges[i, k] = imgVerEdges[i, k] + imgHorEdges[i, k]
            else:
                imgEdges[i, k] = imgVerEdges[i,
                                             k] = imgHorEdges[i, k] = imgArr[i, k]

    return [imgEdges, imgHorEdges, imgVerEdges]


print(""" *** Hello, This program reads gray-level images as input. It can perform contrast enhancement using linear and log transform.
Removes gray level noise using averaging and median filters and performs edge detection using Canny, Sobel and Prewitt techniques. ***\n """)

imgName = input(
    "Enter name of the image file (accepted formats: .PGM(P2)):").upper()

if not imgName.endswith(".PGM"):
    print("only .PGM images are accepted")
    exit(0)

imgPath = os.path.dirname(os.path.realpath(__file__)) + "\\" + imgName

if not path.exists(imgPath):
    print("file doesn't exist in the directory: " +
          os.path.dirname(os.path.realpath(__file__)))
    exit(0)

imgFile = open(imgPath, "r")
imgArr = getImgArr(imgFile)

if imgArr == []:
    print("invalid image file")
    exit(0)

option = 0
while(option != -1):
    print("\n\n Enter the number of the operation that you would like to perform on the input image: ")
    print("\n 1) Enhance the contrast of the image using Linear Transform")
    print("\n 2) Apply Power Law transformation")
    print("\n 3) Apply Averaging Filter for Noise Reduction")
    print("\n 4) Apply Median Filter for Noise Reduction")
    print("\n 5) Perform Edge detection using Prewitt Filter")
    print("\n 6) Perform Edge detection using Sobel Filter")
    # print("\n 7) Perform Edge detection using Canny edge detection")
    print("\n 7) Exit")

    option = input("\n Enter your choice of operation: ")
    if option.isnumeric():
        option = int(option)
    else:
        continue

    if option == 1:
        imgArr = contrastStretching(imgArr)

    if option == 2:
        gamma = input(
            "Enter gamma value, 0 < gamma <= 10 ,default gamma = 1: ")
        if gamma.isdigit() or gamma.replace('.', "", 1).isdigit():
            gamma = float(gamma)
            if gamma > 0 and gamma <= 10:
                imgArr = applyPowerLaw(imgArr, gamma)

    if option == 3:
        fs = input(
            "Enter filter size, Should be an Odd num, 3 <= filter < 10, default size = 3 X 3: ")
        if fs.isnumeric():
            fs = int(fs)
            if fs >= 3 and fs < 10 and (fs % 2 != 0):
                imgArr = applyLinearFilter(imgArr, fs)
        else:
            imgArr = applyLinearFilter(imgArr)

    if option == 4:
        fs = input(
            "Enter filter size, Should be an Odd num, 3 <= filter < 10, default size = 3 X 3: ")
        if fs.isnumeric():
            fs = int(fs)
            if fs >= 3 and fs < 10 and (fs % 2 != 0):
                imgArr = applyMedianFilter(imgArr, fs)
        else:
            imgArr = applyMedianFilter(imgArr)

    if option == 5:
        imgArr = getEdgesUsingPrewitt(imgArr)

    if option == 6:
        imgArr = getEdgesUsingSobel(imgArr)

    if option == 7:
        exit(0)

    if option >= 1 and option <= 6:
        showImg(convert2dTo3d(imgArr))

print("Done")
