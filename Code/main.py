###################################################################
# Title : Final Project : Image Tracing
# Created By : Aimee Martello and Joseph Carpman
# Date : 4/11/2021
# Class : CSCE 489 : Computational Photography
#
# This project is an attempt to recreate the illustrator image trace function in both gray scale and color images.
#
###################################################################


import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
import cv2

# random color generator, used to test the color replacement function before color finder was implemented.
# User can ask for it after selecting the colored mode
def randomColors(numColors):
    colors = np.zeros((numColors, 3))

    # Fill each channel of each color with a random floating point number from 0-1
    for i in range(numColors):
        colors[i][0] = random.uniform(0.0, 1.0)
        colors[i][1] = random.uniform(0.0, 1.0)
        colors[i][2] = random.uniform(0.0, 1.0)

    return colors


# Find the most dominant colors in the image based on the number of colors requested by the user
# Uses KMeans - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def colorFinder(input, numColors):
    # reshape the image so that it is a 1-D list of pixels
    reshapedInput = input.reshape((-1,3))

    # set up our KMeans clusters
    cluster = KMeans(n_clusters=numColors, precompute_distances=True, n_init=5)

    # fit our clusters to the pixels of the image
    colorCluster = cluster.fit(reshapedInput)

    # find the centroids of the color map
    colors = colorCluster.cluster_centers_

    return colors

# given the array of all colors and the current pixel, finds the best color for the current pixel.
def LeastDifference(colors, pixel):

    # Find the difference between the given pixel and every dominant color
    differences = np.subtract(colors, pixel)
    differences = np.sum(abs(differences), axis=1)

    # Find the lowest difference
    bestColor = np.argmin(differences)

    # Return the lowest difference color
    return colors[bestColor]

# Traces the image with color
def colorScale(input, numColors, smoothing, colorMode):
    imageSize = input.shape
    # find the best colors to choose for the current image, pretty hard

    # if the colorMode flag is set to 0, use the real color finding function to fill the dominant color array.
    if (colorMode == 0):
        colors = colorFinder(input, numColors)
    # Else, use the random color function to fill the dominant color array.
    elif (colorMode == 1):
        colors = randomColors(numColors)

    # If the user wants the image to be smoothed, call a gaussian blur function to remove some detail
    # I know that this is not a very good way to do this, but I cannot find a better one. ):
    if (smoothing):
        input = cv2.bilateralFilter(np.float32(input), 5, 60, 60)
        #input = cv2.GaussianBlur(input, (5, 5), 0)
    # Make the output image full of 0's
    output = np.zeros(imageSize)

    # for every pixel in the image, find the color that is the closest match and set the corresponding output pixel to
    # that color. Could probably improve the runtime
    for y in range(imageSize[0]):
        for x in range(imageSize[1]):
            output[y][x] = LeastDifference(colors, input[y][x])
        # Status Printer
        print("\r" + str((y / imageSize[0]) * 100) + "%           ", end="")
    print("\r" + str(100) + "%              ", end="")
    # return the finished image
    return output

# Traces the image with grayscale
def greyScale(input, numColors, smoothing):
    imageSize = input.shape

    # Simple conversion to grayscale image
    input = np.sum(input, axis=2) / 3

    # If the user wants the image to be smoothed, call a gaussian blur function to remove some detail
    # I know that this is not a very good way to do this, but I cannot find a better one. ):
    if (smoothing):
        input = cv2.bilateralFilter(np.float32(input), 5, 60, 60)

    # Make the output image full of 0's
    output = np.zeros((imageSize[0], imageSize[1]))

    # For the number of shades, find every pixel that is within the current color-range in the input image, and save
    # the corresponding pixel as the current color.
    for i in range(numColors):
        output[input > (i / (numColors))] = (i / (numColors - 1))

    # recreate the 3 color channels by copying the single intensity channel
    output = np.stack([output] * 3, axis=2)

    # return the finished image
    return output


# Depreciated, greyScale function does the same job, but better
def monochrome(input):
    imageSize = input.shape


    # simple conversion to grayscale image
    input = np.sum(input, axis=2) / 3


    for y in range (imageSize[0]):
        for x in range (imageSize[1]):
            # if the intensity is less than 50%, make the pixel black
            if (input [y][x] < .5):
                input[y][x] = 0.0
            # Else, make it white
            else:
                input[y][x] = 1.0
        # Status Printer
        print("\r" + str((y / imageSize[0]) * 100) + "%           ", end="")

    # Convert back to a 3D array
    input = np.stack([input] * 3, axis=2)

    return input



# Makes the image look worse and massively increases runtime, will have to think of a better way to do this.
def smooth(input):

    imageSize = input.shape
    output = np.zeros(imageSize)

    # For every pixel, find the mode of the surrounding pixels and set this pixel to it.
    # I expected this to have a smoothing effect and remove some smaller bits from the output, this did not work.
    for y in range (imageSize[0]):
        for x in range (imageSize[1]):
            left = x-1
            right = x+2
            top = y-1
            bottom = y+2
            if (x == 0):
                left = x
            if (y == 0):
                top = y
            if (y == imageSize[0] - 1):
                bottom = y+1
            if (x == imageSize[1] - 1):
                right = x+1

            currentArea = input[top:bottom, left:right]
            print("\r" + str((y / imageSize[0]) * 100) + "%           ", end="")
            output[y][x] = stats.mode(currentArea, nan_policy='omit').mode[0][0]

    return output

if __name__ == '__main__':
    imageDir = '../Images/'
    outDir = '../Results/'

    # Number of images in the input folder
    N = 5

    # user choices of number of colors and deciding whether to output with color or not.
    colors = int(input("How many colors would you like? : "))
    choice = input("Would you like a (c)olor image or a (g)reyscale image? (c/g)? : ")

    # asks the user if they would like the image to be smoothed
    smoothingChoice = input("would you like the image smoothed? (y/n)? : ")

    smoother = False
    if (smoothingChoice == "y"):
        smoother = True

    colorMode = 0


    # asks the user if they would like to use the random color function
    if (choice == "c"):
        funkyChoice = input("Would you like to use random colors rather than the best colors? (y/n)? : ")
        if (funkyChoice == "y"):
            colorMode = 1


    for index in range(1, N + 1):
        # read the current image
        image = plt.imread(imageDir + "image_" + (str(index).zfill(2)+ ".jpg")) / 255

        # choose the correct function
        if (choice != "c"):
            output = greyScale(image, colors, smoother)
        else :
            output = colorScale(image, colors, smoother, colorMode)

        # Save the image and report the status to the user.
        plt.imsave("{}/result_{}.jpg".format(outDir, str(index).zfill(2)), output)
        print("Image", index, "complete!")
