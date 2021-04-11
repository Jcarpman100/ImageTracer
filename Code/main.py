import numpy as np
import random
from matplotlib import pyplot as plt

def colorFinder(input, numColors):
    colors = np.zeros((numColors, 3))

    for i in range(numColors):
        colors[i][0] = random.uniform(0.0, 1.0)
        colors[i][1] = random.uniform(0.0, 1.0)
        colors[i][2] = random.uniform(0.0, 1.0)

    return colors

def LeastDifference(colors, pixel):

    differences = np.subtract(colors, pixel)
    differences = np.sum(abs(differences), axis=1)
    bestColor = np.argmin(differences)

    return colors[bestColor]


def colorScale(input, numColors):
    imageSize = input.shape

    # find the best colors to choose for the current image, pretty hard
    colors = colorFinder(input, numColors)

    # Make the output image full of 0's
    output = np.zeros(imageSize)

    # for every pixel in the image, find the color that is the closest match and set the corresponding output pixel to
    # that color. Could probably improve the runtime
    for y in range(imageSize[0]):
        for x in range(imageSize[1]):
            output[y][x] = LeastDifference(colors, input[y][x])


    # return the finished image
    return output


def greyScale(input, numColors):
    imageSize = input.shape

    # Simple conversion to grayscale image
    input = np.sum(input, axis=2) / 3

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
            if (input [y][x] < .5):
                input[y][x] = 0.0
            else:
                input[y][x] = 1.0

    input = np.stack([input] * 3, axis=2)

    return input



if __name__ == '__main__':
    imageDir = '../Images/'
    outDir = '../Results/'

    # Number of images in the input folder
    N = 5

    # user choices of number of colors and deciding whether to output with color or not.
    colors = int(input("How many colors would you like? : "))
    choice = input("Would you like a (c)olor image or a (g)reyscale image? (c/g)? : ")

    if (choice == "c"):
        print("Color image processing has not been developed yet. ;;")

    for index in range(1, N + 1):
        # read the current image
        image = plt.imread(imageDir + "image_" + (str(index).zfill(2)+ ".jpg")) / 255

        if (choice != "c"):
            output = greyScale(image, colors)
        else :
            output = colorScale(image, colors)


        plt.imsave("{}/result_{}.jpg".format(outDir, str(index).zfill(2)), output)
