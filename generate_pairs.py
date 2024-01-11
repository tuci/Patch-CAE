import numpy as np
import cv2
import numpy.matlib as matlib
from register_images import register_images
from process_image_set import extract_roi
from extract_patch_pairs import extract_patch_pairs_image_pair
from fine_register_images import horizontal_alignment_windows, vertical_alignment, correct_rotation
import matplotlib.pyplot as plt
from extract_patches_image_set import extract_patches_image

class Pair:
    pass

def generate_genuine_pairs(imageSet):
    # GENERATE_GENUINE_PATCH_PAIRS returns an array of all genuine patch pairs
    # in the same set
    #
    #   Returns:
    #       patchPairs - list of all genuine patch pairs containing fields for
    #           reference patch
    #           > patchReference: 2D array(uint8)
    #           > patchObject: 2D array(uint8)
    #           > location: 3D array containing patch pair locations as [row, column]
    #               pairs.

    imagePairs = []
    nImageSets = len(imageSet)

    # loop over all images
    index = 0
    for setIndex in range(nImageSets):
        set = imageSet[setIndex]
        for imageIndexRef in range(len(set)):
            imageRef = set[imageIndexRef]
            refObj = Pair()
            refObj.imageNormalised = imageRef.imageNormalised
            refObj.maskNormalised = imageRef.maskNormalised
            # loop over every other image of the same finger
            for imageIndexObj in range(len(set)):
                if imageIndexObj == imageIndexRef:
                    continue
                imageObj = set[imageIndexObj]
                # register object image
                imageObj = register_images(imageRef, imageObj)
                imagepair = [imageRef, imageObj]
                index += 1
                # add image pair to array
                imagePairs.append(imagepair)

        # status update
        print('Making train set. Progress genuine pairs: {0:.2f}'
              .format(setIndex/nImageSets*100))

    return imagePairs

def generate_genuine_image_pairs(imageSet):
    # GENERATE_GENUINE_PATCH_PAIRS returns an array of all genuine patch pairs
    # in the same set
    #
    #   Returns:
    #       patchPairs - list of all genuine patch pairs containing fields for
    #           reference patch
    #           > patchReference: 2D array(uint8)
    #           > patchObject: 2D array(uint8)
    #           > location: 3D array containing patch pair locations as [row, column]
    #               pairs.

    imagePairs = []
    nImageSets = len(imageSet)

    # loop over all images
    index = 0
    for setIndex in range(nImageSets):
        set = imageSet[setIndex]
        for imageIndexRef in range(len(set)):
            imageRef = set[imageIndexRef]
            # imageRef.image = CorrectVerticalStripes(imageRef.image)
            # loop over every other image of the same finger
            for imageIndexObj in range(len(set)):
                if imageIndexObj == imageIndexRef:
                    continue
                imageObj = set[imageIndexObj]
                # imageObj.image = CorrectVerticalStripes(imageObj.image)
                # register object image
                # if setIndex == 0 and imageIndexRef == 1 and imageIndexObj == 3:
                # fig, ax = plt.subplots(2,2)
                # ax = ax.ravel()
                # ax[0].imshow(imageRef.imageNormalised, cmap='gray')
                # ax[1].imshow(imageObj.imageNormalised, cmap='gray')
                # ax[2].imshow(imageRef.imageNormalised, cmap='gray')
                imageObj = register_images(imageRef, imageObj)
                # ax[3].imshow(imageObj.imageRegistered, cmap='gray')
                # plt.show()
                imagepair = [imageRef, imageObj]
                # add image pair to array
                imagePairs.append(imagepair)

        # status update
        print('Making train set. Progress genuine pairs: {0:.2f}'
              .format(setIndex/nImageSets*100))

    return imagePairs


def select_object_images(imageSet, setIndex, nObjectImages=3):
    # SELECT_OBJECT_IMAGES selects a certain amount of imposter object images
    #
    #   Parameters:
    #       imageSet - set of images.
    #           > object
    #       setIndex - index of the set of genuine images that should not be selected
    #           > integer
    #       nObjectImages - number of images to be selected
    #           > integer
    #           > default - 3
    #   Returns:
    #       objectImages - array of selected images
    #           > object array
    #

    objectImages = []
    nSets = len(imageSet)

    # make list of all image indices (set, image)
    setList = np.arange(nSets)
    setList = np.setdiff1d(setList, setIndex)
    indexList = [np.repeat(setList,len(imageSet[0])), matlib.repmat(np.arange(0,len(imageSet[0])),1,nSets-1)]

    # select nObjectImages indices
    permutedOrder = np.random.permutation(len(indexList[0]))
    # indexListSelected = indexList[:,permutedOrder[0:nObjectImages]]

    for i in range(nObjectImages):
        fIdx = indexList[0][permutedOrder[i]]
        # imIdx = indexList[1][0][permutedOrder[i]]
        imIdx = np.random.randint(0, len(imageSet[fIdx]))
        objectImages.append(imageSet[fIdx][imIdx])

    return objectImages
