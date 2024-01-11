import numpy as np
import matplotlib.pyplot as plt
from process_image_set import compute_properties_image
import cv2

class Patches:
    pass

def extract_patches_image_set(imageSet, patchSize, slide=None):
    # EXTRACT_PATCHES_IMAGE_SET adds patches to normalised images
    # Patches do not overlap
    # Parameters:
    #   imageSet - set of images. Objects represent a set of images containing the
    #       following fields
    #       > imageNormalised - 2D array(uint8)
    #       > maskNormalised - 2D array(logical)
    #   patchSize - size of patches to be taken
    #       > integer
    #
    #   Returns:
    #       imageSetPatches - following fields are added to imageSet
    #           > patches - object containing the following fields
    #               > patch: 2D array(unit8)
    #               > location: 1D array
    #                   > row: integer
    #                   > column: integer
    # Created on: 3-7-2019

    # field is added, image set is copied
    imageSetPatches = imageSet

    if slide == None:
        slide = patchSize

    # loop over all images
    nImageSet = len(imageSet)
    for s in range(nImageSet):
        nImages = len(imageSet[s])
        for i in range(nImages):
            image = imageSet[s][i]
            imagePatches = extract_patches_image(image, patchSize, slide)
            # add patches filed to image set
            imageSetPatches[s][i].patches = imagePatches.patches

        # print status
        print('Extracting image patches. Progress: {0:.2f}'.format(s/nImageSet*100))

    # print final status
    print('Extracting image patches done ...')
    return imageSetPatches

def extract_patches_image(imageSet, patchSize, slide):
    # field is added, imageSet is copied
    imagePatches = imageSet

    # take image and mask
    image = imageSet.imageNormalised
    mask = imageSet.maskNormalised

    # define patch object
    patchObj = Patches()

    # define patch and location array
    patches = []
    locations = []

    # loop over all non-overlapping patches
    [mRow, nColumn] = image.shape

    h, w = patchSize
    numpatches = 0
    for rowBottom in range(h,mRow, slide[0]):
        rowTop = rowBottom - h
        for columnRight in range(w,nColumn, slide[1]):
            columnLeft = columnRight - w
            # check mask
            onMask = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
            if ~onMask:
                continue
            # add patch and its location
            patch = image[rowTop:rowBottom,columnLeft:columnRight]
            row = rowTop
            column = columnLeft
            patches.append(patch)
            locations.append([row, column])
    patchObj.patches = patches
    patchObj.locations = locations
    imagePatches.patches = patchObj
    return imagePatches
