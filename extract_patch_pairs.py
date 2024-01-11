import numpy as np
import numpy.matlib
import copy
import matplotlib.pyplot as plt
import scipy.ndimage

class patchPair:
    # to store patch pair objects
    pass

def extract_patch_pairs_image_pair(refImageObj, objImageObj):
    # EXTRACT_PATCH_PAIRS_IMAGE_PAIR extracts the patch pairs from a reference image
    # of which patches have already been extracted
    #
    #   Parameters:
    #       refImageObj - object representing the reference image
    #           > maskNormalised: 2D array(uint8)
    #           > patches - object containing patches and patch locations
    #               > patch: 3D array containing all patches of the image
    #               > location: 3D array containing the patch locations as
    #                   [row, column] pairs
    #       objImageObj - object representing the object image
    #
    #  Returns:
    #       patchPairs - list of patch pairs containing the following fields
    #           > patchReference: 2D array(uint8)
    #           > patchObject: 2D array(uint8)
    #           > location: 3D array containing patch pair locations as
    #               [row, column] pairs

    patchPairs = []

    # # register object image adding imageRegistered and maskRegistered
    # objImageObj = register_images(refImageObj, objImageObj)
    #
    # compute combined finger mask
    mask = np.logical_and(refImageObj.maskNormalised, objImageObj.maskRegistered)

    # loop over reference patches
    patchSize = len(refImageObj.patches.patches[0])
    patchRefList = refImageObj.patches
    for patch in range(len(patchRefList.patches)):
        # check mask
        rowTop = patchRefList.locations[patch][0]
        columnLeft = patchRefList.locations[patch][1]
        rowBottom = rowTop + patchSize
        columnRight = columnLeft + patchSize
        x = mask[rowTop:rowBottom, columnLeft:columnRight]
        onMask = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
        if ~onMask:
            continue
        # take reference patch, object patch, and location
        pPair = patchPair()
        pPair.patchReference = patchRefList.patches[patch]
        pPair.patchObject = objImageObj.imageRegistered[rowTop:rowBottom,
                            columnLeft:columnRight]
        pPair.location = patchRefList.locations[patch]
        patchPairs.append(pPair)
       
    return patchPairs

