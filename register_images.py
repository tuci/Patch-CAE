import numpy as np
from compute_register_transformation import compute_register_transformation
from process_image_set import apply_transformation
from copy import deepcopy
from MaximumCurvature import MaximumCurvature
import matplotlib.pyplot as plt

class Image:
    def __init__(self):
        pass
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

def register_images(imageRef, imageObj):
    # REGISTER_IMAGE registrates an object image with a reference image,
    # adding fields for the registered image and its registered mask
    #
    #   Parameters:
    #       imageRef - object representing the reference image
    #           > upperEdge: array(double)
    #           > lowerEdge: array(double)
    #           > normaliseTransformation: 2D affine transformation
    #       imageObj - object representing the object image
    #           > image: 2D array(uint8)
    #           > upperEdge: array(double)
    #           > lowerEdge: array(double)
    #           > mask: 2D array(logical)
    #
    #   Returns:
    #       imageRegistered - as imageObj, the following fields are added
    #           > imageRegistered: 2D array(uint8)
    #           > maskRegistered: 2D array(logical)
    #

    # take relevant fields from images
    upperEdgeRef = imageRef.upperEdge
    lowerEdgeRef = imageRef.lowerEdge
    upperEdgeObj = imageObj.upperEdge
    lowerEdgeObj = imageObj.lowerEdge
    image = imageObj.image
    mask = imageObj.mask
    transformationNormalise = imageRef.normaliseTransformation

    # compute full transformation
    transformationRegister = compute_register_transformation(upperEdgeRef, lowerEdgeRef,
                                            upperEdgeObj, lowerEdgeObj)
    transformation = np.dot(transformationRegister,transformationNormalise)
    #transformation = np.transpose(transformation)

    # apply transformation to image and mask
    imageReg = apply_transformation(image, transformation)
    maskReg = apply_transformation(mask, transformation)

    # copy original image and add fields
    imageRegistered = Image()
    imageRegistered = deepcopy(imageObj)
    # imageRegistered = imageObj
    imageRegistered.imageRegistered = imageReg.astype(np.uint8)
    imageRegistered.maskRegistered = np.rint(maskReg).astype(bool)

    return imageRegistered