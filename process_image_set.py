import numpy as np
import cv2
import scipy.ndimage
from skimage.transform import AffineTransform, warp
import matplotlib.pyplot as plt

class Image:
    pass


def compute_properties_image_set(imageSetBasic, histEq=False, correctStripes=True, correctEdges=False):
    # COMPUTE_PROPERTIES_IMAGE_SET computes the relevant properties of the images
    # in an image set
    # Parameters:
    #   imageSetBasic - set of selected images. array of struct(object) containing the
    #       following fields
    #       > images: 2D array(uint8)
    #   *args: options
    #       > 'histogramEqualisation' - whether histogram equalisation is applied to
    #           image and imageNormalised
    #           > logical (default: false)
    # Returns:
    # 	imageSstBasic - object array of images contatining the following fileds
    # 	     > image: 2D array(uint8)
    #	     > upperEdge: array(double)
    #	     > lowerEdge: array(double)
    #	     > mask: 2D array(logical)
    # 	     > normalisedTransform: 2D affine transformation
    #	     > imageNormalised: 2D array(uint8)
    #	     > maskNormalised: 2D array(logical)
    #
    # Created on: 1-7-2019

    nImageSets = len(imageSetBasic)
    # loop through all images
    for f in range(nImageSets):
        # number of images per finger
        nImages = len(imageSetBasic[f])
        for i in range(nImages):
            # compute properties for the selected image
            imageBasic = imageSetBasic[f][i].image
            image = compute_properties_image(imageBasic,histEq, correctStripes, correctEdges)
            # add computed properties to the image set
            imageSetBasic[f][i].image = image.image
            imageSetBasic[f][i].upperEdge = image.upperEdge
            imageSetBasic[f][i].lowerEdge = image.lowerEdge
            imageSetBasic[f][i].mask = image.mask
            imageSetBasic[f][i].radius = image.radius
            imageSetBasic[f][i].imageNormalised = image.imageNormalised#.astype(np.uint8)
            imageSetBasic[f][i].maskNormalised = image.maskNormalised
            imageSetBasic[f][i].normaliseTransformation = \
                image.normaliseTransformation

        # print status
        print('Computing image properties. Progres: {0:.2f}'.format(f/nImageSets*100))

    return imageSetBasic

def compute_properties_image(imageBasic, histEq, correctStripes, correctEdges):
    # COMPUTE_PROPERTIES_IMAGE computes the relevant properties of an image
    # Parameters:
    #   imageBasic - single image
    #       > 2D array(uint8)
    #   hitEq - whether histogram equalisation is applied to image and imageNormalised
    #       > logical
    #
    # Returns:
    #   imageStruct - struct of an image containing the following fields
    #       > image: 2D aray(uint8)
    #       > upperEdge: array(double)
    #       > lowerEdge: array(double)
    #       > mask: 2D array(logical)
    #       > normalisedTransformation: 2D affine transformation
    #       > imageNormalised: 2D array(uint8)
    #       > maskNormalised: 2D array(logical)
    #
    # Created on: 1-7-2019

    # correct vertical stripes
    if correctStripes:
        imageBasic = CorrectVerticalStripes(imageBasic)

    # compute edges
    [upperEdge, lowerEdge] = compute_edges(imageBasic)
    if correctEdges:
        [upperEdge, lowerEdge] = adjust_edges(upperEdge, lowerEdge)
        # upperEdge = adjust_edges_window(upperEdge, edgetype='upper', winsize=50)
        # lowerEdge = adjust_edges_window(lowerEdge, edgetype='lower', winsize=5)
    # compute normalise transformation
    normaliseTranformation = compute_normalise_transformation(upperEdge, lowerEdge,
                                                len(imageBasic),len(imageBasic[0]))

    # compute mask
    mask = compute_mask(upperEdge, lowerEdge, len(imageBasic), len(imageBasic[0]))

    # apply normalise tranformation on mask
    maskNormalised = apply_transformation(mask, normaliseTranformation)

    # crop excess edges
    indices = np.where(np.mean(maskNormalised, axis=1) < 0.6)
    for r in indices:
        maskNormalised[r, :] = 0

    # apply histogram equalisation
    if histEq:
        clahe = cv2.createCLAHE(clipLimit=2.0)
        imageBasic = clahe.apply(imageBasic)

    # apply normalise transformation to image
    imageNormalised = apply_transformation(imageBasic, normaliseTranformation)

    # add fields to image struct
    imgObj = Image()
    imgObj.image = imageBasic
    imgObj.upperEdge = upperEdge
    imgObj.lowerEdge = lowerEdge
    imgObj.radius = np.mean((np.max(upperEdge)+np.min(lowerEdge)) / 2)
    imgObj.mask = mask
    imgObj.imageNormalised = imageNormalised
    imgObj.maskNormalised = maskNormalised
    imgObj.normaliseTransformation = normaliseTranformation

    return imgObj

def apply_transformation(image, normaliseTransformation):
    # APPLY_TRANSFORMATION transforms an image

    imageTransformed = warp(image,AffineTransform(normaliseTransformation), mode='edge',
                            preserve_range=True)

    return imageTransformed

def CorrectVerticalStripes(image):
    lut = np.zeros(256).astype(np.float32)
    nlut = np.zeros(256).astype(np.int)
    for y in range(10, image.shape[0] - 10):
        for x in range(10, image.shape[1] - 10, 2):
            lut[int(round(image[y][x]))] += image[y][x].astype(np.float) - 0.5 * (image[y][x + 1].astype(np.float) +
                                                                      image[y][x - 1].astype(np.float))
            nlut[int(round(image[y][x]))] += 1
            lut[int(round(image[y][x + 1]))] += 0.5 * (image[y][x + 2].astype(np.float) + image[y][x].astype(np.float)) - \
                                    image[y][x + 1].astype(np.float)
            nlut[int(round(image[y][x + 1]))] += 1

    slut = np.zeros(256)
    for i in range(256):
        slut[i] = lut[i]
        n = nlut[i]
        for d in range(1, 256):
            if i + d < 256:
                n += nlut[i + d]
                slut[i] += lut[i + d]
            if i - d >= 0:
                n += nlut[i - d]
                slut[i] += lut[i - d]
            if n > 1000 and d > 5:
                break
        slut[i] /= n

    for i in range(256):
        if nlut[i] > 0:
            lut[i] /= nlut[i]

    ilut = np.zeros(256).astype(np.int)
    for i in range(256):
        if slut[i] >= 0:
            ilut[i] = int(slut[i] + 0.5)
        else:
            ilut[i] = int(slut[i] - 0.5)
    for y in range(image.shape[0]):
        for x in range(2, image.shape[1], 2):
            l = image[y][x]
            l -= ilut[int(round(l))]
            if l < 0:
                image[y][x] = 0
            elif l > 255:
                image[y][x] = 255
            else:
                image[y][x] = l
    return image

def log_transformation(image):
    # Apply log transformation method
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))

    # Specify the data type so that
    # float value will be converted to int
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image

def compute_mask(upperEdge, lowerEdge, imgHeight, imgWidth):
    # COMPUTE_MASK computes binary map in which pixels on the finger
    # is true

    # create mask based on edges covering the finger in the
    # original image
    mask = np.transpose(np.zeros((imgHeight, imgWidth), dtype=bool))
    for c in range(imgWidth):
        mask[c][upperEdge[c]:lowerEdge[c]] = True
    mask = np.transpose(mask)

    return mask

def compute_edges(image, maskHeight=4, maskWidth=20):
    # COMPUTE_EDGES localise the finger outline using simple filtering mask
    # Parameters:
    #   image - finger vein image
    #       > 2D array(uint8)
    #   maskHeight - height of the filtering mask
    #       > integer
    #   maskWidth - width of the filtering mask
    #       > integer
    #
    # Returns:
    #   upperEdge - column coordinates of upper edge
    #       > array(double)
    #   lowerEdge - column coordinates of lower edge
    #       > array(double)
    #
    #  Reference:
    #  Finger vein recognition using minutia-based alignment and local binary
    #  pattern-based feature extraction
    #  E.C. Lee, H.C. Lee and K.R. Park
    #  International Journal of Imaging Systems and Technology
    #  Volume 19, Issue 3, September 2009, Pages 175-178
    #  doi: 10.1002/ima.20193
    #
    #  Author:  Bram Ton <b.t.ton@alumnus.utwente.nl>
    #  Date:    20th March 2012
    #  License: Simplified BSD License
    #
    # Last edit: 2-7-2019

    # upper edge is searhed in upper half of the image
    image = cv2.normalize(image.astype(float), None, 0.0, 1.0, cv2.NORM_MINMAX)
    imageHeight = len(image)
    imageWidth = len(image[0])
    if imageHeight % 2 == 0:
        imageCenter = int(imageHeight/2) + 1
    else:
        imageCenter = np.ceil(imageHeight/2).astype(int)

    # construct mask for filtering
    maskUpper = np.full((int(maskHeight/2),maskWidth),-1)
    maskLower = np.full((int(maskHeight/2),maskWidth),1)
    mask = np.concatenate((maskUpper, maskLower), axis=0)


    # filter image
    imageFiltered = scipy.ndimage.correlate(image, mask, mode='nearest')

    # upper and lower part of filtered image
    imageFilteredUpper = imageFiltered[0:int(np.floor(imageHeight/2)),:]
    imageFilteredLower = imageFiltered[imageCenter:,:]

    # Edges relative to their half
    upperEdgeRelative = np.argmax(imageFilteredUpper, axis=0)
    lowerEdgeRelative = np.argmin(imageFilteredLower,axis=0)

    # Edges with respect to the entire image
    upperEdge = upperEdgeRelative
    lowerEdge = np.round(lowerEdgeRelative + len(imageFilteredLower))

    return [upperEdge, lowerEdge]

def compute_edges_kono(image, sigma):
    # upper edge is searhed in upper half of the image
    image = cv2.normalize(image.astype(float), None, 0.0, 1.0, cv2.NORM_MINMAX)
    imageHeight = len(image)
    imageWidth = len(image[0])
    if imageHeight % 2 == 0:
        imageCenter = int(imageHeight / 2) + 1
    else:
        imageCenter = np.ceil(imageHeight / 2).astype(int)

    # construct filter kernel
    winsize = np.ceil(4 * sigma)
    x = np.append(np.arange(-winsize, winsize, 1), winsize)
    [X, Y] = np.meshgrid(x, x)
    hy = (-Y / (2*np.pi*np.power(sigma, 4)))*np.exp(-(np.power(X, 2) + np.power(Y, 2)) / (2*np.power(sigma, 2)))

    # filter image
    # filter image
    imageFiltered = scipy.ndimage.correlate(image, hy, mode='nearest')

    # upper and lower part of filtered image
    imageFilteredUpper = imageFiltered[0:int(np.floor(imageHeight / 2)), :]
    imageFilteredLower = imageFiltered[imageCenter:, :]

    # Edges relative to their half
    upperEdgeRelative = np.argmax(imageFilteredUpper, axis=0)
    lowerEdgeRelative = np.argmin(imageFilteredLower, axis=0)

    # Edges with respect to the entire image
    upperEdge = upperEdgeRelative
    lowerEdge = np.round(lowerEdgeRelative + len(imageFilteredLower))

    return upperEdge, lowerEdge

def compute_normalise_transformation(upperEdge, lowerEdge, imgHeight, imgWidth):
    # COMPUTE_NORMALISE_TRANSFORMATION computes tranformation to normalise finger
    # vein image.

    # compute the center line parameters
    b = compute_center_line(upperEdge, lowerEdge)

    # compute center of image
    centerRow = imgHeight/2
    centerColumn = imgWidth/2

    # define translation and rotation
    ty = -(np.dot(np.array([1,centerColumn]),b[0])) + centerRow
    q = -(np.arctan(b[0][1]))

    # rotate around origin to make center line horizontal
    rotation = np.identity(3)
    rotation[0][0] = rotation[1][1] = np.cos(q)
    rotation[0][1] = np.sin(q)
    rotation[1][0] = -(np.sin(q))

    # translate center line to origin
    rotation[1][2] = -ty

    transformationMatrix = rotation
    return transformationMatrix

def compute_center_line(upperEdge, lowerEdge):
    # COMPUTE_CENTER_LINE finds the center line of the finger by least-square
    # regression

    nColumn = len(upperEdge)

    # Take mean of upper and lower edge to reduce influence of far edges
    meanEdge = np.transpose((upperEdge+lowerEdge)/2)
    edgepoints = np.arange(nColumn).tolist()
    ones = np.ones(nColumn, dtype=int).tolist()

    # regression line by least-square method
    X = np.transpose([ones, edgepoints])
    Y = meanEdge
    b = np.linalg.lstsq(X,Y)

    return b

# extract reagion of interest
def extract_roi(image):
    # image = np.pad(image, ((2,2), (0,0)), 'constant', constant_values=(0,0))
    # h, w = image.shape
    [upperEdge, lowerEdge] = compute_edges(image)
    # tangents
    upperTanget = np.max(upperEdge)
    lowerTangent = np.min(lowerEdge)

    # vein_roi = veins[upperTanget:lowerTangent, :]
    image = image[upperTanget:lowerTangent, :]

    return image, upperTanget, lowerTangent

def adjust_edges(upperEdge, lowerEdge, root=20, tip=20):
    upperEdgeRevised = np.copy(upperEdge)
    lowerEdgeRevised = np.copy(lowerEdge)

    # finger tip
    auxEdge = np.copy(upperEdge)
    meanedge = np.mean(auxEdge[tip:-root])
    auxEdge[auxEdge < meanedge - 5] = meanedge
    auxEdge[auxEdge > meanedge + 5] = meanedge
    upperEdgeRevised = auxEdge
    auxEdge = np.copy(lowerEdge)
    meanedge = np.mean(auxEdge[tip:-root])
    auxEdge[auxEdge > meanedge + 5] = meanedge
    auxEdge[auxEdge < meanedge - 5] = meanedge
    lowerEdgeRevised = auxEdge

    return upperEdgeRevised, lowerEdgeRevised

def adjust_edges_window(edge, edgetype, winsize):
    edgemean = np.mean(edge[20:-20])
    for i in range(len(edge) - winsize):
        window = edge[i:i+winsize]
        winMean = np.mean(window)
        if edgetype == 'upper':
            if np.all(window < edgemean + 10):
                auxEdge = edge[i+winsize:]
                closest = np.where(auxEdge > edgemean + 1)[0]
                if closest.size == 0:
                    window = np.ones((1, len(window)))[0] * np.round(winMean)
                else:
                    window = np.ones((1, len(window)))[0] * auxEdge[closest[0]]
            elif np.any(window <= edgemean + 10):
                window[window > winMean] = np.round(winMean)
            # check abruptly high edges
            if np.any(window > edgemean + 60):
                window[window > winMean] = np.round(winMean)
        elif edgetype == 'lower':
            if np.all(window > edgemean + 10):
                auxEdge = edge[i+winsize:]
                closest = np.where(auxEdge < edgemean - 1)[0]
                if closest.size == 0:
                    window = np.ones((1, len(window)))[0] * np.round(winMean)
                else:
                    window = np.ones((1, len(window)))[0] * auxEdge[closest[0]]
            elif np.any(window >= edgemean + 10):
                window[window > winMean] = np.round(winMean)
            if np.any(window < edgemean - 60):
                window[window < winMean] = np.round(winMean)
        edge[i:i+winsize] = window.astype(int)

    return edge


