import numpy as np
import pandas as pd
import cv2, torch, pickle, lmdb
import matplotlib.pyplot as plt
from scipy.ndimage import shift, gaussian_filter
from register_images import register_images
from read_images import read_images_SD
# from scipy import signal
from process_image_set import compute_properties_image, extract_roi, compute_edges, compute_mask
from process_image_set import compute_properties_image_set, adjust_edges
# from generate_pairs import generate_genuine_pairs_shift
# from cae_models import CAE_patch
from collections import OrderedDict
from skimage.metrics import structural_similarity
# from local_patch_shifts import generate_global_shift_patch_pairs
from dataloader import norm_to_unit_length
from MaximumCurvature import MaximumCurvature

class Image:
    pass

def min_max_scaler(data):
    data_std = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_scaled = data_std * (np.max(data) - np.min(data)) + np.min(data)

    return data_std

# load cae model
def load_model(model,path):
    model_parameters = torch.load(path)
    model_weights = model_parameters['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in model_weights.items():
        state_dict_remove_module[k] = v
    model.load_state_dict(state_dict_remove_module)
    return model

def shift_then_crop(image, s):
    refpair = image[0]
    probepair = image[1]
    shifted = shift(probepair.imageRegistered, (0, s), mode='nearest')
    shiftedMask = shift(probepair.maskRegistered.astype(int), (0, s),
            mode='nearest').astype(np.float64)

    # crop shifted part on both reference and probe image
    ref = Image()
    probe = Image()
    if s > 0:
        ref.imageNormalised = refpair.imageNormalised[:, s:]
        ref.maskNormalised = refpair.maskNormalised[:, s:]
        ref.upperEdge = refpair.upperEdge[s:]
        ref.lowerEdge = refpair.lowerEdge[s:]
        probe.imageRegistered = shifted[:, s:]
        probe.maskRegistered = shiftedMask[:, s:]

    elif s < 0:
        s = abs(s)
        ref.imageNormalised = refpair.imageNormalised[:, :-s]
        ref.maskNormalised = refpair.maskNormalised[:, :-s]
        ref.upperEdge = refpair.upperEdge[:-s]
        ref.lowerEdge = refpair.lowerEdge[:-s]
        probe.imageRegistered = shifted[:, :-s]
        probe.maskRegistered = shiftedMask[:, :-s]
    else:
        return refpair, probepair

    return ref, probe

def horizontal_alignment(imageset, save=''):
    lags = np.arange(-60, 60, 1)

    refimage = imageset[0]
    probeimage = imageset[1]
    # register probe image
    # probeimage = register_images(refimage, probeimage)
    refimage_roi = extract_roi(refimage.imageNormalised)
    probeimage_roi = extract_roi(probeimage.imageRegistered)

    h, w = refimage.imageNormalised.shape
    refJoint, GRef = sliding_windows(refimage_roi, win=30, c=10)
    # refJoint = np.argmax(GRef)
    probeJoint, GProbe = sliding_windows(probeimage_roi, win=20, c=10)

    corr, s = crosscorr(GRef, GProbe, lags)
    ref, probe = shift_then_crop([refimage, probeimage], (0, s))
    [upperEdge, lowerEdge] = compute_edges(probe.imageRegistered)
    
    return ref, probe


def horizontal_alignment_windows(imageset, shiftrange=60):
    refimage = imageset[0]
    probeimage = imageset[1]
    lags = np.arange(-shiftrange, shiftrange, 1)
    # register probe image
    #probeimage = register_images(refimage, probeimage)
    refimage_roi, _, _ = extract_roi(refimage.imageNormalised)
    probeimage_roi, _, _ = extract_roi(probeimage.imageRegistered)
    cropsize = np.arange(1, 11, 5)
    # cropsize = [1]
    cropcorr = []
    shft = []
    for size in cropsize:
        refcrop = refimage_roi#[:, size:-size]
        probecrop = probeimage_roi[:, size:-size]
        _, GRef = sliding_windows(refcrop, win=30, c=10)
        # refJoint = np.argmax(GRef)
        _, GProbe = sliding_windows(probecrop, win=30, c=10)
        GRef = min_max_scaler(GRef)
        GProbe = min_max_scaler(GProbe)

        corr, s = crosscorr(GRef, GProbe, lags)
        cropcorr.append(corr)
        shft.append(s)

    shiftCons = np.ceil(np.mean(shft)).astype(int)
    refpair, probepair = shift_then_crop([refimage, probeimage], shiftCons)

    return refpair, probepair

def horizontal_alignment_doublewindows(imageset, save):
    refimage = imageset[0]
    probeimage = imageset[1]
    # register probe image
    probeimage = register_images(refimage, probeimage)
    refimage_roi = extract_roi(refimage.imageNormalised)
    probeimage_roi = extract_roi(probeimage.imageRegistered)

    h, w = refimage.imageNormalised.shape
    refJoint, GRef = sliding_windows(refimage_roi, win=10, c=5)
    # refJoint = np.argmax(GRef)
    probeJoint, GProbe = sliding_windows(probeimage_roi, win=10, c=5)

    i = 0
    subprofiles = split_probe_profile(GProbe)

    suggestedShifts = []
    for i, sp in enumerate(subprofiles):
        corr, s = correlation(GRef, sp, [-100, 100], i)
        suggestedShifts.append(s)
   
    accepted = np.count_nonzero(suggestedShifts)
    if accepted != 0:
        meanShift = np.ceil(np.mean(suggestedShifts)).astype(int)
    else:
        meanShift = 0
    refRegistered, probeRegistered = shift_then_crop([refimage, probeimage], (0, meanShift))
    return refRegistered, probeRegistered

def vertical_alignment(pair, shiftrange=20, pad=False):
    refimage = pair[0]
    probeimage = pair[1]
    shiftCorr = []
    shifts = np.append(np.arange(-shiftrange, shiftrange, 5), shiftrange)
    if pad:
        probeimage.imageRegistered = np.pad(probeimage.imageRegistered, ((40, 40), (0, 0)), 'reflect')
        probeimage.maskRegistered = np.pad(probeimage.maskRegistered, ((40, 40), (0, 0)), 'reflect')
   
    for s in shifts:
        shiftimage = Image()
        shiftimage.imageRegistered = shift(np.copy(probeimage.imageRegistered), (s, 0), mode='reflect')
        shiftimage.maskRegistered = shift(np.copy(probeimage.maskRegistered).astype(int), (s, 0), mode='constant').astype(bool)
        shiftimage.upperEdge = probeimage.upperEdge + s
        shiftimage.lowerEdge = probeimage.lowerEdge + s

        refROI, probeROI = extract_common_roi([refimage, shiftimage])
        if refROI.shape[0] < 70:
            shiftCorr.append(-1.0)
            continue

        if refROI.size == 0:
            shiftCorr.append(-1.0)
            continue
        refROI = cv2.resize(refROI, (256, 128), interpolation=cv2.INTER_AREA)
        probeROI = cv2.resize(probeROI, (256, 128), interpolation=cv2.INTER_AREA)
        pair_correlation = pair_correlation_set([refROI, probeROI])
        
        shiftCorr.append(np.mean(pair_correlation))

    verticalshift = shifts[np.argmax(shiftCorr)]
    if np.all(shiftCorr == 0):
        verticalshift = 0

 
    pair[1].imageRegistered = shift(probeimage.imageRegistered, (verticalshift, 0),
                                    mode='reflect')
    pair[1].maskRegistered = shift(probeimage.maskRegistered.astype(int), (verticalshift, 0),
                                  mode='reflect').astype(bool)

    if pad:
        pair[1].imageRegistered = pair[1].imageRegistered[:refimage.imageNormalised.shape[0],:]
        pair[1].maskRegistered = pair[1].maskRegistered[:refimage.imageNormalised.shape[0],:]

    return pair[1]

def shift_finger_region(imageset, s):
    mask = imageset.maskRegistered
    image = imageset.imageRegistered
    finger = image * mask
    shiftedRegion = np.zeros_like(image)
    fingerMean = np.mean(finger[np.nonzero(finger)])

    for column in range(mask.shape[1]):
        fingerRegion = np.where(mask[:, column] == 1)[0]
        strip = image[min(fingerRegion):max(fingerRegion), column]
        if s < 0:
            cval = strip[-1]
        else:
            cval = strip[0]
        strip = shift(strip, s, cval=cval)
        shiftedRegion[min(fingerRegion):max(fingerRegion), column] = strip

    return shiftedRegion

def vertical_shift_in_region(pair, shiftrange=20):
    refimage = pair[0]
    probepair = pair[1]
    pairSim = []
    shiftrange = np.append(np.arange(-shiftrange, shiftrange, 5), shiftrange)
    for s in shiftrange:
        shifted = Image()
        shifted.imageRegistered = shift_finger_region(probepair, s)
        shifted.maskRegistered = probepair.maskRegistered
        shifted.upperEdge = probepair.upperEdge
        shifted.lowerEdge = probepair.lowerEdge
        refROI, probeROI = extract_common_roi([refimage, shifted])
        ssim = pair_correlation_set([refROI, probeROI])
        pairSim.append(ssim)

    shiftVal = shiftrange[np.argmax(pairSim)]

    shiftedPair = Image()
    shiftedPair.imageRegistered = shift_finger_region(probepair, shiftVal)
    shiftedPair.maskRegistered = probepair.maskRegistered
    shiftedPair.upperEdge = probepair.upperEdge
    shiftedPair.lowerEdge = probepair.lowerEdge

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(refimage.imageNormalised, cmap='gray')
    # ax2.imshow(shiftedPair.imageRegistered, cmap='gray')
    # plt.suptitle('Shift {}'.format(shiftVal))
    # plt.show()

    return shiftedPair

def correct_rotation(pair, rotRange=45):
    refimage = pair[0]
    probeimage = pair[1]
    upperEdge, lowerEdge = find_edges(refimage.maskNormalised, islower=True)
    refimage.upperEdge = upperEdge
    refimage.lowerEdge = lowerEdge
    rotAngles = np.append(np.arange(-rotRange, rotRange, 5), rotRange)

    rotcorr = []

    for angle in rotAngles:
        probeRot = rotate_image(probeimage, angle)

        refROIC, probeROIC = extract_common_roi([refimage, probeRot])
        if refROIC.size == 0:
            rotcorr.append(0.0)
            continue
        if refROIC.shape[0] < 65:
            rotcorr.append(0.0)
            continue
        pair_correlation = image_correlation(refROIC, probeROIC, angle)
        rotcorr.append(pair_correlation)

    rotAngle = rotAngles[np.argmax(rotcorr)]
    if np.all(np.array(rotcorr) == 0):
        rotAngle = 0
    probeimage = rotate_image(probeimage, rotAngle)

    return probeimage#, rotAngle

def image_correlation(img1, img2, angle):
    img1 = norm_to_unit_length(img1)
    img2 = norm_to_unit_length(img2)
    diff = img1.shape[0] - img2.shape[0]
    if diff < 0: # img1 is smaller
        # fill with image mean
        h, w = img1.shape
        aux = np.full((h + abs(diff), w), np.mean(img1))
        if angle < 0:
            aux[:diff, :] = img1
        else:
            aux[abs(diff):, :] = img1
        img1 = aux
    elif diff > 0: # img2 smaller
        # fill with image mean
        h, w = img2.shape
        aux = np.full((h + diff, w), np.mean(img1))
        if angle < 0:
            aux[:-diff, :] = img2
        else:
            aux[diff:, :] = img2
        img2 = aux
    corr = img1 * img2
    return np.sum(corr)

def fill_zeros(section):
    zeros = np.where(section==0)[0]
    for z in zeros:
        section[z] = section[z-1]
    return section

def rotate_image(image, angle):
    mask = image.maskRegistered
    rotAngle = np.radians(angle)
    rotMat = np.identity(2)
    rotMat[0][0] = np.cos(-rotAngle)
    rotMat[0][1] = -np.sin(-rotAngle)
    rotMat[1][0] = np.sin(-rotAngle)
    rotMat[1][1] = np.cos(-rotAngle)
    h, w = mask.shape
    maskRot = np.zeros((h+500, w)).astype(np.float32)
    imageRot = np.zeros((h+500, w)).astype(np.float32)
    # fill finger region with average values
    # maskRot[maskRot == 1] = np.mean(probeimage.imageRegistered[probeMask == 1])
    for column in range(mask.shape[1]):
        fingerRegion = np.where(mask[:, column] == 1)[0]
        r = abs(np.min(fingerRegion) + np.max(fingerRegion)) / 2
        xs = fingerRegion - r
        ys = np.sqrt(np.square(r) - np.square(xs))
        coords = np.vstack((xs, ys))
        coordP = (np.matmul(rotMat, coords))
        if rotAngle > 0:
            xrs = coordP[0] + r#- abs(coordP[0][0] - fingerRegion[0])
        else:
            #x = abs(coordP[0] - fingerRegion[0])
            xrs = coordP[0] + r#+ abs(coordP[0][0] - fingerRegion[0])
        for i, (xr, x) in enumerate(zip(xrs, xs)):
            # print(i)
            imageRot[int(xr), column] = image.imageRegistered[int(x + r), column]
            maskRot[int(xr), column] = 1.0
        xrs = xrs.astype(int)
        imageRot[xrs[0]:xrs[-1], column] = fill_zeros(imageRot[xrs[0]:xrs[-1], column])
        maskRot[xrs[0]:xrs[-1], column] = fill_zeros(maskRot[xrs[0]:xrs[-1], column])

    imgUpperEdge, imgLowerEdge = find_edges(image.maskRegistered, islower=True)
    imgCenterLine = (imgUpperEdge + imgLowerEdge) / 2
    rotUpperEdge, rotLowerEdge = find_edges(maskRot, islower=True)
    rotCenterLine = (rotLowerEdge + rotUpperEdge) / 2

    diff = np.mean(imgCenterLine - rotCenterLine)
    imageRot = shift(imageRot, (diff, 0), mode='constant')[:-500, :]
    maskRot = shift(maskRot, (diff, 0), mode='constant')[:-500, :] > 0.6
    upperEdge, lowerEdge = find_edges(maskRot, islower=True)

    probeRot = Image()
    probeRot.imageRegistered = gaussian_filter(imageRot, sigma=0.7)
    probeRot.maskRegistered = maskRot.astype(bool)
    probeRot.upperEdge = upperEdge[0].astype(int)
    probeRot.lowerEdge = lowerEdge[0].astype(int)

    return probeRot

def find_edges(mask, islower=False):
    maskCopy = mask.astype(int)
    upperEdge = np.zeros((1, maskCopy.shape[1]))
    lowerEdge = np.zeros((1, maskCopy.shape[1]))
    for i in range(mask.shape[1]):
        upperEdge[0][i] = np.where(maskCopy[:, i] == 1)[0][0]
        if islower:
            lowerEdge[0][i] = np.where(maskCopy[:, i] == 1)[0][-1]

    return upperEdge[0].astype(int), lowerEdge[0].astype(int)

def mutual_information(ref_image_crop, cmp_image, bins=256, normed=True):
    """
    :param ref_image_crop: ndarray, cropped image from the center of reference image, needs to be same size as `cmp_image`
    :param cmp_image: ndarray, comparison image data data
    :param bins: number of histogram bins
    :param normed: return normalized mutual information
    :return: mutual information values
    """
    ref_range = (ref_image_crop.min(), ref_image_crop.max())
    cmp_range = (cmp_image.min(), cmp_image.max())
    joint_hist, _, _ = np.histogram2d(ref_image_crop.flatten(), cmp_image.flatten(), bins=bins, range=[ref_range, cmp_range])
    ref_hist, _ = np.histogram(ref_image_crop, bins=bins, range=ref_range)
    cmp_hist, _ = np.histogram(cmp_image, bins=bins, range=cmp_range)
    joint_ent = entropy(joint_hist)
    ref_ent = entropy(ref_hist)
    cmp_ent = entropy(cmp_hist)
    mutual_info = ref_ent + cmp_ent - joint_ent
    if normed:
        mutual_info = mutual_info / np.sqrt(ref_ent * cmp_ent)
    return mutual_info

def entropy(img_hist):
    """
    :param img_hist: Array containing image histogram
    :return: image entropy
    """
    img_hist = img_hist / float(np.sum(img_hist))
    img_hist = img_hist[np.nonzero(img_hist)]
    return -np.sum(img_hist * np.log2(img_hist))

def correlation(datax, datay, lag, part):
    offset = datay[1] + lag[1]#datay[0]
    # if offset < np.abs(lag[0]):
    dataxPad = np.pad(datax, pad_width=(np.abs(lag[0]), np.abs(lag[0])), mode='constant',
                          constant_values=(0, 0))
    maxCorr = 0.0
    shift = 0
    correlation = []
    for i in range(0, len(dataxPad)-100):
        # if part != 1:
        #     break
        corr = pd.Series(dataxPad[i:i+len(datay[0])]).corr(pd.Series(datay[0]))
        correlation.append(corr)
        if corr > maxCorr and (i >= offset - lag[1] and i <= offset+lag[1]):
            maxCorr = corr
            shift = i
    return maxCorr, shift - offset

def split_probe_profile(profile):
    subProfiles = []

    subProfiles.append([profile[0:90], 0])
    subProfiles.append([profile[90:240], 90])
    subProfiles.append([profile[240:], 240])
    return subProfiles

def extract_common_roi(pair):
    refpair = pair[0]
    probepair = pair[1]
    upperEdge = np.max((np.max(refpair.upperEdge), np.max(probepair.upperEdge)))
    lowerEdge = np.min((np.min(refpair.lowerEdge), np.min(probepair.lowerEdge)))
    refroi = refpair.imageNormalised[upperEdge:lowerEdge, :]
    proberoi = probepair.imageRegistered[upperEdge:lowerEdge, :]

    return refroi, proberoi

def sliding_windows(image, win=30, c=10):
    right = win + (c//2)
    left = win + (c//2)
    # image = image.astype(np.float)
    h, w = image.shape
    image = cv2.copyMakeBorder(image, 0, 0, left, right, borderType=cv2.BORDER_REPLICATE).astype(float)
    image = (image - np.min(image)) / (np.max(image - np.min(image)))
    G = []
    for j in range(0, w - win - c//2):
        W1 = image[0:h, j:j+win]
        W2 = image[0:h, j+win+c:j+2*win+c]
        G.append(np.sum(W1) - np.sum(W2))
    jl = np.argmax(G)# - left
    return jl, G

# Time lagged cross correlation
def crosscorr(datax, datay, lag):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    correlation = 0.0
    shift = 0
    # convert datax and datay to pandas Series object
    datax = pd.Series(datax)
    datay = pd.Series(datay)
    for l in lag:
        corr = datax.corr(datay.shift(l))
        if corr > correlation:
            correlation = corr
            shift = l
    return correlation, shift

def pair_correlation_set(patchpair):
    ref = np.copy(patchpair[0]) #/ 255.
    # ref = ref[:, 50:-50]
    probe = np.copy(patchpair[1]).astype(np.float) #/ 255.
    # probe = probe[:, 50:-50]
    patch_correlation = structural_similarity(ref, probe)#np.mean(signal.correlate2d(ref, probe))

    return patch_correlation#np.mean(patch_correlation)

def compute_props_imageset(imageset, histEq=True, correctStripes=True):
    for i, pair in enumerate(imageset):
        imageset[i][0] = compute_properties_image(pair[0], histEq=histEq, correctStripes=correctStripes,
                                                  filter=False)
        # imageset[i][0] = extract_patches_image(pair[0], patchSize=(32,32), slide=(32,32))
        imageset[i][1] = compute_properties_image(pair[1], histEq=histEq, correctStripes=correctStripes,
                                                  filter=False)
        # imageset[i][1] = extract_patches_image(pair[1], patchSize=(32,32), slide=(32,32))
    return imageset

def horizontal_alignment_imageset(imageset):
    for i, pair in enumerate(imageset):
        imageset[i][1] = horizontal_alignment(pair)
    return imageset

def vertical_alingnment_imageset(imageset, savefolder):
    for i, pair in enumerate(imageset):
        imagename = savefolder + '/pair{}.png'.format(i)
        imageset[i][1] = vertical_alignment(pair, imagename)
    return imageset

