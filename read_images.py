import numpy as np
import os, re
import cv2, glob
import matplotlib.pyplot as plt

class FVImage:
    pass

def read_images_SD(pathData, numSample=90, mode='cae', fx=2.5, fy=2.0):
    # read images from SDUMLA-HMT database
    # numSamples - number of subjects selected for train 
    #		  > integer - defult 90:q   
    
    # read number of subjects
    subjects = os.listdir(pathData)
    # select subjects based on partition
    if mode == 'cae':
         subjects = subjects[-numSample:]
    elif mode == 'eval':
         subjects = subjects[:numSample]

    imageset = []
    # imageset.append([])
    imgid = -1
    for subject in range(len(subjects)):
        # sdumla stores each hands in different folders
        # read hands one-by-one
        handpath = pathData + subjects[subject] + '/'
        hands = os.listdir(handpath)
        for h, hand in enumerate(hands):
            # get all images
            imagepath = handpath + hand + '/'
            images = os.listdir(imagepath)
            fingers = {'index': 0, 'middle': 1, 'ring': 2}
            fngr = ''
            imagesPerFinger = []
            # loop over all images
            for im, image in enumerate(images):
                namesplit = re.split('[_ .]', image)
                if namesplit[-1] != 'bmp':
                    continue
                finger = namesplit[0]
                # if fngr != finger:
                #     imageset.append(imagesPerFinger)
                #     imagesPerFinger = []
                #     fngr = finger
                img = FVImage()
                img.image = cv2.resize(cv2.imread(imagepath + image, 0)[:, 50:-50],
                        (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
                img.f = fingers[finger] + (3 * h)
                img.p = int(subjects[subject])
                img.i = int(namesplit[1])
                if fngr == finger:
                    imageset[imgid].append(img)
                else:
                    imgid += 1
                    imageset.append([])
                    imageset[imgid].append(img)
                    fngr = finger
                # imagesPerFinger.append(img)
                # else:
                #     imageset.append(imagesPerFinger)
                #     imagesPerFinger = []
                #     imagesPerFinger.append(img)
                #     fngr = finger
                # # check if it is a new finger
                # if img.i == 6:
                #     # new finger
                #     imageset[imgid].append(img)
                #     imgid += 1
                #     imageset.append([])
                #     # fngr = finger
                # else:
                #     imageset[imgid].append(img)
    return imageset

def read_images_SD_cleared(pathData, numSample=90, mode='cae'):
    # read images from SDUMLA-HMT database
    # numSamples - number of subjects selected for train
    #		  > integer - defult 90:q

    # read number of subjects
    subjects = os.listdir(pathData)
    # select subjects based on partition
    if mode == 'cae':
        subjects = subjects[:numSample]
    elif mode == 'eval':
        subjects = subjects[-numSample:]

    imageset = []
    # imageset.append([])
    imgid = 0
    for subject in range(len(subjects)):
        # sdumla stores each hands in different folders
        # read hands one-by-one
        handpath = pathData + subjects[subject] + '/'
        hands = os.listdir(handpath)
        for h, hand in enumerate(hands):
            # get all images
            imagepath = handpath + hand + '/'
            subfolders = os.listdir(imagepath)
            fingers = {'index': 0, 'middle': 1, 'ring': 2}
            # fngr = ''
            for subfolder in subfolders:
                namesplit = re.split('[.]', subfolder)
                if namesplit[-1] == 'db':
                    continue
                imageset.append([])
                images = os.listdir(imagepath + subfolder)
                # loop over all images
                for im, image in enumerate(images):
                    namesplit = re.split('[_ .]', image)
                    finger = namesplit[0]
                    img = FVImage()
                    img.image = cv2.imread(imagepath + subfolder + '/' + image, 0)[:, 50:-50]
                    img.f = fingers[finger] + (3 * h)
                    img.p = int(subjects[subject])
                    img.i = int(namesplit[1])
                    # if fngr == finger:
                    imageset[imgid].append(img)
                    # else:
                imgid += 1
                        # imageset.append([])
                        # imageset[imgid].append(img)
                        # fngr = finger
    return imageset

def read_images_UT(pathData, numSample=20, mode='cae'):
    # read images from pathData
    # Parameters:
    #   pathData - path to finger vein images
    #       > string
    #   numSample - number of subjects to be read from
    #       the dataset
    #       > integer: default - 20
    #
    # Created on: 1-7-2019

    # finger image struct list
    fingersPerPerson = []
    # get the folders in pathData
    imageFolders = os.listdir(pathData)
    if mode == 'cae':
        imageFolders = imageFolders[0:numSample]
    elif mode == 'eval':
        imageFolders = imageFolders[-numSample:]

    # Replace len(imageFolders) with numSamples
    for fldr in range(len(imageFolders)):
        # get the images in the folder of a subject
        imagesInFolder = os.listdir(pathData + imageFolders[fldr])
        imagesPerFinger = []
        fingerId = 1
        for img in range(len(imagesInFolder)-1):
            # image path
            imgPath = pathData + '/' + imageFolders[fldr] + '/' + imagesInFolder[img]
            fvImage = FVImage()
            fvImage.image = cv2.imread(imgPath,0)
            fvImage.p = fldr + 1
            fvImage.f = int(imagesInFolder[img].split('_')[1])
            fvImage.i = int(imagesInFolder[img].split('_')[2])
            # store the image belonging to the same finger together
            imagesPerFinger.append(fvImage)
            fingerId += 1
            if fingerId > 4:
                fingerId = 1
                # store all the finger images belonging to the same finger together
                fingersPerPerson.append(imagesPerFinger)
                imagesPerFinger = []

    return fingersPerPerson

def read_images_CCFVP(pathData, numSample=20, mode='cae'):
    # read images from pathData
    # Parameters:
    #   pathData - path to finger vein images
    #       > string
    #   numSample - number of subjects to be read from
    #       the dataset
    #       > integer: default - 20
    #
    # Created on: 1-7-2019

    # finger image struct list
    fingersPerPerson = []
    # get the folders in pathData
    imageFolders = os.listdir(pathData)
    if mode == 'cae':
        imageFolders = imageFolders[0:numSample]
    elif mode == 'eval':
        imageFolders = imageFolders[-numSample:]

    # Replace len(imageFolders) with numSamples
    for fldr in imageFolders:
        if not os.path.isdir('{}/{}/'.format(pathData, fldr)):
            continue
        # get the images in the folder of a subject
        fingersInFolder = os.listdir(pathData + fldr)
        imagesPerFinger = []
        fingerId = 1
        for finger in fingersInFolder:
            fingerPath = '{}/{}/{}/'.format(pathData, fldr, finger)
            fingerimages = glob.glob(fingerPath + '/*.png')
            for i, img in enumerate(fingerimages):
                # image path
                fvImage = FVImage()
                fvImage.image = np.rot90(cv2.imread(img, 0), k=3)[500:-500, 60:-80]
                fvImage.p = int(fldr)
                fvImage.f = int(finger)
                fvImage.i = i + 1
                # store the image belonging to the same finger together
                imagesPerFinger.append(fvImage)
                fingerId += 1
                if fingerId > 3:
                    fingerId = 1
                    # store all the finger images belonging to the same finger together
                    fingersPerPerson.append(imagesPerFinger)
                    imagesPerFinger = []

    return fingersPerPerson

def read_images_MMCBNU(path, numSample=20, mode='cae'):
    # finger image struct list
    fingersPerPerson = []
    # get the folders in pathData
    subjects = os.listdir(path)
    if mode == 'cae':
        subjects = subjects[0:numSample]
    elif mode == 'eval':
        subjects = subjects[-numSample:]

    # get fingers for each subject
    for s, subject in enumerate(subjects):
        fingers = os.listdir(path + subject)
        # fingersPerPerson.append([])
        imagesPerFinger = []
        # get all images for each finger
        for f, finger in enumerate(fingers):
            imagesPerFinger.append([])
            images = os.listdir(path + '/{}/{}/'.format(subject, finger))
            # read all images
            for i, image in enumerate(images):
                fvimage = FVImage()
                fvimage.image = cv2.imread(path + '/{}/{}/{}'.format(subject, finger, image), 0)[:, 0:-1]
                fvimage.p = s + 1
                fvimage.f = f + 1
                fvimage.i = i + 1
                imagesPerFinger[f].append(fvimage)
        fingersPerPerson.extend(imagesPerFinger)
    return fingersPerPerson

def read_images_FVUSM(datapath, numSample=20, mode='cae'):

    sessions = os.listdir(datapath)[:2]
    session1 = os.listdir(datapath + sessions[0] + '/raw_data/')
    session2 = os.listdir(datapath + sessions[1] + '/raw_data/')
    if mode == 'cae':
        session1 = session1[:numSample * 4]
        session2 = session2[:numSample * 4]
    elif mode == 'eval':
        session1 = session1[-(numSample*4):]
        session2 = session2[-(numSample*4):]

    fingerPerPerson = []
    for s1, s2 in zip(session1, session2):
        ses1 = os.listdir(datapath + sessions[0] + '/raw_data/' + s1)
        ses2 = os.listdir(datapath + sessions[1] + '/raw_data/' + s2)
        imagePerFinger = []
        for img1, img2 in zip(ses1, ses2):
            ses1img = cv2.imread(datapath + sessions[0] + '/raw_data/' + s1 + '/' + img1, 0)
            subj, finger = re.split('[_]', s1)
            img = re.split('[.]', img1)[0]
            # first session
            fvsession1 = FVImage()
            fvsession1.image = np.flip(np.transpose(ses1img), 0)[150:-150, 10:-80]
            fvsession1.p = int(subj)
            fvsession1.f = int(finger)
            fvsession1.i = int(img)
            imagePerFinger.append(fvsession1)
            # second session
            ses2img = cv2.imread(datapath + sessions[1] + '/raw_data/' + s2 + '/' + img2, 0)
            fvsession2 = FVImage()
            fvsession2.image = np.flip(np.transpose(ses2img), 0)[150:-150, 10:-80]
            fvsession2.p = int(subj)
            fvsession2.f = int(finger)
            fvsession2.i = int(img) + len(ses1)
            imagePerFinger.append(fvsession2)
        fingerPerPerson.append(imagePerFinger)
    return fingerPerPerson

def read_images_PKU(datapath, numSample=50, mode='cae', fx=1.0, fy=1.0):
    subjects = os.listdir(datapath)
    if mode == 'cae':
        subjects = subjects[:numSample]
    else:
        subjects = subjects[-numSample:]

    # loop over subjects
    imagesPerSubject = []
    for s, sub in enumerate(subjects):
        # each subject has only one finger for the ease of processing
        # add an entry for the finger in imagesPerSubject array
        imagesPerSubject.append([])
        images = glob.glob(datapath + sub + '/*.bmp')
        # loop over images
        for i, image in enumerate(images):
            imageSet = FVImage()
            imageSet.image = cv2.resize(cv2.imread(image, 0), (0, 0), fx=fx, fy=fy,
                    interpolation=cv2.INTER_AREA)
            imageSet.p = s
            imageSet.f = 1# single finger is captured per subject
            imageSet.i = i
            imagesPerSubject[s].append(imageSet)

    return imagesPerSubject

if __name__ == '__main__':
    path = './dataset/Peking/'
    images_train = read_images_PKU(path, mode='cae')
    print()
