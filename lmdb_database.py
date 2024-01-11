import numpy as np
import lmdb, os, re, cv2
import pickle
from read_images import read_images_UT
from process_image_set import compute_properties_image_set
from augmentation import crop_image, flip_image
from MaximumCurvature import MaximumCurvature

# read images store subject and finger numbers
# convert subject and finger to binary for labels
# different sessions will have the same labels with the first one

# parse command line arguments

# define image class
class Image:
    pass

# FV_USM database with labels
def make_lmdb_database_fvusm(datapath, dbpath, network):
    # define train and validation set via 'partititon' parameter
    # determine the number of samples on train set via 'numSample' parameter
    # the remaining samples will be on validation set

    # open database
    trainpath = dbpath + 'train/'
    valpath = dbpath + 'val/'
    testpath = dbpath + 'test/'

    databasetrain = lmdb.open(trainpath, map_size=1e8)
    databaseval = lmdb.open(valpath, map_size=1e8)
    databasetest = lmdb.open(testpath, map_size=1e8)

    # generate database keys
    numkeys = 123 * 4 * 6
    keys = iter(np.random.permutation(numkeys))
    # get folders
    # sessions are stored in different folders
    sessions = os.listdir(datapath)

    if network == 'cae':
        # read only first session data
        session = sessions[0]
    elif network == 'cnn':
        # read only the second session data
        session = sessions[1]
 
    # split folders
    # 2/3 train, 1/6 validation, 1/6 test
    subfolder = 'extractedvein' # read only roi images !!!!!!DO NOT FORGET TO ADD THIS!!!!!!
    folders = os.listdir(datapath + '/' + session + '/' + subfolder)
    trainlen = int(len(folders) * 2 / 3)
    vallen = trainlen + int(len(folders) / 6)
    train_folder = folders[0:trainlen]
    val_folder = folders[trainlen:vallen]
    test_folder = folders[vallen:]

    # locate cursor to train lmdb
    with databasetrain.begin(write=True) as txn_train:
        # get finger image folders
        # loop over all folders
        # extract subject and finger info( will be used for labeling )
        imagepath = datapath + '/' + session + '/' + subfolder +'/'
        for folder in train_folder:
            namesplit = re.split('[_]', folder)
            subject = re.sub('\D', '', namesplit[0])
            finger = namesplit[1]
            label = int(subject+finger)
            images = os.listdir(imagepath + '/' + folder)
            # loop over all images of the subject and the finger
            for image in images:
                filename = imagepath + '/' + folder + '/' + image
                if re.split('[/_.]', filename)[-1] != 'jpg':
                    print('filename: {}'.format(filename))
                    continue
                image = cv2.resize(cv2.imread(filename), (50, 150), interpolation=cv2.INTER_CUBIC)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
                image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
                image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
                data = (image, label)
                key = str(next(keys))
                txn_train.put(key.encode('ascii'), pickle.dumps(data))
    
    # locate cursor to validation lmdb
    with databaseval.begin(write=True) as txn_val:
        # get finger image folders
        # loop over all folders
        # extract subject and finger info( will be used for labeling )
        for folder in val_folder:
            namesplit = re.split('[_]', folder)
            subject = re.sub('\D', '', namesplit[0])
            finger = namesplit[1]
            label = int(subject+finger)
            images = os.listdir(imagepath + '/' + folder)
            # loop over all images of the subject and the finger
            for image in images:
                filename = imagepath + '/' + folder + '/' + image
                if re.split('[/_.]', filename)[-1] != 'jpg':
                    continue
                image = cv2.resize(cv2.imread(filename), (50, 150), interpolation=cv2.INTER_CUBIC)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image[:, :, 0] = (image[:, :, 0])
                image[:, :, 1] = (image[:, :, 1])
                image[:, :, 2] = (image[:, :, 2])
                data = (image, label)
                key = str(next(keys))
                txn_val.put(key.encode('ascii'), pickle.dumps(data))

    # locate cursor to test lmdb
    with databasetest.begin(write=True) as txn_test:
        # get finger image folders
        # loop over all folders
        # extract subject and finger info( will be used for labeling )
        for folder in test_folder:
            namesplit = re.split('[_]', folder)
            subject = re.sub('\D', '', namesplit[0])
            finger = namesplit[1]
            label = int(subject+finger)
            images = os.listdir(imagepath + '/' + folder)
            # loop over all images of the subject and the finger
            for image in images:
                filename = imagepath + '/' + folder + '/' + image
                if re.split('[/_.]', filename) != 'jpg':
                    continue
                image = cv2.resize(cv2.imread(filename), (50, 150), interpolation=cv2.INTER_CUBIC)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image[:, :, 0] = (image[:, :, 0])
                image[:, :, 1] = (image[:, :, 1])
                image[:, :, 2] = (image[:, :, 2])
                data = (image, label)
                key = str(next(keys))
                txn_test.put(key.encode('ascii'), pickle.dumps(data))

def make_lmdb_utfvp(images, dbpath, network, mapsize=None):
    # generate databases for cae-cnn system
    # images - either patches or patch/image pair
    # dbpath - base path to database
    # network - type of network to be the db generated
    #           cae or cnn
    #           if cae is selected only patches are stored
    #           if cnn is selected patches/images are paired and
    #               stored along with their label(genuine - 1, imposter - 0)

    # open database
    if not os.path.exists(dbpath):
        db = lmdb.open(dbpath, map_size=mapsize)
        numkeys = len(images)
        keys = np.random.permutation(numkeys).astype(int)
    else:
        db = lmdb.open(dbpath)
        lendb = db.stat()['entries']
        numkeys = len(images)
        keys = np.random.permutation(numkeys).astype(int) + lendb
    if network == 'cae':
        imgindex = 0
        try:
            with db.begin(write=True) as txn:
                for idx, (image, key) in enumerate(zip(images, keys)):
                    key = str(key)
                    graypatch = image[0].astype(np.uint8)
                    curvpatch = image[1].astype(np.float16)
                    image = [graypatch, curvpatch]
                    txn.put(key.encode('ascii'), pickle.dumps(image))
                    imgindex = idx

        except lmdb.MapFullError:
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
            db.set_mapsize(new_limit)

            continue_make_lmdb(images, keys, db, imgindex)

    elif network == 'nn':
        imageindex = 0
        keys = np.random.permutation(len(images))
        # add pairs to database
        try:
            with db.begin(write=True) as txn:
                for idx, (pair, key) in enumerate(zip(images, keys)):
                    key = str(key)
                    txn.put(key.encode('ascii'), pickle.dumps(pair))
                    imageindex = idx

        except lmdb.MapFullError:
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
            db.set_mapsize(new_limit)

            continue_make_lmdb(images, keys, db, imageindex)
    db.close()

def make_lmdb_utfvp_patch(images, dbpath, network, mapsize=None):
    # generate databases for cae-cnn system
    # images - either patches or patch/image pair
    # dbpath - base path to database
    # network - type of network to be the db generated
    #           cae or cnn
    #           if cae is selected only patches are stored
    #           if cnn is selected patches/images are paired and
    #               stored along with their label(genuine - 1, imposter - 0)

    patches = []
    for finger in images:
        for image in finger:
            patches.extend(image.patches.patches)
    if network == 'cae':
        # open database
        if not os.path.exists(dbpath):
            db = lmdb.open(dbpath, map_size=mapsize)
            numkeys = len(patches)
            keys = np.random.permutation(numkeys)
        else:
            db = lmdb.open(dbpath)
            lendb = db.stat()['entries']
            numkeys = len(patches)
            keys = np.random.permutation(numkeys) + lendb

        imgindex = 0
        try:
            with db.begin(write=True) as txn:
                for idx, (patch, key) in enumerate(zip(patches, keys)):
                    key = str(key)
                    txn.put(key.encode('ascii'), pickle.dumps(patch))
                    imgindex = idx

        except lmdb.MapFullError:
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
            db.set_mapsize(new_limit)

            continue_make_lmdb(patches, keys, db, imgindex)

    elif network == 'nn':
        # open database
        db = lmdb.open(dbpath, map_size=mapsize)
        keys = np.random.permutation(len(patches))
        # add pairs to database
        with db.begin(write=True) as txn:
            for (pair, key) in zip(patches, keys):
                # check how to read labels
                label = pair[2]
                imgpair = [pair[0], pair[1]]
                key = str(key)
                data = (imgpair, label)
                txn.put(key.encode('ascii'), pickle.dumps(data))

def continue_make_lmdb(data, keys, database, index):
    imgindex = 0
    try:
        with database.begin(write=True) as txn:
            for idx, (image, key) in enumerate(zip(data, keys), start=index+1):
                key = str(key)
                txn.put(key.encode('ascii'), pickle.dumps(image))
                imgindex = idx

    except lmdb.MapFullError:
        curr_limit = database.info()['map_size']
        new_limit = curr_limit * 2
        print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
        database.set_mapsize(new_limit)

        continue_make_lmdb(data, keys, database, imgindex)


def extract_veins(imageset):
    MS = MaximumCurvature(sigma=3)
    for set in imageset:
        for img in set:
            img.veinprobs = MaximumCurvature.__call__(MS, img.imageNormalised,
                                                      img.maskNormalised)
    return imageset
