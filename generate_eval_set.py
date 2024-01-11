# import numpy as np
import os, argparse, time
from read_images import *
from process_image_set import compute_properties_image_set
from generate_pairs import generate_genuine_pairs, select_object_images
from register_images import register_images
from fine_register_images import horizontal_alignment_windows, correct_rotation, vertical_alignment
from MaximumCurvature import MaximumCurvature
from lmdb_database import make_lmdb_utfvp

# # parse command line arguments
parser = argparse.ArgumentParser(description='Dual-branch evaluation datasets')
parser.add_argument('--image_path', help='Folder for raw images', nargs='+', type=str)
parser.add_argument('--database', help='Lmdb database folder', nargs='+', type=str)
parser.add_argument('--n_sample', help='Number of subjects', nargs='+', type=int)
parser.add_argument('--n_probe', help='Number of imposter pairs per image(Def. 3)', default=3, type=int)
parser.add_argument('--is_align', help='Fine alignment(Def. False)', default=False, type=bool)
parser.add_argument('--sigma', help='Maximum Curvature sigma value(Def. 4.0)', default=4.0, type=float)

args = parser.parse_args()
path = args.image_path[0]
database = args.database[0]
nProbe = args.n_probe
is_align = args.is_align
sigma = args.sigma
n_sample = args.n_sample[0]

def select_object_images_pku(imageSet, setIndex, nObjectImages=3):
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
    fingers = []
    images = []
    for i, lst in enumerate(setList):
        fingers.extend(np.repeat(setList[i],len(imageSet[lst])))
        images.extend(np.arange(0,len(imageSet[lst])))
    indexList = [fingers, [images]]

    # select nObjectImages indices
    permutedOrder = np.random.permutation(len(indexList[0]))
    # indexListSelected = indexList[:,permutedOrder[0:nObjectImages]]

    for i in range(nObjectImages):
        fIdx = indexList[0][permutedOrder[i]]
        imIdx = indexList[1][0][permutedOrder[i]]
        objectImages.append(imageSet[fIdx][imIdx])

    return objectImages

def generate_image_pairs(imageset, nProbe=3):
    # only ICP registration
    genuine_pairs = generate_genuine_pairs(imageset)

    imposter_pairs = []
    pairid = 0
    # genuine_pairs = []
    for refID in range(len(imageset)):
        set = imageset[refID]
        for refimage in set:
            nProbe = len(set) - 1
            object_images = select_object_images_pku(imageset, refID, nProbe)
            for objimage in object_images:
                if refimage.p == objimage.p and refimage.f == objimage.f:
                    print('Genuine in imposter set!!')
                    return 0
                # only ICP registration
                objimage = register_images(refimage, objimage)
                imagepair = [refimage, objimage]
                pairid += 1
                # add image pair to array
                imposter_pairs.append(imagepair)

    return genuine_pairs, imposter_pairs

# apply fine alignment on the generated pairs
def align_image_pairs(imagepairs):
    for i, pair in enumerate(imagepairs):
        print(i)
        refpair, probepair = horizontal_alignment_windows(pair)
        probepair = correct_rotation([refpair, probepair])
        probepair = vertical_alignment([refpair, probepair])
        pair = [refpair, probepair]
    return imagepairs

def extract_veins(pairset, sigma=5.0):
    # maximum curvature object
    mc = MaximumCurvature(sigma=sigma)
    # gray-vein image pairs
    imagepairs = []
    for i, pair in enumerate(pairset):
        # start = time.time()
        # if i == 5:
        #     break
        refveins = mc.__call__(pair[0].imageNormalised, pair[0].maskNormalised)
        probeveins = mc.__call__(pair[1].imageRegistered, pair[1].maskRegistered)
        imagepairs.append([pair[0].imageNormalised, refveins, pair[1].imageRegistered, probeveins])
        # end = time.time()
        # print('Vein extraction per image: {:.4f}'.format((end - start)/60))
    return imagepairs

if __name__ == '__main__':
    # read images
    path = './dataset/SDUMLA-HMT/'
    n_sample = 2
    images = read_images_SD(path, numSample=n_sample, mode='eval')
    images = compute_properties_image_set(images, histEq=True, correctStripes=False)

    # generate pairs
    nProbe = 6
    genuinepairs, imposterpairs = generate_image_pairs(images, nProbe=nProbe)

    # align image pairs
    is_align = True
    if is_align:
        genuinepairs = align_image_pairs(genuinepairs[:10])
        imposterpairs = align_image_pairs(imposterpairs[:10])

    # generate lmdb databases
    #database = './database/sdumla_eval/'
    if not os.path.exists(database):
        os.mkdir(database)

    genuinedb = '{}/{}/'.format(database, '/genuine/')
    make_lmdb_utfvp(genuinepairs, genuinedb, network='nn', mapsize=1e8)

    imposterdb = '{}/{}/'.format(database, '/imposter/')
    make_lmdb_utfvp(imposterpairs, imposterdb, network='nn', mapsize=1e8)