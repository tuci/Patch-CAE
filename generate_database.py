import argparse
from process_image_set import *
from lmdb_database import make_lmdb_utfvp
from extract_patches_image_set import extract_patches_image_set
from read_images import read_images_UT
from MaximumCurvature import MaximumCurvature

# # parse command line arguments
parser = argparse.ArgumentParser(description='Generate databases')
parser.add_argument('--datapath', help='image path', nargs='+', type=str)
parser.add_argument('--dbpath', help='database path', nargs='+', type=str)
parser.add_argument('--sigma', help='Sigma value for vein extraction(def. 5.0)', default=5.0, type=float)
parser.add_argument('--numsubject', help='Number of subjects(def. 74)', default=74, type=int)

args = parser.parse_args()
datapath = args.datapath[0]
dbpath = args.dbpath[0]

def generate_lmdb_patch(subjects, dbpath, psize=64, slide=25):
    # compute image properties
    props = compute_properties_image_set(subjects, histEq=True, correctStripes=False)

    trainlen = (len(props) * 2) // 3
    trainpartition = props[0:trainlen]

    # extact patch pairs
    trainpartition = extract_patches_image_set(trainpartition, patchSize=(psize, psize),
                                        slide=(slide, slide))

    # prepare patchset
    patchset = patch_set(trainpartition)

    print('Num patches {}'.format(len(patchset)))
    # save patches in database
    # flip patches
    patches = flip_patches(patchset)
    print('# patches - train(after flip): {}'.format(len(patches)))

    make_lmdb_utfvp(patches, dbpath + 'ut_train', network='cae', mapsize=1e8)

    # validation part
    valpartition = props[trainlen:]
    valpartition = extract_patches_image_set(valpartition, patchSize=(psize, psize),
                                        slide=(slide, slide))
    # prepare patchset
    patchset = patch_set(valpartition)
    print('# patches - val.: {}'.format(len(patchset)))
    # generate database
    make_lmdb_utfvp(patches, dbpath=dbpath + 'ut_val', network='cae',
                    mapsize=1e8)

def patch_set(imageset):
    patchset = []
    for finger in imageset:
        for image in finger:
            patchset.extend(image.patches.patches)
    return patchset

def flip_patches(patchset):
    copyarray = np.copy(patchset)
    for patch in patchset:
        flipgray = np.expand_dims(cv2.flip(patch, 0), axis=0)
        copyarray = np.append(copyarray, flipgray, axis=0)
    return copyarray

if __name__ == '__main__':
    # read images
    #datapath = './dataset/Twente/dataset/data/'
    images = read_images_UT(datapath, numSample=args.numsubject)
    # cae subjects
    generate_lmdb_patch(images, dbpath=dbpath, psize=65, slide=65)
