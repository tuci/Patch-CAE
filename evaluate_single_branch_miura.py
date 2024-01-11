import numpy as np
import lmdb, pickle, cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch, os, argparse
import torch.nn.functional as F
import scipy.signal as signal
from cae_models import CAE_patch
from dataloader import dataloader_miura_search
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from process_image_set import compute_edges

# parse command line arguments
parser = argparse.ArgumentParser(description='Dual-branch model evaluation')
parser.add_argument('--database', help='Lmdb Database folder', nargs='+')
parser.add_argument('--modelpath', help='Weight file', nargs='+')
parser.add_argument('--savefolder', help='Folder for saved figures', nargs='+')
parser.add_argument('--ch', help='Vertical displacement(Def. 10)', default=10, type=int)
parser.add_argument('--cw', help='Horizontal displacement(Def. 30)', default=30, type=int)
parser.add_argument('--stride', help='Stride(Def. 15)', default=15, type=int)
parser.add_argument('--fx', help='Vertical scale(Def. 1.0)', default=1.0, type=float)
parser.add_argument('--fy', help='Horizontal scale(Def. 1.0)', default=1.0, type=float)
parser.add_argument('-est', '--estimate', dest='estimate', action='store_true', 
        help='If perform shift parameter estimation')

args = parser.parse_args()
database = args.database[0]
weights = args.modelpath[0]
savefolder = args.savefolder[0]
ch = args.ch
cw = args.cw

def estimate_alignment_params_database(database):
    matedest = []
    nonmatedest = []
    # databases = os.listdir(database)
    # for db in databases:
    data = lmdb.open(database, readonly=True)
    cursor = data.begin(write=False)
    for c in range(data.stat()['entries']):
        pair = pickle.loads(cursor.get(str(c).encode('ascii')))
        refpair, _, _ = extract_roi(pair[0])
        refpair = cv2.resize(refpair, (256, 128))
        probepair, _, _ = extract_roi(pair[2])
        probepair = cv2.resize(probepair, (256, 128))

        im_corr, sh, sw = cross_image(refpair, probepair)
        matedest.append([sh, sw])

    data.close()
    del data
    del cursor 

    _, binsCH = np.histogram(matedest[1], bins=10)
    _, binsCW = np.histogram(matedest[0], bins=10)
    
    ch = np.mean(binsCH[binsCH <-64][-5:])
    cw = np.mean(binsCW[-5:])
    return ch, cw

# extract reagion of interest
def extract_roi(image):
    [upperEdge, lowerEdge] = compute_edges(image)

    upperTanget = np.max(upperEdge)
    lowerTangent = np.min(lowerEdge)

    image = image[upperTanget:lowerTangent, :]

    return image, upperTanget, lowerTangent

def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    cross_im = signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    idx = np.unravel_index(np.argmax(cross_im), cross_im.shape)

    h, w = cross_im.shape[0]//2, cross_im.shape[1]//2

    return cross_im, abs(idx[0] - h), abs(idx[1] - w)

def load_model(model,path, device):
    model_parameters = torch.load(path, map_location=device)
    model_weights = model_parameters['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in model_weights.items():
        state_dict_remove_module[k] = v
    model.load_state_dict(state_dict_remove_module)
    return model.to(device)

def patch_latents(patchmatrix, veinmatrix, model, device): # needs to handle dual-data
    _, m, n, _ , _= patchmatrix.shape
    # define empty latent matrix
    latent_matrix = torch.zeros((m, n, 1, 32))
    for row in range(m):
        for col in range(n):
            if np.isnan(patchmatrix[0][row][col]).all():
                latent_matrix[row][col][0] = torch.nan
                continue
            gray = patchmatrix[0][row][col].unsqueeze(0).unsqueeze(0).float()
            latent_matrix[row][col][0] = model(gray.to(device))
    return latent_matrix

def miura_search(reflatent, probelatent, step_size=1):# will stay the same
    # print('---- Miura Match ----')
    m, n, _, _ = reflatent.shape
    k, l, _, _ = probelatent.shape
    if (m-k) < 0:
        probelatent = probelatent[:m, :, :, :]
        k = m
    n_candidates = (int((m-k) / step_size) + 1) * (int((n-l) / step_size) + 1)
    num_candidate = 0
    image_pair_sim = np.zeros((n_candidates, 3))
    probe_latent = probelatent.view(1, k * l, 1, 32)[0]
    # start = time.time()
    for i in range(0, m-k+1, step_size):
        for j in range(0, n-l+1, step_size):
            ref_candidate = reflatent[i:i+k, j:j+l, :, :]
            ref_candidate = ref_candidate.contiguous().view(1, ref_candidate.shape[0]*ref_candidate.shape[1], 1, 32)[0]
            candidate_sim = []
            for ref, probe in zip(ref_candidate, probe_latent):
                # skip comparison if the reference patch does not exists
                if torch.isnan(ref[0]).all():
                    continue
                sim = F.cosine_similarity(ref, probe).detach().item()
                candidate_sim.append(sim)
            if len(candidate_sim) == 0:
                break
            image_pair_sim[num_candidate][0] = np.mean(candidate_sim)
            image_pair_sim[num_candidate][1] = i
            image_pair_sim[num_candidate][2] = j
            num_candidate += 1
    # end = time.time()
    # print('Comparison time for {} candidate in minutes {:.4f}'.format(num_candidate, (end - start)/60))
    image_pair_sim = np.transpose(image_pair_sim)
    return np.max(image_pair_sim[0, :])

def evaluate(genuine, imposter):
    labels = [1] * len(genuine) + [0] * len(imposter)
    scores = np.append(genuine, imposter)

    # evaluation
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return [fpr, tpr], auc, eer

def compute_similarity_histograms(genuine, imposter):
    scores = np.append(genuine, imposter)
    labels = [1] * len(genuine) + [0] * len(imposter)
    stepSize = .02
    axisRange = np.append(np.arange(0.0, 1.0, stepSize), 1.0)
    nRange = axisRange.size
    axisNGenuine = np.zeros(nRange)
    axisNImposter = np.zeros(nRange)
    for score, label in zip(scores, labels):
        rangeIndex = 0
        while (score > axisRange[rangeIndex]) and (rangeIndex < nRange):
            rangeIndex += 1
        if label == 1:
            axisNGenuine[rangeIndex] += 1.
        else:
            axisNImposter[rangeIndex] += 1.
    nGenuine = np.count_nonzero(labels)
    nImposter = len(labels) - nGenuine
    axisNImposter /= nImposter / 100
    axisNGenuine /= nGenuine / 100

    return axisNGenuine, axisNImposter

if __name__ == '__main__':
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CAE_patch(num_layers=6, in_channels=16, disable_decoder=True).to(device)
    #weights = './models/from_cluster/patch_utfvp/model_cae_64-64_CL2.0_wd10-8_odd_epoch49.pt'
    model = load_model(model, weights, device)
    model.eval()


    # folder for saved figures
    #savefolder = './figures/CAE-single-utfv-miura/'
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
        os.mkdir(savefolder + '/Genuine/')
        os.mkdir(savefolder + '/Imposter/')

    # estimate shift paramters
    if args.estimate:
        ch, cw = estimate_alignment_params_database(database + '/genuine')
    print(ch, cw)

    # data loaders
    matedloader = dataloader_miura_search(database + '/genuine/', patchSize=(64, 64), scale=(args.fx, args.fy),
                                          slide=args.stride, maxshift=(ch, cw), correctEdges=True, mode='eval')
    matedloader = DataLoader(matedloader, batch_size=1, shuffle=False, num_workers=0)
    nonmatedloader = dataloader_miura_search(database + '/imposter/', patchSize=(64, 64), scale=(args.fx, args.fy),
                                             slide=args.stride, maxshift=(ch, cw), correctEdges=True, mode='eval')
    nonmatedloader = DataLoader(nonmatedloader, batch_size=1, shuffle=False, num_workers=0)

    matedscores = []
    nonmatedscores = []
    # loop over dataloader objects
    for i, (refpatches, probepatches) in enumerate(matedloader):
        # get mated pair latent vectors and score
        matedreference = patch_latents(refpatches[0], refpatches[1], model, device)
        matedprobe = patch_latents(probepatches[0], probepatches[1], model, device)
        matedscore = miura_search(matedreference, matedprobe)
        matedscores.append(matedscore)

    for i, (refpatches, probepatches) in enumerate(nonmatedloader):
        # get non-mated pair latent vectors and score
        nonmatedreference = patch_latents(refpatches[0], refpatches[1], model, device)
        nonmatedprobe = patch_latents(probepatches[0], probepatches[1], model, device)
        nonmatedscore = miura_search(nonmatedreference, nonmatedprobe)
        nonmatedscores.append(nonmatedscore)
    # save scores
    np.savetxt(savefolder + '/mated_scores.csv', matedscores, delimiter=',')
    np.savetxt(savefolder + '/nonmated_scores.csv', nonmatedscores, delimiter=',')

    # evaluate
    [fpr, tpr], auc, eer = evaluate(matedscores, nonmatedscores)
    matedhist, nonmatedhist = compute_similarity_histograms(matedscores, nonmatedscores)
    np.savetxt(savefolder + '/matedhist.csv', matedhist, delimiter=',')
    np.savetxt(savefolder + '/nonmatedhist.csv', nonmatedhist, delimiter=',')
    np.savetxt(savefolder + '/fpr.csv', fpr, delimiter=',')
    np.savetxt(savefolder + '/tpr.csv', tpr, delimiter=',')
    np.savetxt(savefolder + '/eer_auc.csv', [eer, auc], delimiter=',')

    print('EER: {:.4f} - AUC: {:.4f} - AVR Mated: {:.4f} AVR Non-mated: {:.4f}'.format(
        eer, auc, np.mean(matedscores), np.mean(nonmatedscores)))

    # print histograms
    xRange = np.append(np.arange(0, 1.0, .02), 1.0)
    fig, ax = plt.subplots()
    ax.plot(xRange, matedhist, label='Mated', c='indigo', linewidth=2.5)
    ax.plot(xRange, nonmatedhist, label='Non-mated', c='darkorange', linewidth=2.7,
            linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count(%)')
    ax.legend(loc='best')
    fig.savefig(savefolder + 'histogram.png')
    fig.set_rasterized(True)
    fig.savefig(savefolder + 'histogram.eps', format='eps', dpi=300)
    plt.close()
