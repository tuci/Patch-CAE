import numpy as np
import torch, math, warnings
import argparse, os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import dataloader_unit
from cae_models import CAE_patch
from MaximumCurvature import MaximumCurvature

warnings.filterwarnings("ignore")

# parse command line arguments
parser = argparse.ArgumentParser(description='Train with weighted MSE')
parser.add_argument('traindb', help='Path for train database', nargs='+')
parser.add_argument('valdb', help='Path for validation database', nargs='+')
parser.add_argument('savepath', help='Folder for saved figures', nargs='+')

args = parser.parse_args()
cae_train = args.traindb[0]
cae_val = args.valdb[0]
saveto = args.savepath[0]

if not os.path.exists(saveto):
    os.mkdir(saveto)

# generate grid image for gray scale images
def grid_image(images):
    # make grid image
    b_size, c, h, w = images.shape  # image size
    nCol = 8  # 16 images per row
    # number of images per column
    nRow = int(math.ceil(b_size / nCol))
    padding = 10
    h_grid = nRow * h + (nRow + 1) * padding
    w_grid = nCol * w + (nCol + 1) * padding
    grid_image = np.zeros((h_grid, w_grid))

    # fill grid image
    steph = padding + h
    stepw = padding + w
    rowTop = padding
    colLeft = padding
    for img in images:
        rowBottom = rowTop + h
        colRight = colLeft + w
        grid_image[rowTop:rowBottom, colLeft:colRight] = img[0, :, :]
        colLeft = colLeft + stepw
        if colLeft >= w_grid:
            colLeft = padding
            rowTop = rowTop + steph
    return grid_image

def save_model(model, optimiser, epoch, path_to_model):
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optim_dict': optimiser.state_dict()}
    torch.save(state, path_to_model)

def train(train_loader, model, criterion, optimiser, device):
    # switch to train model
    model.train()
    running_loss = 0.0

    for idx, image in enumerate(train_loader):
        image = image.to(device)
        optimiser.zero_grad()
        output, incurv, outcurv = model(image)
        lossG = criterion(image, output)
        lossC = criterion(incurv, outcurv)
        loss = lossG + lossC
        # compute grads and update
        loss.backward()
        optimiser.step()
        # sum all batch losses
        running_loss += loss.detach().item()
    avr_loss = running_loss / train_loader.__len__()
    return avr_loss

def validate(val_loader, model, criterion, optimiser, device):
    # randomly select a batch to generate grid image
    # rand_batch = np.random.randint(val_loader.__len__(), size=1)[0]
    rand_batch = np.random.randint(5, size=1)[0]
    # switch to train model
    model.eval()
    running_loss = 0.0
    for idx, image in enumerate(val_loader):
        image = image.to(device)
        optimiser.zero_grad()
        output, incurv, outcurv = model(image)
        # save inout and output images if the batch is selected
        if idx == rand_batch:
            input_images_val.append(image.detach().cpu().numpy())
            output_images_val.append(output.detach().cpu().numpy())
            in_curvature.append(incurv.detach().cpu().numpy())
            out_curvature.append(outcurv.detach().cpu().numpy())
        lossG = criterion(image, output)
        lossC = criterion(incurv, outcurv)
        loss = lossG + lossC
        # sum all batch losses
        running_loss += loss.detach().item()

    avr_loss = running_loss / val_loader.__len__()
    return avr_loss

def main():
    num_epochs_cae = 90
    batch_size = 64 
    lr = 1e-3

    #saveto = './figures/train_enhanced/'

    # generate enhanced dataset for each alpha value
    alphas = np.append(np.arange(0.0, 1.0, .05), 1.0)
    for alpha in alphas:
        # create folders for saved files
        alphapath = saveto + '/Alpha{}/'.format(alpha)
        # if not os.path.exists(alphapath):
        #     os.mkdir(alphapath)
        #     os.mkdir(alphapath + '/Validation/')
        #     os.mkdir(alphapath + '/Validation/input/')
        #     os.mkdir(alphapath + '/Validation/output/')
        # # train data for CAE
        #trainset = './database/ut_train/'
        cae_trainset = dataloader_unit(trainset, mode='train')
        cae_trainloader = DataLoader(cae_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # val data for CAE
        #validationset = './database/ut_val/'
        cae_valset = dataloader_unit(validationset, mode='train')
        cae_valloader = DataLoader(cae_valset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(cae_valloader.__len__())

        # select cuda device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'No GPU found')

        # create cae network
        cae_model = CAE_patch(num_layers=6, in_channels=16, disable_decoder=False, sigma=3.5).to(device)

        # optimiser
        optimiser_cae = optim.Adam(cae_model.parameters(), lr=lr, weight_decay=1e-8)

        # cae criterion/loss
        criterion_cae = nn.L1Loss()

        # train CAE
        cae_loss = []
        val_loss = []
        epoch_gave = []
        for epoch in range(num_epochs_cae):
            if np.mod(epoch, 30) == 0 and epoch != 0:
                for group in optimiser_cae.param_groups:
                    group['lr'] = lr / 10
                print('Reducing Lr from {} to {}'.format(lr, lr / 10))
                lr /= 10
            # define input and output images
            input_images_val = []
            output_images_val = []
            in_curvature = []
            out_curvature = []
            # train one epoch
            epoch_loss = train(cae_trainloader, cae_model, criterion_cae, optimiser_cae, device)
            # epoch_gave.append(ave)
            epoch_loss_val = validate(cae_valloader, cae_model, criterion_cae, optimiser_cae, device)
            # for each 10 epochs save figures
            if epoch % 10 == 0 or epoch == num_epochs_cae - 1:
                # save validation images
                input_grid_val = grid_image(input_images_val[0])
                output_grid_val = grid_image(output_images_val[0])
                input_grid_curv = grid_image(in_curvature[0])
                output_grid_curv = grid_image(out_curvature[0])
                # save figures
                figin, axin = plt.subplots(1, 2)
                axin = axin.ravel()
                axin[0].imshow(input_grid_val, cmap='gray')
                axin[1].imshow(output_grid_val, cmap='gray')
                plt.title('Input images - Epoch {}'.format(epoch))
                plt.savefig(alphapath + '/Validation/input/input_grid_epoch{}.png'.format(epoch))
                plt.close(figin)
                #plt.show(block=False)

                figout, axout = plt.subplots(1, 2)
                axout = axout.ravel()
                axout[0].imshow(input_grid_curv, cmap='gray')
                axout[1].imshow(output_grid_curv, cmap='gray')
                plt.title('Output images - Epoch {}'.format(epoch))
                plt.savefig(alphapath + '/Validation/output/output_grid_epoch{}.png'.format(epoch))
                plt.close(figout)
                #plt.show()

                # # save reconstructed images along with the feature space???
                modelfile = alphapath + '/model_cae_epoch{}.pt'.format(epoch)
                save_model(cae_model, optimiser_cae, epoch, modelfile)

            cae_loss.append(epoch_loss)
            val_loss.append(epoch_loss_val)

            # display the epoch training loss
            print("epoch : {}/{}, train loss = {:.4f} validation loss(w) = {:.4f} ".format(epoch + 1, num_epochs_cae,
                                                                                       epoch_loss, epoch_loss_val))


        # plot rain loss of cae
        axEpoch = np.arange(1, num_epochs_cae + 1)
        figaeloss, axaeloss = plt.subplots()
        axaeloss.plot(axEpoch, cae_loss, label='Train loss', color='b')
        axaeloss.plot(axEpoch, val_loss, label='Validation loss', color='r')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.title('CAE Train/Val. Loss')
        plt.savefig(alphapath + '/train_val_loss.png')
        plt.close()
        #plt.show()
