##### train a model

import argparse
import shutil
import torch
import util.import_functions as import_functions
import numpy as np
import torch.optim as optim
import os
from losses.loss_functions import Aleatoric_Loss
from network_architecture import UNet3d
import time
import multiprocessing
import pandas as pd
import glob
import visdom

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
help='path to latest checkpoint (default: none)')

parser.add_argument('--epochs', default=8000, type=int, metavar='N',
help='number of total epochs to run')

parser.add_argument('--iterations', default='100', type=int,
                    metavar='N', help='iterations per dataset (default: 10)')

parser.add_argument('--batch_size', default='4', type=int,
                    metavar='N', help='Batches loaded in RAM')

parser.add_argument('--input_directory', default='./data', type=str, metavar='DIR', help='directory name')

parser.add_argument('--output_directory', default='./data_output', type=str, metavar='DIR', help='directory name')

parser.add_argument('--device', default='cuda:0', type=str, help='Device name')

parser.add_argument('--channels', default='5', type=int, help='Number of channels of the input data')

parser.add_argument('--use_visdom', default=False, type=bool, help='Make use of visdom')

parser.add_argument('--use_cuda', default=False, type=bool, help='Use Cuda True or False')


def save_checkpoint(state, save_location, filename='initial_parameters.pth.tar'):
    torch.save(state, os.path.join(save_location, filename))

class train_model:
    def __init__(self, model):
        self.model = model
        # TODO: aus config datei paths holen

    def save_checkpoint(self,state, is_best_mse, save_location, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(save_location, filename))
        if is_best_mse:
            shutil.copyfile(os.path.join(save_location, filename), os.path.join(save_location, 'model_best.pth.tar'))

    def swapaxis(self, matrix):
        swapped = np.swapaxes(matrix, 0, 3)
        matrix = np.squeeze(swapped, 3)
        del swapped
        return(matrix)

    def train(self, data_path, criterion, optimizer, device, save_location, n_channels,
              BATCH_SIZE=4, scheduler=None, epochs=800, use_visdom=True):

        self.model.train()
        train_list = glob.glob(os.path.join(data_path, 'train', '*'))
        validation_list = glob.glob(os.path.join(data_path, 'validation', '*'))

        # Define the dataloaders
        n_cpus = multiprocessing.cpu_count()
        train_cpus = np.int8(np.floor(n_cpus*0.6))
        val_cpus = np.int8(np.floor(n_cpus*0.3))

        if use_visdom:
            vis = visdom.Visdom()
            id_train = None
            id_val = None

        a = import_functions.augmentation3d(train_list, n_channels=n_channels, BATCH_SIZE=BATCH_SIZE,
                                            spatial_transform=False, shape_data=(96, 96, 96))
        a_1 = a.transform(cores=train_cpus, use_cuda=args.use_cuda)

        a_validation = import_functions.augmentation3d(validation_list, n_channels=n_channels, BATCH_SIZE=BATCH_SIZE,
                                                       spatial_transform=False, shape_data=(96, 96, 96))
        a_val = a_validation.transform(cores=val_cpus, use_cuda=args.use_cuda)

        # Define training parameters
        error = []
        error_val = []
        converged = False  # Will be set to True if model did converge
        is_best_score = False
        best_training_loss = 100
        count_no_improvement = 0
        error_val_best = 100000

        # Start training
        for epoch in range(epochs):
            # Training
            error_train_tmp = []
            count = 0
            for data in a_1:
                time_start = time.time()
                optimizer.zero_grad()
                output = self.model(data['data'].to(device))
                errG = criterion(output, data['seg'][:, 0:1].to(device), data['seg'][:, 1:2].to(device))
                if errG is None:
                    continue
                print(errG.item())
                error_train_tmp.append(errG.item())
                errG.backward()
                optimizer.step()
                time_end = time.time()
                diff = time_end - time_start
                print('time difference:', diff)

                # Calculate mean training error
                if count >= 400 and not converged:
                    mean_error = np.mean(error_train_tmp)
                    error.append(mean_error)
                    if use_visdom:
                        id_train = vis.line(error, np.arange(len(error)), win=id_train, opts=dict(title='Training'))
                    savestring = os.path.join(save_location, 'model_tmp.pkl')
                    print('saving model...')
                    torch.save(self.model, savestring)
                    torch.save(error, os.path.join(save_location, 'error_array.pkl'))
                    break
                count += 1

            # Check for convergence:
            if mean_error < (best_training_loss-0.001):
                best_training_loss = mean_error
                count_no_improvement = 0
            else:
                count_no_improvement += 1
            if count_no_improvement > 10:  # Model converged
                print('model converged')
                return(self.model)

            # Validation:
            with torch.set_grad_enabled(False):
                print('validating')
                error_val_tmp = []
                count_val = 0
                for data in a_val:
                    output = self.model(data['data'].to(device))
                    errG = criterion(output, data['seg'][:, 0:1].to(device), data['seg'][:, 1:2].to(device))
                    error_val_tmp.append(errG.item())
                    count_val += 1
                    if count_val >= 10:
                        mean_val_error = np.mean(error_val_tmp)
                        error_val.append(mean_val_error)
                        is_best_score = mean_val_error < error_val_best
                        if is_best_score:
                            torch.save(self.model, os.path.join(save_location, 'model_best.pkl'))
                            error_val_best = error_val[-1]
                        if use_visdom:
                            id_val = vis.line(error_val, np.arange(len(error_val)), win=id_val, opts=dict(title='Validation'))
                        torch.save(error_val, os.path.join(save_location, 'error_validation_array.pkl'))
                        break

            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scores': {
                    'train_score': error,
                    'validation_score': error_val
                }
            }, is_best_mse=is_best_score, save_location=save_location)

        return self.model


if __name__=='__main__':
    args = parser.parse_args()
    cross_val = False

    if args.use_cuda:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    # Create blank model
    n_channels = args.channels
    first_layers = 4
    n_outputs = 1  # 1 MRI-image is predicted together with 1 uncertainty estimation
    model = UNet3d(n_channels, n_outputs, first_layers).to(device)

    save_location = args.output_directory
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    else:
        print('Warning: directory already exists. Saved data might be overwritten')
        print('directory:', save_location)
        input('Press enter to continue anyway')
        time.sleep(1)

    criterion = Aleatoric_Loss()

    lr_decreasing_step = 1
    optimizer = optim.Adam(model.parameters(), lr=10e-5, eps=1e-4, amsgrad=True, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decreasing_step)

    z = 0
    scores = None

    # load a checkpoint if it exists
    if args.resume:
        print(args.resume)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)

            z = checkpoint['epoch']
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('optimizer loaded')
            except Exception:  # too broad, I know
                optimizer = optim.Adam(model.parameters(), lr=10e-5, eps=1e-4, amsgrad=True, weight_decay=1e-5)

            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decreasing_step)
            model.load_state_dict(checkpoint['state_dict'])
            scores = checkpoint['scores']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model_trainer = train_model(model.train())
    del model

    initial_dict={'label_batch': False,  # TODO: explanation
                  'difference': True,
                  'batchnorm': True,
                  'BATCH_SIZE': args.batch_size,
                  'iterations':  args.iterations,
                  'epoch': z,
                  'optimizer': optimizer.state_dict(),
                  'lr_decreasing': lr_decreasing_step,
                  'criterion': criterion
                  }

    save_checkpoint(initial_dict, save_location)

    df = pd.DataFrame.from_dict([initial_dict])
    df.to_csv(os.path.join(save_location, 'initial_params.txt'), header=False, index=True, mode='a')

    # Location of the data
    data_path = args.input_directory
    print('data path:', data_path)
    model = model_trainer.train(data_path=data_path, scheduler=scheduler, n_channels=n_channels, BATCH_SIZE=initial_dict['BATCH_SIZE'],
                                criterion=criterion,
                                device=device,
                                optimizer=optimizer, save_location=save_location, epochs=args.epochs, use_visdom=args.use_visdom)
    location_final_model = os.path.join(save_location, 'model_final.pkl')
    torch.save(model, location_final_model)
