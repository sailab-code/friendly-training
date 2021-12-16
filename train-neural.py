from __future__ import print_function

import argparse
import json
import sys
import time

import numpy as np
import os
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from models.cnn import CNN, CNN2
from models.ff import Net, Net2
from models.resnet import ResNet18
from models.simplifier import SimplifierFF, SimplifierUNet
from utils import split_train_valid, minmaxscale, set_lr_of_optimizer, accuracy, CosAnnealingScheduler


def add_noise_cifar_labels(v, frac, permanent=False):
    index_path = 'data/cifar_index.pt'
    n_shuffle = int(len(v) * frac)
    t = v[:n_shuffle]
    if permanent and os.path.exists(index_path):
        print('Loading saved permutation..')
        idx = torch.load(index_path)
    else:
        idx = torch.randperm(t.nelement())
        if permanent:
            print('Saving label permutation..')
            torch.save(idx, index_path)
    t = t.view(-1)[idx].view(t.size())
    v[:n_shuffle] = t
    return v


def train(networks, loaders, optimizers, epochs, steps_with_simp, acc_thres,
          iterations_simp, beta_simp, scaling):
    """Train a target classifier exploiting a simplification model."""

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_test_acc_seen = 0.0
    best_test_acc = 0.0

    classification_net = networks['clf']
    simplifier = networks['simp']
    classification_net.train()

    # classifier properties
    loss_fcn = torch.nn.CrossEntropyLoss()
    device = next(classification_net.parameters()).device

    steps = 0

    # loop on epochs
    for e in range(0, epochs):

        train_loss = 0.
        train_acc = 0.
        t = 0
        nb = 0

        start = time.time()
        for X_minibatch, y_minibatch in tqdm(loaders['train']):
            B = X_minibatch.size(0)
            X_minibatch, y_minibatch = X_minibatch.to(device), y_minibatch.to(device)
            optimizers['clf'].zero_grad()
            optimizers['simp'].zero_grad()

            # adapting the parameters of the simplifier (they change during the learning stage)
            scale = max(1. - float(steps) / float(steps_with_simp), 0.0) if steps_with_simp > 0 else 0.0
            if steps == steps_with_simp and 'clf_lr_factor' in optimizers and optimizers['clf_lr_factor'] != 1.0:
                print(".. entering refinement stage, changing lr")
                optimizers['clf_scheduler'].enter_refinement(lr = optimizers['clf_lr_initial'] * optimizers['clf_lr_factor'])
            if scaling == 'quadratic':
                scale = np.power(scale, 2)

            # computing the simplified input (this will involve a single forward over the simplification module)
            if scale > 0.0:
                X_minibatch_s = simplifier(X_minibatch, y_minibatch)
            else:
                X_minibatch_s = X_minibatch

            optimizers['clf'].zero_grad()
            optimizers['simp'].zero_grad()
            classification_net.train()

            # computing the output of the classifier (using the simplified input data)
            outputs = classification_net(X_minibatch_s)
            loss_value_on_minibatch = loss_fcn(outputs, y_minibatch)

            # measuring some stuff
            with torch.no_grad():
                acc_train_on_minibatch = accuracy(outputs, y_minibatch)
                train_loss += loss_value_on_minibatch * B  # needed to estimate the loss on the training set
                train_acc += acc_train_on_minibatch * B  # needed to estimate the accuracy on the training set

            # backward
            loss_value_on_minibatch.backward()
            optimizers['clf'].step()

            nb += 1
            steps += 1
        if 'clf_scheduler' in optimizers:
            lr = optimizers['clf_scheduler'].get_lr()
            print("lr: ", lr)
            optimizers['clf_scheduler'].make_step(scale)

        simp_params = {'iterations_simp': iterations_simp, 'scale': scale, 'beta_simp': beta_simp,
                       'acc_thres': acc_thres}
        train_loss /= len(loaders['train'].sampler)
        train_acc /= len(loaders['train'].sampler)

        if scale > 0.0:
            for it in range(iterations_simp):
                for X_minibatch, y_minibatch in tqdm(loaders['train']):
                    optimizers['clf'].zero_grad()
                    optimizers['simp'].zero_grad()
                    X_minibatch, y_minibatch = X_minibatch.to(device), y_minibatch.to(device)
                    X_s = simplifier(X_minibatch, y_minibatch)
                    outputs = classification_net(X_s)

                    task_loss = loss_fcn(outputs, y_minibatch)
                    total_loss = task_loss

                    if beta_simp is not None and beta_simp > 0:
                        norm_penalty = (1.0 - scale) * torch.nn.MSELoss()(X_s, X_minibatch)
                        norm_loss = beta_simp * norm_penalty
                        total_loss += norm_loss
                    else:
                        norm_loss = torch.tensor(0)
                    total_loss += norm_loss

                    # backward
                    total_loss.backward()
                    optimizers['simp'].step()
                eval_dic = test(networks, loss_fcn, valid_loader=loaders['valid'], test_loader=loaders['test'], epoch=e,
                                simp_params=simp_params)
                if acc_thres is not None and eval_dic['val_acc'] > acc_thres: break
        else:
            eval_dic = test(networks, loss_fcn, valid_loader=loaders['valid'], test_loader=loaders['test'], epoch=e,
                            simp_params=simp_params)

        print("Elapsed time: ", "{:.2f}".format(time.time() - start))
        print("epoch: {}, loss: {:.4f}, acc: {:.2f}".format(e + 1, train_loss, train_acc))

        if eval_dic['test_acc'] > best_test_acc_seen:
            best_test_acc_seen = eval_dic['test_acc']
        if eval_dic['val_acc'] > best_val_acc and scale == 0.0:
            best_epoch = e
            best_train_acc = train_acc
            best_val_acc = eval_dic['val_acc']
            best_test_acc = eval_dic['test_acc']
        log_dic = {"epoch": e + 1, "train_loss": train_loss,
                   "train_acc": train_acc, "scale": scale}
        if 'clf_scheduler' in optimizers:
            log_dic['lr_schedule'] = lr
        log_dic.update(eval_dic)

    result_dic = {'best_epoch': best_epoch, 'best_train_acc': best_train_acc, "best_val_acc": best_val_acc,
                  "best_test_acc": best_test_acc, "best_test_acc_seen": best_test_acc_seen}

    return result_dic


def test(networks, loss_fcn, valid_loader, test_loader, epoch, simp_params):
    classification_net = networks['clf']
    simplifier = networks['simp']
    classification_net.eval()
    device = next(classification_net.parameters()).device
    test_loss = 0
    test_correct_detached = 0
    valid_loss = 0
    valid_correct_detached = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_detached = classification_net(data)

            test_loss += loss_fcn(output_detached, target).item() * test_loader.batch_size  # sum up batch loss
            test_pred_detached = output_detached.argmax(dim=1, keepdim=True)
            test_correct_detached += test_pred_detached.eq(target.view_as(test_pred_detached)).sum().item()

        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output_detached = classification_net(data)

            valid_loss += loss_fcn(output_detached, target).item() * valid_loader.batch_size  # sum up batch loss
            valid_pred_detached = output_detached.argmax(dim=1, keepdim=True)
            valid_correct_detached += valid_pred_detached.eq(target.view_as(valid_pred_detached)).sum().item()

    test_loss /= len(test_loader.dataset)
    valid_loss /= len(valid_loader.sampler)

    val_acc_detached = valid_correct_detached / len(valid_loader.sampler)
    test_acc_detached = test_correct_detached / len(test_loader.dataset)
    print()
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        valid_loss, valid_correct_detached, len(valid_loader.sampler),
        100. * val_acc_detached))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, test_correct_detached, len(test_loader.dataset),
        100. * test_acc_detached))

    dic = {'val_loss': valid_loss, 'val_acc': val_acc_detached, 'test_acc': test_acc_detached, 'test_loss': test_loss}

    if len(data.shape) == 2: return dic

    simplified_images = []
    delta_images = []
    original_images = []

    z = 0

    X_orig, y_orig = next(iter(test_loader))

    X_orig, y_orig = X_orig[:8].to(device), y_orig[:8].to(device)
    shape_orig = X_orig.shape

    with torch.no_grad():
        X_simpl = simplifier(X_orig, y_orig)
        X_delta = X_simpl - X_orig

    X_simpl = X_simpl.view(shape_orig)
    X_orig = X_orig.view(shape_orig)
    X_delta = X_delta.view(shape_orig)

    X_orig = X_orig.cpu().numpy()
    X_simpl = X_simpl.cpu().numpy()
    X_delta = X_delta.cpu().numpy()

    for x in X_orig:
        if x.shape[0] == 3: x = np.transpose(x)
        original_images.append(wandb.Image(minmaxscale(x)))
    for x in X_delta:
        if x.shape[0] == 3: x = np.transpose(x)
        delta_images.append(wandb.Image(minmaxscale(x)))
    for x in X_simpl:
        if x.shape[0] == 3: x = np.transpose(x)
        simplified_images.append(wandb.Image(minmaxscale(x)))

    dic.update({'test_original_images': original_images, 'test_simplified_images': simplified_images,
                'test_delta_images': delta_images})
    return dic


ups = 0
mnist_mean = 0.1307
mnist_std = 0.3081


def main():
    datasets_splits = {
        'mnist_back_image': {'train': 10000, 'valid': 2000},
        'mnist_rot_back_image': {'train': 10000, 'valid': 2000},
        'mnist_rot': {'train': 10000, 'valid': 2000},
        'rectangles_image': {'train': 10000, 'valid': 2000},
        'rectangles': {'train': 1000, 'valid': 200},
        'convex': {'train': 6000, 'valid': 2000}
    }

    larochelle_datasets = ['mnist_back_image', 'mnist_rot_back_image', 'mnist_rot', 'rectangles', 'rectangles_image',
                           'convex']
    mnist_dropin_replacement_datasets = ['fashion', 'kmnist']
    cifar_datasets = ['cifar10', 'cifar10-n10']
    nonimage_datasets = ['imdb50k', 'wine', 'winedr']

    datasets_dimensionality = {}
    datasets_classes = {}
    for d in larochelle_datasets + mnist_dropin_replacement_datasets:
        datasets_dimensionality[d] = [28, 28, 1]
        datasets_classes[d] = 10
    for d in cifar_datasets:
        datasets_dimensionality[d] = [32, 32, 3]
        datasets_classes[d] = 10

    datasets_dimensionality['imdb50k'] = [20002]
    datasets_classes['imdb50k'] = 2

    datasets_dimensionality['wine'] = [20002]
    datasets_classes['wine'] = 2

    datasets_dimensionality['winedr'] = [768]
    datasets_classes['winedr'] = 2

    datasets_features = {k: np.prod(v) for (k, v) in datasets_dimensionality.items()}

    # Training settings
    parser = argparse.ArgumentParser(description='Experiments friendly-train')
    parser.add_argument('--lr_simp', type=float, default=None, metavar='LR',
                        help='learning rate of simplifier network (default: 1e-4')
    parser.add_argument('--lr_clf', type=float, default=None, metavar='LR',
                        help='learning rate of classifier network (default: 1e-4')
    parser.add_argument('--lr_factor_clf', type=float, default=None, metavar='LR',
                        help='learning rate factor of classifier network (default: 1')
    parser.add_argument('--ratio_simp', type=float, default=0.25, metavar='r',
                        help='ratio of simplified steps with respect to total learning steps [0,1]')
    parser.add_argument('--iterations_simp', type=int, default=1, metavar='N',
                        help='maximum number of iterations of simplifier')
    parser.add_argument('--beta_simp', type=float, default=10.0,
                        metavar='BETA',
                        help='coefficient of delta norm penalty (default: 10.0)')
    parser.add_argument('--acc_thres', type=float, default=None, metavar='C',
                        help='accuracy threshold for early stopping simplifier epochs (default None)')
    parser.add_argument('--scaling', type=str, default='quadratic',
                        choices=['linear', 'quadratic'], metavar='X',
                        help='time scaling strategy (default: linear)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky-relu', 'tanh'], metavar='A',
                        help='activation function (default: relu)')
    parser.add_argument('--simplifier', type=str, default='ff',
                        choices=['ff', 'unet'], metavar='S',
                        help='simplifier architecture (default: unet)')
    parser.add_argument('--target_conditioning', type=str, default='no',
                        choices=['no', 'yes'], metavar='X',
                        help='simplifier target conditioning (default: no)')
    parser.add_argument('--data_augmentation', type=str, default=None,
                        choices=['no', 'yes'], metavar='DA',
                        help='data augmentation (only for images)')
    parser.add_argument('--cos_scheduler', type=str, default=None,
                        choices=['delayed', 'monotonic', 'restart'], metavar='SCH',
                        help='cosine annealing scheduler (only used with optim = sgdc+adam')
    parser.add_argument('--noisy_labels', type=float, metavar='NL', help='fraction of labels shuffled', default='0.0')
    parser.add_argument('--sigmoid_postprocessing', type=str, default='no',
                        choices=['no', 'yes'], metavar='S',
                        help='Sigmoid postprocessing on the output network')
    parser.add_argument('--arch', type=str, default='ff',
                        choices=['ff', 'ff2', 'cnn', 'cnn2', 'resnet'], metavar='A',
                        help='classifier architecture (default: ff)')
    parser.add_argument('--optim', type=str, default=None,
                        choices=['adam', 'adadelta', 'rmsprop', 'sgdc+adam'], metavar='OPT',
                        help='optimizer (default: adam)')
    parser.add_argument('--weight_decay_clf', type=float, default=None, metavar='WD',
                        help='weight decay (default: 1e-8)')
    parser.add_argument('--weight_decay_simp', type=float, default=1e-8, metavar='WD',
                        help='weight decay (default: 1e-8)')

    parser.add_argument('--n_deep', type=int, default=None, metavar='N',
                        help='depth of unet simplifier')
    parser.add_argument('--n_filters_base', type=int, default=None, metavar='N',
                        help='base number of filters of unet simplifier')
    parser.add_argument('--filters', nargs="+", default=[])
    parser.add_argument('--kernels', nargs="+", default=[])
    parser.add_argument('--strides', nargs="+", default=[])
    parser.add_argument('--padding', nargs="+", default=[])
    parser.add_argument('--hidden', nargs="+", default=[])
    parser.add_argument('--dataset', type=str, default='mnist_back_image',
                        choices=larochelle_datasets + mnist_dropin_replacement_datasets + cifar_datasets + nonimage_datasets,
                        metavar='D', help='dataset for the learning problem')

    parser.add_argument('--run', type=int, default=2,
                        help='run identifier (default: 2)')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=None, metavar='N',
                        help='input batch size for testing (default: 512)')
    args_cmd = parser.parse_args()

    if len(args_cmd.filters) == 1 and type(args_cmd.filters[0]) == str:
        args_cmd.filters = json.loads(args_cmd.filters[0])
    if len(args_cmd.kernels) == 1 and type(args_cmd.kernels[0]) == str:
        args_cmd.kernels = json.loads(args_cmd.kernels[0])
    if len(args_cmd.strides) == 1 and type(args_cmd.strides[0]) == str:
        args_cmd.strides = json.loads(args_cmd.strides[0])
    if len(args_cmd.padding) == 1 and type(args_cmd.padding[0]) == str:
        args_cmd.padding = json.loads(args_cmd.padding[0])
    args = args_cmd

    if args.dataset in nonimage_datasets and args.simplifier != 'ff':
        print('Incompatible data type and simplifer type')
        sys.exit(1)

    if args.cos_scheduler is None and args.optim == 'sgdc+adam':
        args.cos_scheduler = 'delayed'
    if args.optim is None:
        args.optim = 'sgdc+adam' if args.arch == 'resnet' else 'adam'
    if args.lr_simp is None: args.lr_simp = 1e-4
    if args.lr_clf is None:
        args.lr_clf = 0.1 if args.arch == 'resnet' else 1e-4
    if args.lr_factor_clf is None:
        args.lr_factor_clf = 0.1 if args.arch == 'resnet' and args.ratio_simp != 0.0 else 1.0
    if args.weight_decay_clf is None:
        args.weight_decay_clf = 5e-4 if args.arch == 'resnet' else 1e-8
    if args.batch_size is None:
        args.batch_size = 128 if args.arch == 'resnet' else 32
    if args.test_batch_size is None:
        args.test_batch_size = 100 if args.arch == 'resnet' else 512
    if args.data_augmentation is None:
        args.data_augmentation = "yes" if (args.arch == 'resnet' and args.dataset in cifar_datasets) else "no"
    if args.simplifier == 'ff':
        if len(args.hidden) == 0:
            args.hidden = [256, datasets_features[args.dataset]]
        del args.filters
        del args.kernels
        del args.strides
        del args.padding
    if args.simplifier == 'unet':
        if args.n_filters_base is None:
            args.n_filters_base = 64
            args.n_deep = 2
        del args.filters
        del args.kernels
        del args.strides
        del args.padding
        del args.hidden

    if args.ratio_simp == 0.0:
        args.iterations_simp = 0
        args.baseline = True
    else:
        args.baseline = False

    print("Total params", args)

    if args.data_augmentation == "yes" and (args.arch != "resnet" or args.dataset not in cifar_datasets):
        raise NotImplementedError("Data Augmentation implemented only for resnet / cifar10")

    use_cuda = torch.cuda.is_available()
    if use_cuda and 'NO_CUDA' in os.environ: use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.run)  # so that different runs get different weights
    np.random.seed(args.seed)  # so that train/valid split is consistent

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    numw = int(os.environ.get('NUMW')) if 'NUMW' in os.environ else 0

    if use_cuda:
        print("I will use {:d} workers..".format(numw))
        cuda_kwargs = {'num_workers': numw,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ### Creating datasets
    if args.dataset in larochelle_datasets:
        train_data = np.loadtxt("data/" + args_cmd.dataset + "/train.amat")
        test_data = np.loadtxt("data/" + args_cmd.dataset + "/test.amat")

        X_train = train_data[:, :-1] / 1.0
        y_train = train_data[:, -1:]

        X_test = test_data[:, :-1] / 1.0
        y_test = test_data[:, -1:]

        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train).squeeze().long()

        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test).squeeze().long()

        X_train = X_train.permute(0, 1, 3, 2)
        X_test = X_test.permute(0, 1, 3, 2)

        train_idx = np.asarray(range(datasets_splits[args_cmd.dataset]['train']))
        valid_idx = np.asarray(range(datasets_splits[args_cmd.dataset]['train'],
                                     datasets_splits[args_cmd.dataset]['train'] + datasets_splits[args_cmd.dataset][
                                         'valid']))

        dataset1 = torch.utils.data.TensorDataset(X_train, y_train)
        dataset2 = torch.utils.data.TensorDataset(X_test, y_test)
    elif args.dataset == 'imdb50k':
        X_train = torch.load('data/reviews/reviews_tfidf_train.pt')
        y_train = torch.load('data/reviews/reviews_labels_train.pt').long()
        X_test = torch.load('data/reviews/reviews_tfidf_test.pt')
        y_test = torch.load('data/reviews/reviews_labels_test.pt').long()
        valid_size = 5000
        n_train = X_train.size(0)
        n_valid_per_class = valid_size // 2
        n_train_per_class = n_train // 2
        train_idx = np.asarray(list(range(n_train_per_class - n_valid_per_class)) + list(
            range(n_train_per_class, n_train_per_class + n_train_per_class - n_valid_per_class)))
        valid_idx = np.asarray(list(range(n_train_per_class - n_valid_per_class, n_train_per_class)) + list(
            range(n_train_per_class + n_train_per_class - n_valid_per_class, n_train)))
        dataset1 = torch.utils.data.TensorDataset(X_train, y_train)
        dataset2 = torch.utils.data.TensorDataset(X_test, y_test)
    elif args.dataset == 'wine' or args.dataset == 'winedr':
        X_train = torch.load('data/wine/' + args.dataset + '_input_train.pt')
        y_train = torch.load('data/wine/' + args.dataset + '_labels_train.pt').long()
        X_test = torch.load('data/wine/' + args.dataset + '_input_test.pt')
        y_test = torch.load('data/wine/' + args.dataset + '_labels_test.pt').long()

        dataset1 = torch.utils.data.TensorDataset(X_train, y_train)
        dataset2 = torch.utils.data.TensorDataset(X_test, y_test)
        train_idx, valid_idx = split_train_valid(y_train, valid_size=30000)
    else:
        transform_train_list = [transforms.ToTensor()]
        transform_test_list = [transforms.ToTensor()]
        if args_cmd.dataset in cifar_datasets and args.arch == 'resnet':
            transform_train_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()]
            transform_test_list = [transforms.ToTensor()]
        transform_train = transforms.Compose(transform_train_list)
        transform_test = transforms.Compose(transform_test_list)
        if args_cmd.dataset == "fashion":
            dataset1 = datasets.FashionMNIST('data', train=True, download=True,
                                             transform=transform_train)
            dataset2 = datasets.FashionMNIST('data', train=False,
                                             transform=transform_test)
        elif args_cmd.dataset == "kmnist":
            dataset1 = datasets.KMNIST('data', train=True, download=True,
                                       transform=transform_train)
            dataset2 = datasets.KMNIST('data', train=False,
                                       transform=transform_test)

        elif args_cmd.dataset in cifar_datasets:
            if args.data_augmentation == "yes":  # we need two different datasets in order to apply augmentation transforms only on train
                dataset1_train = datasets.CIFAR10('data', train=True, download=True,
                                                  transform=transform_train)
                dataset1_valid = datasets.CIFAR10('data', train=True, download=True,
                                                  transform=transform_test)
                dataset2 = datasets.CIFAR10('data', train=False,
                                            transform=transform_test)
                dataset1_train.targets = torch.tensor(dataset1_train.targets)
                dataset1_valid.targets = torch.tensor(dataset1_valid.targets)
                dataset2.targets = torch.tensor(dataset2.targets)
                if args.dataset == "cifar10-n10":
                    dataset1_train.targets = add_noise_cifar_labels(dataset1_train.targets, frac=0.1, permanent=True)
                else:
                    if args.noisy_labels > 0.0: dataset1_train.targets = add_noise_cifar_labels(dataset1_train.targets,
                                                                                                frac=args.noisy_labels)

                dataset1 = dataset1_train
            else:
                dataset1 = datasets.CIFAR10('data', train=True, download=True,
                                            transform=transform_train)
                dataset2 = datasets.CIFAR10('data', train=False,
                                            transform=transform_test)
                dataset1.targets = torch.tensor(dataset1.targets)
                dataset2.targets = torch.tensor(dataset2.targets)
                if args.noisy_labels > 0.0: raise NotImplementedError("No shuffling implemented")

        train_idx, valid_idx = split_train_valid(dataset1.targets, valid_size=10000)

    ### Creating dataloaders
    if args.data_augmentation == "yes":
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset1_train, sampler=train_sampler, **train_kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset1_valid, sampler=valid_sampler, **train_kwargs)
    else:
        train_sampler = SubsetRandomSampler(train_idx)  # implements shuffling
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **train_kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset1, sampler=valid_sampler, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    args.steps_with_simp = int(args.ratio_simp * args.epochs * len(loaders['train']))
    print("steps_with_simp", args.steps_with_simp)

    dataset_dic = {'n_classes': datasets_classes[args.dataset], 'input_size':datasets_dimensionality[args.dataset]}
    if args.arch == 'ff':
        model = Net(**dataset_dic).to(device)
    elif args.arch == 'ff2':
        model = Net2(**dataset_dic).to(device)
    elif args.arch == 'cnn':
        model = CNN(**dataset_dic).to(device)
    elif args.arch == 'cnn2':
        model = CNN2(**dataset_dic).to(device)
    elif args.arch == 'resnet':
        model = ResNet18(**dataset_dic).to(device)

    if args.simplifier == 'unet':
        simplifier = SimplifierUNet(sigmoid_postprocessing=args.sigmoid_postprocessing == "yes",
                                    target_conditioning=args.target_conditioning == "yes",
                                    input_size=datasets_dimensionality[args.dataset],
                                    n_deep=args.n_deep, n_filters_base=args.n_filters_base).to(device)
    elif args.simplifier == 'ff':
        simplifier = SimplifierFF(datasets_features[args.dataset], hidden=args.hidden, activation=args.activation,
                                  target_conditioning=args.target_conditioning == "yes",
                                  sigmoid_postprocessing=args.sigmoid_postprocessing == "yes").to(device)

    print(simplifier)

    print(model)

    optimizers = {}
    if args.optim == 'adadelta':
        optimizers['clf'] = optim.Adadelta(model.parameters(), lr=args.lr_clf, weight_decay=args.weight_decay_clf)
        optimizers['simp'] = optim.Adadelta(simplifier.parameters(), lr=args.lr_simp,
                                            weight_decay=args.weight_decay_simp)
    elif args.optim == 'adam':
        optimizers['clf'] = optim.Adam(model.parameters(), lr=args.lr_clf, weight_decay=args.weight_decay_clf)
        optimizers['simp'] = optim.Adam(simplifier.parameters(), lr=args.lr_simp, weight_decay=args.weight_decay_simp)
    elif args.optim == 'rmsprop':
        optimizers['clf'] = optim.RMSprop(model.parameters(), lr=args.lr_clf, weight_decay=args.weight_decay_clf)
        optimizers['simp'] = optim.RMSprop(simplifier.parameters(), lr=args.lr_simp,
                                           weight_decay=args.weight_decay_simp)
    elif args.optim == 'sgdc+adam':
        optimizers['clf'] = optim.SGD(model.parameters(), lr=args.lr_clf, momentum=0.9,
                                      weight_decay=args.weight_decay_clf)
        optimizers['simp'] = optim.Adam(simplifier.parameters(), lr=args.lr_simp, weight_decay=args.weight_decay_simp)
        optimizers['clf_scheduler'] = CosAnnealingScheduler(optimizers['clf'], ratio_simp=args.ratio_simp,
                                                            epochs=args.epochs, type=args.cos_scheduler)
        optimizers['clf_lr_factor'] = args.lr_factor_clf
        optimizers['clf_lr_initial'] = args.lr_clf

    networks = {'clf': model, 'simp': simplifier}

    result_dic = train(networks,
                       loaders,
                       optimizers,
                       epochs=args.epochs,
                       steps_with_simp=args.steps_with_simp,
                       beta_simp=args.beta_simp,
                       acc_thres=args.acc_thres,
                       iterations_simp=args.iterations_simp,
                       scaling=args.scaling)

    print(result_dic)


if __name__ == '__main__':
    main()
