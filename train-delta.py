from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import wandb

import os
import time
from argparse import Namespace
from models.ff import Net, Net2
from models.cnn import CNN, CNN2
from models.resnet import ResNet18
from utils import split_train_valid, minmaxscale, accuracy, CosAnnealingScheduler


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

def train(classification_net, loaders, optimizers, epochs=2000, steps_with_simp=1000, lr=0.001, minibatch_size=32,
          conf_thres=None,
          iterations_simp=10, step_simp=0.01, scaling='linear'):
    """Train a target classifier exploiting a simplification model."""

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_test_acc_seen = 0.0
    best_test_acc = 0.0

    classification_net.train()

    # classifier properties
    loss_fcn = torch.nn.CrossEntropyLoss()
    device = next(classification_net.parameters()).device

    steps = 0

    # loop on epochs
    for e in range(0, epochs):
        it_simp = 0
        mse_simp = 0.
        task_penalty_w_simp = 0.

        train_loss = 0.
        train_acc = 0.
        simple_ratio_avg = 0.
        t = 0
        nb = 0

        start = time.time()
        for X_minibatch, y_minibatch in tqdm(loaders['train']):
            B = X_minibatch.size(0)
            X_minibatch, y_minibatch = X_minibatch.to(device), y_minibatch.to(device)
            optimizers['clf'].zero_grad()

            # adapting the parameters of the simplifier (they change during the learning stage)
            scale = max(1. - float(steps) / float(steps_with_simp), 0.0)
            if steps == steps_with_simp and 'clf_lr_factor' in optimizers:
                print(".. entering refinement stage, changing lr if needed")
                optimizers['scheduler'].enter_refinement(lr=optimizers['clf_lr_initial'] * optimizers['clf_lr_factor'])
            if scaling == 'quadratic':
                scale = np.power(scale, 2)


            # computing the simplified input (this will involve multiple iterations over the simplification module)
            simp_dic, _, X_minibatch_s, y_minibatch_s = simplify(classification_net, X_minibatch, y_minibatch, loss_fcn,
                                                                 iterations_simp, scale=scale, conf_thres=conf_thres,
                                                                 step_simp=step_simp)

            it_simp += simp_dic['it']
            mse_simp += simp_dic['mse_simp']
            simple_ratio_avg += simp_dic['simple_ratio']

            task_penalty_w_simp += simp_dic['task_penalty_w']

            classification_net.train()
            # computing the output of the classifier (using the simplified input data)
            outputs = classification_net(X_minibatch_s)
            loss_value_on_minibatch = loss_fcn(outputs, y_minibatch_s)

            # measuring some stuff
            with torch.no_grad():
                acc_train_on_minibatch = accuracy(outputs, y_minibatch_s)
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

        mse_simp /= len(loaders['train'])
        task_penalty_w_simp /= len(loaders['train'])
        it_simp_avg = 1.0 * it_simp / len(loaders['train'])
        simple_ratio_avg /= len(loaders['train'])
        train_loss /= len(loaders['train'].sampler)
        train_acc /= len(loaders['train'].sampler)
        simp_params = {'iterations_simp': iterations_simp, 'scale': scale, 'step_simp': step_simp,
                       'conf_thres': conf_thres}

        print("Elapsed time: ", time.time() - start)
        print("epoch: {}, loss: {:.4f}, acc: {:.2f}".format(e + 1, train_loss, train_acc))
        eval_dic = test(classification_net, loss_fcn, valid_loader=loaders['valid'], test_loader=loaders['test'],
                        epoch=e, simp_params=simp_params)

        if eval_dic['test_acc'] > best_test_acc_seen:
            best_test_acc_seen = eval_dic['test_acc']
        if eval_dic['val_acc'] > best_val_acc and scale == 0.0:
            best_epoch = e
            best_train_acc = train_acc
            best_val_acc = eval_dic['val_acc']
            best_test_acc = eval_dic['test_acc']
        log_dic = {"epoch": e + 1, "train_loss": train_loss, "train_acc": train_acc,
                   "simple_ratio_avg": simple_ratio_avg, "mse_simp": mse_simp, "scale": scale,
                   "it_simp_avg": it_simp_avg, "task_penalty_w": task_penalty_w_simp}
        if 'clf_scheduler' in optimizers:
            log_dic['lr_schedule'] = lr
        log_dic.update(eval_dic)

    result_dic = {'best_epoch': best_epoch, 'best_train_acc': best_train_acc, "best_val_acc": best_val_acc,
                  "best_test_acc": best_test_acc, "best_test_acc_seen": best_test_acc_seen}
    return result_dic


def test(classification_net, loss_fcn, valid_loader, test_loader, epoch, simp_params):
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

    if args.dataset not in nonimage_datasets:
        simplified_images = []
        delta_images = []
        original_images = []
        print('Logging corrected images..')
        z = 0

        X_minibatch, y_minibatch = next(iter(test_loader))

        X_minibatch, y_minibatch = X_minibatch[:8].to(device), y_minibatch[:8].to(device)
        shape_orig = X_minibatch.shape
        simp_dic, X_orig, X_simpl, _ = simplify(classification_net, X_minibatch, y_minibatch, loss_fcn,
                                                iterations=simp_params['iterations_simp'], scale=simp_params['scale'],
                                                step_simp=simp_params['step_simp'],
                                                conf_thres=simp_params['conf_thres'])

        X_simpl = X_simpl.view(shape_orig)
        X_orig = X_orig.view(shape_orig)
        with torch.no_grad():
            X_delta = X_simpl - X_orig

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

def simplify(classifier, X, y, loss_fcn, iterations, scale,
             step_simp, conf_thres=None):
    """Simplify the given data X to improve the predictions of a target classifier.
    """

    # checking arguments
    if iterations is not None and iterations < 0.:
        raise ValueError("Invalid number of iterations: " + str(iterations))
    if step_simp is not None and step_simp < 0.:
        raise ValueError("Invalid step size: " + str(step_simp))

    classifier.eval()
    dic = {'it': 0, 'task_penalty_w': 0.0}

    B = X.size(0)

    output_natural = []

    iter_clean_data = X.cuda().detach()
    iter_target = y.cuda().detach()
    output_target = []

    # learnable parameters (perturbation offsets or simplifier network)
    iter_simpl = X

    it = 0

    # optimization iterations
    for it in range(0, int(iterations * scale)):
        # print("it",it, ".. still",len(iter_target))
        iter_simpl.requires_grad_()
        # forward (perturbed data)
        classifier.zero_grad()
        outputs = classifier(iter_simpl)

        task_penalty = loss_fcn(outputs, iter_target)
        preds = outputs.max(1, keepdim=True)[1]
        probs = F.softmax(outputs, dim=1)
        output_index = []
        iter_index = []

        # Calculate the indexes of data that still need to be iterated
        for idx in range(len(preds)):
            if preds[idx] == iter_target[idx] and (conf_thres is None or probs[idx, preds[idx]] > conf_thres):
                output_index.append(idx)
            else:
                iter_index.append(idx)

        if len(output_index) != 0:
            if len(output_target) == 0:
                output_simpl = iter_simpl[output_index].cuda()
                output_natural = iter_clean_data[output_index].cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
            else:
                output_simpl = torch.cat((output_simpl, iter_simpl[output_index].cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
        dic['task_penalty_w'] += task_penalty.item()

        loss_value = task_penalty

        # backward
        loss_value.backward()
        grad = iter_simpl.grad

        iter_simpl = iter_simpl[iter_index]
        iter_clean_data = iter_clean_data[iter_index]
        iter_target = iter_target[iter_index]

        if len(iter_index) != 0:
            grad = grad[iter_index]
            iter_simpl = iter_simpl.detach() -step_simp * grad
        else:
            break

    dic['simple_ratio'] = 1.0 * len(output_target) / B

    if len(output_target) == 0:
        output_simpl = iter_simpl.cuda()
        output_target = iter_target.reshape(-1).cuda()
        output_natural = iter_clean_data.cuda()
    else:
        output_simpl = torch.cat((output_simpl, iter_simpl.cuda()), dim=0)
        output_target = torch.cat((output_target, iter_target.reshape(-1).cuda()), dim=0)
        output_natural = torch.cat((output_natural, iter_clean_data.cuda()), dim=0)
    output_simpl = output_simpl.detach()

    dic['it'] = it
    dic['task_penalty_w'] /= it + 1
    dic['mse_simp'] = torch.nn.MSELoss()(output_simpl, output_natural)
    return dic, output_natural.detach(), output_simpl.detach(), output_target.detach()


def main():
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

    parser.add_argument('--ratio_simp', type=float, default=0.25, metavar='r',
                        help='ratio of simplified steps with respect to total learning steps [0,1]')
    parser.add_argument('--iterations_simp', type=int, default=0, metavar='N',
                        help='maximum number of iterations of simplifier')
    parser.add_argument('--step_simp', type=float, default=0.0, metavar='LR',
                        help='step of simplifier function (default: 1e-4)')
    parser.add_argument('--conf_thres', type=float, default=None, metavar='C',
                        help='confidence threshold for input simplification (default None)')
    parser.add_argument('--scaling', type=str, default='linear',
                        choices=['linear', 'quadratic'], metavar='X',
                        help='time scaling strategy (default: linear)')
    parser.add_argument('--arch', type=str, default='ff',
                        choices=['ff', 'ff2', 'cnn', 'cnn2', 'resnet'], metavar='A',
                        help='classifier architecture (default: ff)')
    parser.add_argument('--data_augmentation', type=str, default=None,
                        choices=['no', 'yes'], metavar='DA',
                        help='data augmentation (only for images)')
    parser.add_argument('--noisy_labels', type=float, metavar='NL', help='fraction of labels shuffled', default='0.0')
    parser.add_argument('--dataset', type=str, default='mnist_back_image',
                        choices=larochelle_datasets + mnist_dropin_replacement_datasets + cifar_datasets + nonimage_datasets,
                        metavar='D', help='dataset for the learning problem')
    parser.add_argument('--load_classifier', type=str, default=None,
                        metavar='P',
                        help='path of classifier weights')
    parser.add_argument('--cos_scheduler', type=str, default=None,
                        choices=['delayed', 'monotonic', 'restart'], metavar='SCH',
                        help='kind of cosine annealing scheduler (if any)')
    parser.add_argument('--run', type=int, default=2,
                        help='run identifier (default: 2)')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                        help='input batch size for training (default: 32/128)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4')
    parser.add_argument('--lr_factor', type=float, default=None, metavar='LRF',
                        help='learning rate factor (default: 1')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                       help='weight decay (default: 1e-4')
    parser.add_argument('--init', type=str, default='kaiming',
                        choices=['kaiming', 'xavier'], metavar='OPT',
                        help='optimizer (default: adam)')
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'adadelta', 'rmsprop'], metavar='OPT',
                        help='optimizer (default: adam)')
    parser.add_argument('--baseline', action="store_true",
                        help='baseline mode')
    args_cmd = parser.parse_args()

    global args
    args = Namespace(**vars(args_cmd))
    args.output_simplifier = "linear"

    if args.batch_size is None:
        args.batch_size = 128 if args.arch == 'resnet' else 32

    print("Total params", args)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.run)  # so that different runs get different weights
    np.random.seed(args.seed)  # so that train/valid split is consistent

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    numw = int(os.environ.get('NUMW')) if 'NUMW' in os.environ else 0

    if use_cuda:
        print("I will use {:d} workers..".format(numw))
        cuda_kwargs = {'num_workers': numw,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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

    dataset_dic = {'n_classes': datasets_classes[args.dataset], 'input_size':datasets_dimensionality[args.dataset]}

    if args.lr_factor is None:
        args.lr_factor = 0.1 if args.arch == 'resnet' and args.ratio_simp != 0.0 else 1.0

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

    if args.load_classifier is not None:
        model.load_state_dict(torch.load(args.load_classifier))

    print(model)

    optimizers = {}
    if args.optim == 'adadelta':
        optimizers['clf'] = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizers['clf'] = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optimizers['clf'] = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgdc':
        optimizers['clf'] =  optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizers['clf_scheduler'] = CosAnnealingScheduler(optimizers['clf'], ratio_simp=args.ratio_simp,
                                                        epochs=args.epochs, type=args.cos_scheduler)
        optimizers['clf_lr_factor'] = args.lr_factor
        optimizers['clf_lr_initial'] = args.lr

    result_dic = train(model,
                       loaders,
                       optimizers,
                       epochs=args.epochs, lr=args.lr,
                       minibatch_size=args.batch_size,
                       steps_with_simp=args.steps_with_simp,
                       # a "step" consists in a single update of the parameters of the classifier
                       conf_thres=args.conf_thres,
                       iterations_simp=args.iterations_simp, step_simp=args.step_simp,
                       scaling=args.scaling)
    print(result_dic)


if __name__ == '__main__':
    main()


