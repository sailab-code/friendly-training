import numpy as np
import torch


def split_train_valid(targets, valid_size):
    num_train = len(targets)
    indices = np.asarray(range(num_train))
    classes, counts = torch.unique(targets, return_counts=True)
    frequencies = counts.numpy() / len(targets)

    labels_array = targets.numpy()
    train_idx = []
    valid_idx = []
    for c in classes:
        class_indices = indices[labels_array == c.item()]
        np.random.shuffle(class_indices)
        valid_samples_for_class = np.round(valid_size * frequencies[c]).astype(int)
        valid_idx.extend(class_indices[:valid_samples_for_class].tolist())
        train_idx.extend(class_indices[valid_samples_for_class:].tolist())
    print("Train examples: {:d}, Valid examples: {:d}".format(len(train_idx), len(valid_idx)))
    return train_idx, valid_idx


def minmaxscale(x):
    mi = x.min()
    ma = x.max()
    if ma == mi: return x - mi
    return (x - mi) / (ma - mi)


def set_lr_of_optimizer(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr

class CosAnnealingScheduler:
    def __init__(self, optimizer, epochs, ratio_simp, type):
        self.type = type
        self.optimizer = optimizer
        self.ratio_simp = ratio_simp
        self.epochs = epochs
        if self.type == 'delayed':
            T_max = int(round(1.0 - ratio_simp, 5) * epochs)
            print('initializing cosine scheduler (delayed) and T_max', T_max+1)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max+1)
        elif self.type=='monotonic':
            print('initializing cosine scheduler (monotonic) and T_max', epochs)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.type=='restart':
            T_0 = int(ratio_simp * epochs)
            print('initializing cosine scheduler with restart and T_max', T_0)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_0)


    def make_step(self, scale):
        if self.type == 'delayed':
            if scale == 0.0:
                self.sched.step()
        elif self.type == 'monotonic':
            self.sched.step()
        elif self.type == 'restart':
            self.sched.step()

    def enter_refinement(self, lr):
        if self.type == 'restart':
            T_0 = self.epochs - int(self.ratio_simp * self.epochs)
            print('initializing new cosine scheduler (restart) and T_max', T_0)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_0)
            set_lr_of_optimizer(self.optimizer, lr)
        elif self.type == 'delayed':
            set_lr_of_optimizer(self.optimizer, lr)
        elif self.type == 'monotonic':
            pass

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def accuracy_binary(outputs, y):
    """Compute the accuracy score (%)."""
    pred = outputs.round()
    return torch.mean(pred.eq(y.view_as(pred)).to(torch.float32)).item()


def accuracy(outputs, y):
    """Compute the accuracy score (%)."""
    pred = outputs.argmax(dim=1, keepdim=True)
    return torch.mean(pred.eq(y.view_as(pred)).to(torch.float32)).item()