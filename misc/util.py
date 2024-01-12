import os
import torch
import random
import numpy as np
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def update_meter(dict_meter, dict_content, batch_size):
    for key, value in dict_meter.items():
        if isinstance(dict_content[key], torch.Tensor):
            value.update(dict_content[key].item(), batch_size)
        else:
            value.update(dict_content[key], batch_size)


def load_checkpoint(model, pth_file):
    print(' < Reading from Checkpoint > ')
    checkpoint = torch.load(pth_file)
    pretrained_dict = checkpoint['state_dict']
    
    model_dict = model.module.state_dict()
    model_dict.update(pretrained_dict)
    model.module.load_state_dict(model_dict)
    print(" < Loading from Checkpoint: '{}' (epoch {}) > ".format(pth_file, checkpoint['epoch']))

    return checkpoint 


def set_seeding(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.avg


splits_F1 = {
    'svhn': [
        [3, 7, 8, 2, 4, 6],
        [7, 1, 0, 9, 4, 6],
        [8, 1, 6, 7, 2, 4],
        [7, 3, 8, 4, 6, 1],
        [2, 8, 7, 3, 5, 1]
    ],
    'cifar10': [
        [3, 7, 8, 2, 4, 6],
        [7, 1, 0, 9, 4, 6],
        [8, 1, 6, 7, 2, 4],
        [7, 3, 8, 4, 6, 1],
        [2, 8, 7, 3, 5, 1]
    ],
    'cifar100': [
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9]
    ],
    'cifar100-10': [
        [27, 46, 98, 38, 72, 31, 36, 66, 3, 97],
        [98, 46, 14, 1, 7, 73, 3, 79, 93, 11],
        [79, 98, 67, 7, 77, 42, 36, 65, 26, 64],
        [46, 77, 29, 24, 65, 66, 79, 21, 1, 95],
        [21, 95, 64, 55, 50, 24, 93, 75, 27, 36]
    ],
    'cifar100-50': [
        [27, 46, 98, 38, 72, 31, 36, 66, 3, 97,
         75, 67, 42, 32, 14, 93, 6, 88, 11, 1, 44,
         35, 73, 19, 18, 78, 15, 4, 50, 65, 64,
         55, 30, 80, 26, 2, 7, 34, 79, 43, 74, 29,
         45, 91, 37, 99, 95, 63, 24, 21],
        [98, 46, 14, 1, 7, 73, 3, 79, 93, 11, 37,
         29, 2, 74, 91, 77, 55, 50, 18, 80, 63,
         67, 4, 45, 95, 30, 75, 97, 88, 36, 31,
         27, 65, 32, 43, 72, 6, 26, 15, 42, 19,
         34, 38, 66, 35, 21, 24, 99, 78, 44],
        [79, 98, 67, 7, 77, 42, 36, 65, 26, 64,
         66, 73, 75, 3, 32, 14, 35, 6, 24, 21, 55,
         34, 30, 43, 93, 38, 19, 99, 72, 97, 78,
         18, 31, 63, 29, 74, 91, 4, 27, 46, 2, 88,
         45, 15, 11, 1, 95, 50, 80, 44],
        [46, 77, 29, 24, 65, 66, 79, 21, 1, 95,
         36, 88, 27, 99, 67, 19, 75, 42, 2, 73,
         32, 98, 72, 97, 78, 11, 14, 74, 50, 37,
         26, 64, 44, 30, 31, 18, 38, 4, 35, 80,
         45, 63, 93, 34, 3, 43, 6, 55, 91, 15],
        [21, 95, 64, 55, 50, 24, 93, 75, 27, 36,
         73, 63, 19, 98, 46, 1, 15, 72, 42, 78,
         77, 29, 74, 30, 14, 38, 80, 45, 4, 26,
         31, 11, 97, 7, 66, 65, 99, 34, 6, 18, 44,
         3, 35, 88, 43, 91, 32, 67, 37, 79]
    ],
    'tiny_imagenet': [
        [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
        [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
        [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
        [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
        [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
    ]
}

splits_AUROC = {
    'svhn': [
        [0, 1, 2, 4, 5, 9],
        [0, 3, 5, 7, 8, 9],
        [0, 1, 5, 6, 7, 8],
        [3, 4, 5, 7, 8, 9],
        [0, 1, 2, 3, 7, 8]
    ],
    'cifar10': [
        [0, 1, 2, 4, 5, 9],
        [0, 3, 5, 7, 8, 9],
        [0, 1, 5, 6, 7, 8],
        [3, 4, 5, 7, 8, 9],
        [0, 1, 2, 3, 7, 8]
    ],
    'cifar100': [
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9],
        [0, 1, 8, 9]
    ],
    'cifar100-10': [
        [3, 15, 19, 21, 42, 46, 66, 72, 78, 98],
        [26, 31, 34, 44, 45, 63, 65, 77, 93, 98],
        [7, 11, 66, 75, 77, 93, 95, 97, 98, 99],
        [2, 11, 15, 24, 32, 34, 63, 88, 93, 95],
        [1, 11, 38, 42, 44, 45, 63, 64, 66, 67]
    ],
    'cifar100-50': [
        [1, 2, 7, 9, 10, 12, 15, 18, 21, 23, 26, 30, 32, 33, 34,
         36, 37, 39, 40, 42, 44, 45, 46, 47, 49, 50, 51, 52, 55,
         56, 59, 60, 61, 63, 65, 66, 70, 72, 73, 74, 76, 78, 80,
         83, 87, 91, 92, 96, 98, 99],
        [0, 2, 4, 5, 9, 12, 14, 17, 18, 20, 21, 23, 24, 25, 31,
         32, 33, 35, 39, 43, 45, 49, 50, 51, 52, 54, 55, 56, 60,
         64, 65, 66, 68, 70, 71, 73, 74, 77, 78, 79, 80, 82, 83,
         86, 91, 93, 94, 96, 97, 98],
        [0, 4, 10, 11, 12, 14, 15, 17, 18, 21, 23, 26, 27, 28, 29,
         31, 32, 33, 36, 39, 40, 42, 43, 46, 47, 51, 53, 56, 57,
         59, 60, 64, 66, 71, 73, 74, 75, 76, 78, 79, 80, 83, 87,
         91, 92, 93, 94, 95, 96, 99],
        [0, 2, 5, 6, 9, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23,
         26, 27, 28, 29, 31, 33, 35, 36, 37, 38, 39, 40, 43, 45,
         49, 52, 56, 59, 61, 62, 63, 64, 65, 71, 74, 75, 78, 80,
         82, 86, 87, 91, 93, 94, 96],
        [0, 1, 4, 6, 7, 12, 15, 16, 17, 19, 20, 21, 22, 23, 26,
         27, 28, 32, 39, 40, 42, 43, 44, 47, 49, 50, 52, 53, 54,
         55, 56, 59, 61, 62, 63, 65, 66, 67, 68, 73, 74, 77, 82,
         83, 86, 87, 93, 94, 97, 98]
    ],
    'tiny_imagenet': [
        [2, 3, 13, 30, 44, 45, 64, 66, 76, 101, 111, 121, 128, 130, 136, 158, 167, 170, 187, 193],
        [4, 11, 32, 42, 51, 53, 67, 84, 87, 104, 116, 140, 144, 145, 148, 149, 155, 168, 185, 193],
        [3, 9, 10, 20, 23, 28, 29, 45, 54, 74, 133, 143, 146, 147, 156, 159, 161, 170, 184, 195],
        [1, 15, 17, 31, 36, 44, 66, 69, 84, 89, 102, 137, 154, 160, 170, 177, 182, 185, 195, 197],
        [4, 14, 16, 33, 34, 39, 59, 69, 77, 92, 101, 103, 130, 133, 147, 161, 166, 168, 172, 173]
    ]
}