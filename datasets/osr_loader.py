# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .tools import *

DATA_PATH = '/root/mjxx/datasets'
# DATA_PATH = '_'
TINYIMAGENET_PATH = DATA_PATH + '/tiny_imagenet/'


class CIFAR10_Filter(CIFAR10):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)
        
        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)        
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_Filter(CIFAR100):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR100_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))
        print('Selected Labels: ', known)

        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        

class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """

    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)


class SVHN_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train',
                               download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """

    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot=TINYIMAGENET_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)        
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))