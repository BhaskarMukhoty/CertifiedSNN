import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10 , CIFAR100,SVHN
import torch
import os
import json
import time
import zipfile
import io
import torchvision
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import json
import time
import argparse
import zipfile
import io
from PIL import Image
path = "/home/gubin/bhaskar/datasets/"


def cifar10():
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root=path,
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=path,
                              train=False, download=True, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return train_dataset, val_dataset, norm
def svhn():
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor(),
        ])
        train_dataset = SVHN(root=path,
                            split = 'train', download=True, transform=transform_train)
        val_dataset = SVHN(root=path,
                            split = 'test', download=True, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return train_dataset, val_dataset, norm
def cifar100():
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR100(root=path,
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=path,
                              train=False, download=True, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return train_dataset, val_dataset, norm



def imagenet100(traindir, cache_dataset, distributed):
    # Data loading code
    
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    # cache_path = _get_cache_path(traindir)
    if cache_dataset :
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(traindir, 'train.X'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                #utils.ImageNetPolicy(),
                transforms.ToTensor(),
                # normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    # cache_path = _get_cache_path(valdir)
    if cache_dataset :
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(traindir, 'val.X'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return dataset, dataset_test, norm

