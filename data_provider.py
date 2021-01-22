from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y) means center of holes
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class MNIST_rot(Dataset):
    def __init__(self, root='../data/mnist_rot/mnist_all_rotation_normalized_float_train_valid.amat',transform=[]):
        # x = np.loadtxt(root)
        self.transform = transform
        x = torch.load(root)
        self.x_data = torch.from_numpy(x[:,0:-1].reshape(-1,1,28,28)).type(torch.FloatTensor)
        # self.x_data = transform(x[:,0:-1].reshape(-1,1,28,28))
        self.y_data = torch.from_numpy(x[:,-1].astype(np.int16)).type(torch.LongTensor)
        self.len = x.shape[0]

    def __getitem__(self,index):
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len


def data_provider(dataset, root, batch_size, n_threads=4, download=False):

    if dataset == 'mnist':
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transformtest = transforms.Compose([
            # transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=download,
                                              transform=transform)

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=download,
                                             transform=transformtest)

    elif dataset == 'mnist_rot':
        # r = np.random.uniform(0.15,1.)
        # print(r)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            # # transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1295,), (0.2963,)),
            Cutout(n_holes=2,length=6),
        ])
        transformtest = transforms.Compose([
            transforms.Normalize((0.1295,), (0.2963,))
        ])
        # trainset = MNIST_rot(root='../data/mnist_rot/mnist_all_rotation_normalized_float_train_valid.amat',transform=transform)
        # testset = MNIST_rot(root='../data/mnist_rot/mnist_all_rotation_normalized_float_test.amat',transform=transformtest)
        trainset = MNIST_rot(root='../data/mnist_rot/train.pt',transform=transform)
        testset = MNIST_rot(root='../data/mnist_rot/test.pt',transform=transformtest)

    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=download,
            transform=transform,
            target_transform=lambda x: int(x[0]) - 1)

        testset = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=download,
            transform=transform,
            target_transform=lambda x: int(x[0]) - 1)

    elif dataset == 'cifar10':
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            # Cutout(n_holes=1, length=16)
        ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(norm_mean, norm_std)])

        trainset = torchvision.datasets.CIFAR10(root=root,
                                                train=True,
                                                download=download,
                                                transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root=root,
                                               train=False,
                                               download=download,
                                               transform=test_transform)

    elif dataset == 'cifar100':
        norm_mean = [0.50705882, 0.48666667, 0.44078431]
        norm_std = [0.26745098, 0.25568627, 0.27607843]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            # Cutout(n_holes=1, length=8)
        ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(norm_mean, norm_std)])

        trainset = torchvision.datasets.CIFAR100(root=root,
                                                 train=True,
                                                 download=download,
                                                 transform=train_transform)

        testset = torchvision.datasets.CIFAR100(root=root,
                                                train=False,
                                                download=download,
                                                transform=test_transform)

    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        trainset = torchvision.datasets.ImageFolder(
            os.path.join(root, "train_100/train_100"),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        testset = torchvision.datasets.ImageFolder(
            os.path.join(root, "val_100"),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=n_threads)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=n_threads)

    return trainloader, testloader


def test():
    traindl, testdl = data_provider(
        'mnist_rot',
        '../data/mnist_rot/trainandvalid',
        100)
    # traindl, testdl = data_provider(
    #     'mnist',
    #     '../data/mnist',
    #     100)
    print(len(traindl))
    a = 0.
    for idx, batch in enumerate(traindl):
        img = batch[0]
        a = a + img.mean().numpy()
    a = a / len(traindl)
    print(a)

    b = 0.
    # m = torch.Tensor(a)
    for idx, batch in enumerate(traindl):
        img = batch[0]
        b = b + img.pow(2).mean().numpy()
    b = b / len(traindl)
    print(np.sqrt(b))


if __name__ == "__main__":
    test()
