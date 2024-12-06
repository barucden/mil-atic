from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv


class RandomNoise(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X):
        X = X.clone()
        ix = t.rand_like(X) <= self.p
        noise = t.randn(t.sum(ix), device=X.device)
        X[ix] += noise
        return X


class RandomShift(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, X):
        shifts = t.randint(self.n, (2,))
        return t.roll(X, (shifts[0], shifts[1]), dims=(2, 3))


class MNISTBags:

    def __init__(self, root, bagsize_mean, bagsize_std, n_bags, train=True):
        transforms = [tv.transforms.ToTensor(),
                      tv.transforms.Normalize((0.1307,), (0.3081,))]

        data = tv.datasets.MNIST(root, train,
                                 download=True,
                                 transform=tv.transforms.Compose(transforms))
        TARGET = 9

        sizes = np.random.normal(bagsize_mean, bagsize_std, size=n_bags // 2)
        sizes = np.ceil(sizes).astype('int')

        instance_labels = np.array([y == TARGET for X, y in data])

        positive_instances = (instance_labels == True).nonzero()[0]
        negative_instances = (instance_labels == False).nonzero()[0]

        # Just sample negative instances
        negative_bags = [np.random.choice(negative_instances, s) for s in sizes]

        # Sample positive instances according to the prior probability and
        # complete bags with negative instances
        positive_prior = len(positive_instances) / len(negative_instances)
        positive_count = [np.maximum(1, np.random.binomial(s, positive_prior))
                          for s in sizes]
        positive_bags = [np.concatenate((np.random.choice(positive_instances, n),
                                         np.random.choice(negative_instances, s - n)))
                                        for s, n in zip(sizes, positive_count)]

        # Shuffle instances in the positive bags (otherwise the positive
        # instances would be always at the beginning of the bag)
        for b in positive_bags:
            np.random.shuffle(b)

        bags = negative_bags + positive_bags
        np.random.shuffle(bags)

        self.bags = [t.stack([data[i][0] for i in bag]) for bag in bags]
        self.instance_labels = [t.tensor([int(instance_labels[i]) for i in bag]) for bag in bags]
        self.labels = [labels.max() for labels in self.instance_labels]

        self.transform = None
        if train:
            self.transform = tv.transforms.Compose([
                tv.transforms.GaussianBlur(3),
                RandomNoise(0.05),
                RandomShift(4)
                ])

        # Make sure the dataset is really balanced
        assert abs(sum(self.labels) - n_bags // 2) <= 1, (sum(self.labels), n_bags)

    def __getitem__(self, i):
        bag = self.bags[i]
        label = self.labels[i]
        instance_label = self.instance_labels[i]
        if self.transform:
            bag = self.transform(self.bags[i])
        return bag, label, instance_label

    def __len__(self):
        return len(self.bags)

