import os
import collections.abc
import copy
import json

import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import Dataset as _TorchDataset

from local_data.constants import PATH_DATASETS

template = ['a photo of a {}, a type of aircraft.']


class Dataset(_TorchDataset):

    def __init__(self, transform=None):

        # Transform to apply
        self.transform = transform

        # Set image directories
        root = os.path.abspath(PATH_DATASETS)
        self.dataset_dir = os.path.join(root, "fgvc_aircraft")
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # Produce classnames
        self.classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(self.classnames)}

        # Read images paths
        self.data = self.read_data(cname2lab, 'images_variant_test.txt')

        # Prepare class templates
        self.templates = template
        print("FGVC Num of samples: " + str(len(self.data)), end="\n")

    def __len__(self):
        return len(self.data)

    def _transform(self, index):
        # Retrieve sample
        data_index = copy.deepcopy(self.data[index])
        # Read image
        img = Image.open(data_index["impath"])
        # Transform image
        img = self.transform(img)
        # Include image
        data_index["img"] = img
        return data_index

    def __getitem__(self, index):
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)

        return self._transform(index)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                items.append({"impath": impath, "label": int(label), "classname": classname})

        return items