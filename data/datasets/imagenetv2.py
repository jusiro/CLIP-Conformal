import os
import collections.abc
import copy

from PIL import Image
from torch.utils.data import Subset
from collections import OrderedDict
from torch.utils.data import Dataset as _TorchDataset

from .imagenet import IMAGENET_TEMPLATES_SELECT as TEMPLATES

from local_data.constants import PATH_DATASETS


class Dataset(_TorchDataset):

    def __init__(self, transform=None):

        # Transform to apply
        self.transform = transform

        # Set image directories
        root = os.path.abspath(PATH_DATASETS)
        self.dataset_dir = os.path.join(root, "imagenetv2")
        self.image_dir = os.path.join(self.dataset_dir, "imagenetv2-matched-frequency-format-val")

        # Read classnames files to map codes
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        self.classnames = self.read_classnames(text_file)

        # Read images paths
        self.indexes = list(range(len(self.classnames)))
        self.data = self.read_data(self.classnames)
        self.classnames = list(self.classnames.values())
        print("IN-V2. Num of samples: " + str(len(self.data)), end="\n")

        self.templates = TEMPLATES

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

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = os.listdir(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                items.append({"impath": impath, "label": label, "classname": classname})
        return items