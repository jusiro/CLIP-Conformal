import os
import collections.abc
import copy

from PIL import Image
from torch.utils.data import Subset
from collections import OrderedDict
from torch.utils.data import Dataset as _TorchDataset

from .misc import find_imagenet_r_indexes
from .imagenet import IMAGENET_TEMPLATES_SELECT as TEMPLATES

from local_data.constants import PATH_DATASETS
TO_BE_IGNORED = ["README.txt"]


class Dataset(_TorchDataset):

    def __init__(self, transform=None):

        # Transform to apply
        self.transform = transform

        # Set image directories
        root = os.path.abspath(PATH_DATASETS)
        self.dataset_dir = os.path.join(root, "imagenet-rendition")
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        # Read classnames files to map codes
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        self.classnames = self.read_classnames(text_file)

        # Read images paths
        self.indexes = find_imagenet_r_indexes()
        self.data = self.read_data(self.classnames)
        self.classnames = [list(self.classnames.values())[i] for i in self.indexes]
        print("IN-R. Num of samples: " + str(len(self.data)), end="\n")

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
        folders = os.listdir(image_dir)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        folders.sort()
        items = []

        for label, folder in enumerate(folders):
            imnames = os.listdir(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                items.append({"impath": impath, "label": label, "classname": classname})

        return items
