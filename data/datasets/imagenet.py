import os
import collections.abc
import copy
import numpy as np
import random

from PIL import Image
from torch.utils.data import Subset
from collections import OrderedDict
from torch.utils.data import Dataset as _TorchDataset

from local_data.constants import PATH_DATASETS

IMAGENET_TEMPLATES_SINGLE = ["a photo of a {}."]

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of a {}.",
]

IMAGENET_TEMPLATES_SELECT_FLYP = [
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a sketch of a {}.",
    "a good photo of the {}.",
    "a {} in a video game.",
]

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


class Dataset(_TorchDataset):

    def __init__(self, transform=None, partition="train", shots="all", seed=0):

        # Set val as test partition
        if partition == "test":
            partition = "val"

        # Transform to apply
        self.transform = transform

        # Set image directories
        root = os.path.abspath(PATH_DATASETS)
        self.dataset_dir = os.path.join(root, "imagenet")
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # Read classnames files to map codes
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        self.classnames = self.read_classnames(text_file)

        # Read images paths
        self.indexes = list(range(len(self.classnames)))
        self.data = self.read_data(self.classnames, partition)

        # Creating a few-shot dataset for training
        random.seed(seed)
        if (shots != "all") and (partition == "train"):
            data_fs = []
            labels = np.array([sample["label"] for sample in self.data])
            for ilabel in range(len(self.classnames)):
                indexes = list(np.squeeze(np.argwhere(labels == ilabel)))
                indexes_fs = random.sample(indexes, int(shots))
                [data_fs.append(self.data[i]) for i in indexes_fs]
            self.data = data_fs

        self.classnames = list(self.classnames.values())
        print("IN-" + partition + ". Num of samples: " + str(len(self.data)), end="\n")

        self.templates = IMAGENET_TEMPLATES_SELECT

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

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = os.listdir(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                items.append({"impath": impath, "label": label, "classname": classname})

        return items