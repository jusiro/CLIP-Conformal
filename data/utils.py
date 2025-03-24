from data.datasets import imagenet, imagenetv2, imagenet_sketch, imagenet_a, imagenet_r, oxford_pets, eurosat,\
    stanford_cars, caltech101, sun397, fgvc, food101, oxford_flowers, dtd, ucf101
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import \
    Compose, ToTensor, Normalize, RandomResizedCrop, RandomRotation, RandomHorizontalFlip


def set_loader(id_dataset, transforms=None, partition="test", batch_size=128, shots="all", num_workers=4):

    # Initiate dataset
    if id_dataset == "imagenet":
        dataset = imagenet.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "imagenetv2":
        dataset = imagenetv2.Dataset(transform=transforms)
    elif id_dataset == "imagenet-sketch":
        dataset = imagenet_sketch.Dataset(transform=transforms)
    elif id_dataset == "imagenet-a":
        dataset = imagenet_a.Dataset(transform=transforms)
    elif id_dataset == "imagenet-r":
        dataset = imagenet_r.Dataset(transform=transforms)
    elif id_dataset == "oxford_pets":
        dataset = oxford_pets.Dataset(transform=transforms)
    elif id_dataset == "eurosat":
        dataset = eurosat.Dataset(transform=transforms)
    elif id_dataset == "stanford_cars":
        dataset = stanford_cars.Dataset(transform=transforms)
    elif id_dataset == "caltech":
        dataset = caltech101.Dataset(transform=transforms)
    elif id_dataset == "sun397":
        dataset = sun397.Dataset(transform=transforms)
    elif id_dataset == "aircraft":
        dataset = fgvc.Dataset(transform=transforms)
    elif id_dataset == "food101":
        dataset = food101.Dataset(transform=transforms)
    elif id_dataset == "flowers":
        dataset = oxford_flowers.Dataset(transform=transforms)
    elif id_dataset == "dtd":
        dataset = dtd.Dataset(transform=transforms)
    elif id_dataset == "ucf":
        dataset = ucf101.Dataset(transform=transforms)
    else:
        print("Dataset not supported")
        return None

    # Set dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                        drop_last=False)
    return loader


def augm_transforms(n_px=224):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
        RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
        RandomHorizontalFlip(),
        RandomRotation(degrees=20),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])


def _convert_to_rgb(image):
    return image.convert('RGB')