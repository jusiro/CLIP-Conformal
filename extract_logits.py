"""
Main function for logit extraction using CLIP models
"""

import argparse
import torch
import os
import clip
import time
import datetime
import numpy as np

from data.utils import set_loader
from modeling.utils import extract_vision_features, predict_from_features
from modeling.models import Adapter

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # %---------------------------------------------------------
    # Training data and hyper-params
    batch_size = args.bs
    # Pre-trained VLM
    backbone = args.backbone  # "RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"
    # Datasets
    # Test datasets: "imagenet", "imagenetv2", "imagenet-r", "imagenet-sketch", "imagenet-a"
    test_datasets = args.test_datasets

    # %---------------------------------------------------------

    # Load CLIP Model
    if backbone.split("-")[0] == "CLIP":
        model_clip, transforms = clip.load("-".join(backbone.split("-")[1:]))
        model_clip.to(device).float()
        model_clip.eval()  # Disable dropout and batch-norm statistics update
    elif backbone.split("-")[0] == "MetaCLIP":
        from transformers import AutoProcessor, AutoModel
        # Name to Transformers Huggingface ID.
        name_id_match = {"MetaCLIP-ViT-B/16": "facebook/metaclip-b16-fullcc2.5b",
                         "MetaCLIP-ViT-H/14": "facebook/metaclip-h14-fullcc2.5b"}
        # Load model and processor
        transforms = AutoProcessor.from_pretrained(name_id_match[backbone]).image_processor
        model_clip = AutoModel.from_pretrained(name_id_match[backbone])
        model_clip.to(device).float()
        model_clip.eval()
    else:
        print("Architecture not suported...")
        return

    # Set training dataset
    experiment = {"test": {}}
    # Set testing datasets
    for i in range(len(test_datasets)):
        experiment["test"][i] = {}
        experiment["test"][i]["domain"] = {test_datasets[i]}
        experiment["test"][i]["dataloader"] = set_loader(test_datasets[i], transforms=transforms, batch_size=batch_size)

    # Test on different domains
    time_extraction = []
    for i_domain in range(0, len(experiment["test"])):
        print("  Processing: [{dataset}]".format(dataset=test_datasets[i_domain]))

        # Set classification head
        model_clip = model_clip.to(device)
        adapter = Adapter(model_clip,
                          classnames=experiment["test"][i_domain]["dataloader"].dataset.classnames,
                          adapter="ZS",
                          templates=experiment["test"][i_domain]["dataloader"].dataset.templates,
                          clip_id=backbone).to(device)

        # Extract vision features
        time_adapt_i_1 = time.time()
        id = "./local_data/cache/" + test_datasets[i_domain] + "_" + backbone.lower().replace("/", "_")
        if not os.path.isfile(id + ".npz"):

            print("  Extracting features and saving in disk")
            feats_ds, refs_ds = extract_vision_features(model_clip, experiment["test"][i_domain]["dataloader"],
                                                        clip_id=backbone)

            print("  Extracting logits")
            logits_ds = predict_from_features(adapter, torch.tensor(feats_ds), bs=args.bs, act=False, epsilon=1.0)
            logits_ds = logits_ds.cpu().numpy()
            time_adapt_i_2 = time.time()

            print("  Saving in disk")
            np.savez(id, logits_ds=logits_ds, refs_ds=refs_ds)
        else:
            time_adapt_i_2 = time.time()
        time_adapt_i = time_adapt_i_2 - time_adapt_i_1
        time_extraction.append(time_adapt_i)
        print(str("Feature extraction time: " + str(datetime.timedelta(seconds=time_adapt_i))))
    print("Average time: " + str(datetime.timedelta(seconds=np.mean(time_extraction))))


def main():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--test_datasets',
                        default='imagenet,imagenet-a,imagenetv2,imagenet-r,imagenet-sketch,'
                                'sun397,aircraft,eurosat,stanford_cars,food101,oxford_pets,flowers,caltech,dtd,ucf',
                        help='imagenet,imagenet-a,imagenetv2,imagenet-r,imagenet-sketch,'
                             'sun397,aircraft,eurosat,stanford_cars,food101,oxford_pets,flowers,caltech,dtd,ucf',
                        type=lambda s: [item for item in s.split(',')])

    # Model to employ
    parser.add_argument('--backbone', default='CLIP-ViT-B/16',
                        help='"CLIP-RN50", "CLIP-RN101", "CLIP-ViT-B/32","CLIP-ViT-B/16", "CLIP-ViT-L/14",'
                             '"MetaCLIP-ViT-B/16", "MetaCLIP-ViT-H/14"')

    # Other hyper-param
    parser.add_argument('--bs', default=128, type=int, help='Batch size')

    # Resources
    parser.add_argument('--cache_features', default=False, type=lambda x: (str(x).lower() == 'true'))

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()