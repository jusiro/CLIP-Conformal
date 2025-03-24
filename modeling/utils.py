import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_vision_features(model_clip, loader, augmentations=False, clip_id="CLIP"):

    # Set repetitions based on required augmentations
    reps = 10 if augmentations else 1

    # Loop to extract features
    refs_ds_rep, feats_ds_rep = [], []
    for irep in range(reps):
        refs_ds, feats_ds = [], []
        for step, batch in enumerate(loader):
            print("  Batch {ii}/{II}".format(
                ii=step + 1, II=len(loader)), end="\r")

            # Retrieve images and labels
            images = batch["img"].to(device).to(torch.float32)
            refs_batch = batch["label"].to(device).to(torch.float32)

            # Forward predictions
            with torch.no_grad():
                if clip_id.split("-")[0] == "CLIP":
                    feats = model_clip.visual(images)
                elif clip_id.split("-")[0] == "MetaCLIP":
                    feats = model_clip.get_image_features(images["pixel_values"][0].to(device))

            # Store labels and predictions
            refs_ds.append(refs_batch.detach().cpu().numpy()), feats_ds.append(feats.detach().cpu().numpy())

        # Concatenate features and refs
        refs_ds = np.concatenate(refs_ds)
        feats_ds = np.concatenate(feats_ds, axis=0)

        refs_ds_rep.append(np.expand_dims(refs_ds, -1)), feats_ds_rep.append(np.expand_dims(feats_ds, -1))

    # Concatenate augmentations
    refs_ds_rep = np.squeeze(np.concatenate(refs_ds_rep, axis=-1))
    feats_ds_rep = np.squeeze(np.concatenate(feats_ds_rep, axis=-1))

    return feats_ds_rep, refs_ds_rep


def predict_from_features(adapter, feats_ds, bs=512, act=True, epsilon=1.0):
    preds, idx = [], 0
    while idx <= feats_ds.shape[0]:

        # Retrieve features
        x = feats_ds[idx:idx+bs, :].to(device).to(torch.float32)

        # Forward predictions
        with torch.no_grad():
            pred = adapter(x)
            if act:
                pred = torch.softmax(pred / epsilon, dim=-1)

        # Store labels and predictions
        preds.append(pred.detach().cpu())

        # Update iterator
        idx += bs

    # Concatenate predictions
    preds = torch.cat(preds, axis=0)

    return preds
