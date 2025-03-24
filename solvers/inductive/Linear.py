import torch
import random
import tqdm

import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Linear probe fitting in the logit space. The class prototypes are initialized in the simplex corners, and then adjusted
based on labeled data by minimizing cross-entropy using gradient descent.
"""


def compute_codes(logits_calib, labels_calib, logits_test, that=100, hp_search=False):

    # Run hyper-param search for temp-scaling parameter
    if hp_search:
        torch.cuda.empty_cache()
        print("Searching for best config based on calib supervision")
        search_t = [0.1, 1, 5, 10, 15, 30, 60, 100, 200]  # Grid of search for tau
        that, best = None, 0
        for t_i in tqdm.tqdm(search_t):
            z = linear(logits_calib, labels_calib, logits_test, t=t_i, disp=False)
            acc = round(accuracy(z[:labels_calib.shape[0], :].cpu(), labels_calib, (1, 5))[0].item(), 2)
            if acc > best:
                that, best = t_i, acc
                print("Best running config: acc={acc}-temp={temp}".format(acc=acc, temp=that))
        print("", end="\n")
        print("Best config: temp={temp}".format(temp=that))
    else:
        print("No hyper-param grid search")

    # Compute codes trough optimal transport
    z = linear(logits_calib, labels_calib, logits_test, t=that, disp=False)
    torch.cuda.empty_cache()

    return z, that


def linear(logits_calib, labels_calib, logits_test, base_lr=0.1, iterations=1000, bs=2048, disp=True, cov=False, t=100):

    # Get dimensions
    n, K = logits_calib.shape

    # Move input to cpu
    logits_calib = logits_calib.cpu()

    # l2-norm logits
    logits_calib /= logits_calib.norm(dim=-1, keepdim=True)
    logits_test /= logits_test.norm(dim=-1, keepdim=True)

    # Calculate epochs to perform 100 iterations
    epochs = int(iterations / (n/bs))

    # Produce Adapter
    Adapter = LinearProbeHead(K, t=t, cov=cov).to(device).float()

    # Set training optimizer
    optim = torch.optim.Adam(params=Adapter.parameters(), lr=base_lr)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=iterations)

    # Fit Adapter
    for i_epoch in range(epochs):

        # Set training indexes
        indexes = np.arange(0, n)
        random.shuffle(indexes)  # Shuffle
        tracking_loss = 0.0

        for i_step in range(max(1, n // bs)):

            # Select batch indexes
            init = int(i_step * bs)
            end = int((1 + i_step) * bs)

            # Retrieve features
            x = logits_calib[indexes[init:end], :].to(device).to(torch.float32)

            # Retrieve labels
            y = labels_calib[indexes[init:end]].to(device).to(torch.long)

            # Forward
            logits_prima = Adapter.forward(x)

            # Adapter loss
            loss = torch.nn.functional.cross_entropy(logits_prima, y)

            # Update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # Update tracking loss
            tracking_loss += loss.item() / (max(1, n // bs))

            # Update scheduler
            scheduler.step()

            if disp:
                print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                    i_epoch=i_epoch + 1, epochs=epochs, i_step=i_step + 1, steps=int(n // bs),
                    loss=round(loss.item(), 4)), end="\r")

        if disp:
            print("Epoch {i_epoch}/{epochs} -- Iteration {i_step}/{steps} -- loss={loss}".format(
                i_epoch=i_epoch + 1, epochs=epochs, i_step=int(n // bs),
                steps=int(n // bs), loss=round(tracking_loss, 4)), end="\n")

    Adapter.eval()

    with torch.no_grad():
        z = torch.softmax(Adapter(torch.cat([logits_calib, logits_test], dim=0).to(device)), -1).cpu()

    return z


class LinearProbeHead(torch.nn.Module):
    def __init__(self, K, t=1, cov=True):
        super().__init__()
        self.cov = cov
        self.K = K
        self.t = t

        # Produce initial class prototypes
        mu = 100 * torch.nn.functional.one_hot(torch.arange(0, K)).float().to(device)

        # Trainable parameters
        self.mu = torch.nn.Parameter(mu.clone())
        self.mu.requires_grad = True

        # Norm parameters
        if self.cov:
            self.std = (torch.eye(K).diag() * 1/K).cuda()
            self.std.requires_grad = True

    def forward(self, x):

        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)

        chunk_size = 250
        N = x.shape[0]
        M = self.mu.shape[0]

        logits = torch.empty((N, M), dtype=x.dtype, device=x.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            if not self.cov:
                logits[start_idx:end_idx] = - self.t * 0.5 * torch.sum(
                    (x[start_idx:end_idx][:, None, :] - mu[None, :, :]) ** 2, dim=-1)
            else:
                logits[start_idx:end_idx] = -0.5 * torch.einsum(
                    'ijk,ijk->ij',
                    (x[start_idx:end_idx][:, None, :] - mu[None, :, :]) ** 2,
                    1 / self.std[None, None, :])

        return logits


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res