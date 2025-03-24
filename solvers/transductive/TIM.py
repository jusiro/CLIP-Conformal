import torch
import random
import tqdm

import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Transductive Information Maximization (TIM): a general framework based on mutual information for transductive
adaptation. We perform minor modifications to include the observed label-marginal distribution in the cal set.
Base formulation:
    https://proceedings.neurips.cc/paper/2020/file/196f5641aa9dc87067da4ff90fd81e7b-Paper.pdf
"""


def compute_codes(logits, observed_marginal=False, labels_count=None, that=100, hp_search=False, labels_calib=None):

    # Get label-marginal distribution
    if observed_marginal:
        label_dist = labels_count / np.sum(labels_count)
        r = torch.tensor(label_dist).to(device)
    else:
        r = torch.ones(logits.shape[-1]).to(device) / logits.shape[-1]

    # Run hyper-param search for temp-scaling parameter
    if hp_search and labels_calib is not None:
        torch.cuda.empty_cache()
        print("Searching for best config based on calib supervision")
        search_t = [0.1, 1, 5, 10, 15, 30, 60, 100]  # Grid of search for tau
        that, best = None, 0
        for t_i in tqdm(search_t):
            z = tim(logits, marginal=r, t=t_i, disp=False)
            acc = round(accuracy(z[:labels_calib.shape[0], :].cpu(), labels_calib, (1, 5))[0].item(), 2)
            if acc > best:
                that, best = t_i, acc
                print("Best running config: acc={acc}-temp={temp}".format(acc=acc, temp=that))
        print("", end="\n")
        print("Best config: temp={temp}".format(temp=that))
    else:
        print("No hyper-param grid search // No calibration labels")

    # Compute codes trough optimal transport
    z = tim(logits, marginal=r, t=that, disp=False)
    torch.cuda.empty_cache()

    return z, that


def tim(logits, marginal, base_lr=0.001, iterations=100, bs=2048, disp=True, cov=False, t=100, alpha=0.1):

    # Get dimensions
    n, K = logits.shape

    # Move input to cpu
    logits = logits.cpu()

    # l2-norm logits
    logits /= logits.norm(dim=-1, keepdim=True)

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
            x = logits[indexes[init:end], :].to(device).to(torch.float32)

            # Forward
            logits = Adapter.forward(x)
            pik = torch.softmax(logits, -1)
            pk = torch.mean(pik, dim=0)

            # Sample shannon entropy
            H_yx = - torch.mean(torch.sum(pik * torch.log(pik + 1e-3), -1))

            # Label marginal entropy
            Lkl = torch.sum(marginal.clone() * torch.log((marginal.clone() / pk) + 1e-3))

            # Overall loss
            loss = Lkl + alpha * H_yx

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
        z = torch.softmax(Adapter(logits).to(device), -1).cpu()

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