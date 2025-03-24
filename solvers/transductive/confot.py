import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_codes(logits, epsilon=0.8, num_iters=3, observed_marginal=False, labels_count=None):

    # Compute similarities and soft codes for all crops together
    with torch.no_grad():
        # Estimate marginal distributions
        r, c = None, None
        if observed_marginal:
            label_dist = labels_count / np.sum(labels_count)
            r = torch.tensor(label_dist).to(device)

        # Compute soft code pseudo-labels
        soft_code = distributed_sinkhorn(logits, epsilon=epsilon, num_iters=num_iters, r=r, c=c)

    return soft_code


def distributed_sinkhorn(similarities, epsilon=0.8, num_iters=3, r=None, c=None):
    similarities = similarities.to(device)
    Q = torch.exp(similarities / epsilon).t()  # Q is K-by-B for consistency with notations
    K, B = Q.shape  # Number of clusters and batch size

    if r is None:
        r = torch.ones(K).to(similarities.device) / K
    if c is None:
        c = torch.ones(B).to(similarities.device) / B

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(num_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q *= r.unsqueeze(1)

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q *= c.unsqueeze(0)

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    Q = Q.t()

    return Q