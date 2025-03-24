import torch
import torch.nn.functional as F
import torch.nn as nn
import time

"""
TransCLIP solver for VLMs transductive adaptation - modified for operating on the logit space.
Base code:
    https://github.com/MaxZanella/transduction-for-vlms
"""


def compute_codes(logits, labels):

    # L2-norm logits
    logits = (logits / logits.norm(dim=-1, keepdim=True)).to("cuda")

    # Init class prototypes in the simplex corners
    mu = 100 * torch.nn.functional.one_hot(torch.arange(0, logits.shape[-1])).float()
    mu = (mu / mu.norm(dim=-1, keepdim=True)).to("cuda")

    # Fit GMM adapter
    z, _ = TransCLIP_solver(support_features=None, support_labels=None, val_features=None,
                            val_labels=None, query_features=logits, query_labels=labels,
                            clip_prototypes=mu, initial_prototypes=None, initial_predictions=None,
                            verbose=True)
    torch.cuda.empty_cache()

    return z


def TransCLIP_solver(support_features, support_labels, val_features, val_labels, query_features, query_labels,
                     clip_prototypes, initial_prototypes=None, initial_predictions=None, verbose=True):
    """
    This function implement our TransCLIP method and computes the Top-1 test accuracy for a given set of embeddings
    and labels. The computation involves several inputs, including embeddings for shots and validation samples,
    labels for shots, validation, and test samples, and text embeddings represented as clip weights. These inputs are
    provided as PyTorch tensors.

    |S|: total number of shots. |Q|: number of test samples. d: dimensionality size of the feature space. K: number of classes.

    Parameters:
    - support_features (torch.Tensor): A tensor of shape (|S|, d) containing embeddings for each shot.
    - support_labels (torch.Tensor): A tensor of shape (|S|, K) containing labels for each shot.
    - val_features (torch.Tensor): A tensor of shape (min(4*K, |S|), d) containing embeddings for validation samples.
    - val_labels (torch.Tensor): A tensor of shape (min(4*K, |S|)) containing labels for validation samples.
    - query_features (torch.Tensor): A tensor of shape (|Q|, d) containing embeddings for query samples.
      '|Q|' represents the number of query samples.
    - query_labels (torch.Tensor): A tensor of shape (|Q|) containing labels for query samples, used solely for evaluation.
    - clip_prototypes (torch.Tensor): A tensor of shape (d, K) containing text embeddings for each class.
    - initial_prototypes (torch.Tensor): A tensor of shape (d, K) containing precomputed class prototypes.
      If initial_prototypes is None, the clip_prototypes are used instead.
    - initial_predictions (torch.Tensor) : A tensor of shape (|Q|, K) that contains query predictions (already softmaxed)
      from another method. Can be used when a method does not compute any class prototypes. If None, clip_prototypes are
      used instead to compute initial pseudo-labels.
    - verbose (bool): indicate if the accuracies should be printed.

    Returns:
    - assignments z: A Tensor of shape (|Q|, K) containing the query set predictions.
    - float: The Top-1 accuracy on the test set, evaluated using the specified embeddings and labels.
    """

    start_time = time.time()

    ################
    # General init #
    ################

    K = len(torch.unique(query_labels))
    d = query_features.size(1)
    num_samples = query_features.size(0)

    y_hat, query_features, query_labels, val_features, val_labels, support_features, support_labels, neighbor_index = \
        prepare_objects(query_features, query_labels,
                        val_features, val_labels,
                        support_features, support_labels,
                        clip_prototypes, initial_prototypes,
                        initial_predictions, verbose=verbose)

    max_iter = 10  # number of iterations
    std_init = 1 / d
    n_neighbors = 3
    best_val = -1  # to keep track in few-shot
    test_acc_at_best_val = -1  # to keep track in few-shot

    # Get hyperparameter ranges and prepare shots if any

    lambda_value, gamma_list, support_features, support_labels = get_parameters(support_labels, support_features)

    ##########
    # Z init #
    ##########

    y_hat, z = init_z(y_hat, softmax=False if initial_predictions is not None else True)

    ###########
    # MU init #
    ###########

    mu = init_mu(K, d, z, query_features, support_features, support_labels)

    ##############
    # SIGMA init #
    ##############

    std = init_sigma(d, std_init)

    adapter = Gaussian(mu=mu, std=std).cuda()

    ###################
    # Affinity matrix #
    ###################

    W = build_affinity_matrix(query_features, support_features, num_samples, n_neighbors)

    # Iterate over gamma (gamma = [0] in zero-shot):

    for idx, gamma_value in enumerate(gamma_list):

        for k in range(max_iter + 1):
            gmm_likelihood = adapter(query_features, no_exp=True)

            ############
            # Z update #
            ############

            new_z = update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, support_labels)[0:num_samples]
            z = new_z
            if k == max_iter:  # STOP
                acc = cls_acc(z, query_labels)
                if support_features is not None:  # Few-shot : validate gamma
                    acc_val = cls_acc(z[neighbor_index, :], val_labels)
                    if acc_val > best_val:
                        best_val = acc_val
                        test_acc_at_best_val = acc

                else:
                    acc = cls_acc(z, query_labels)
                    if verbose:
                        print("\n**** TransCLIP's test accuracy: {:.2f} ****\n".format(acc))
                break

            #############
            # MU update #
            #############

            adapter = update_mu(adapter, gamma_value, query_features, z, support_features, support_labels)

            ################
            # SIGMA update #
            ################

            adapter = update_sigma(adapter, gamma_value, query_features, z, support_features, support_labels)

        if support_features is not None:
            if verbose:
                print("{}/{} TransCLIP's test accuracy: {:.2f} on test set @ best validation accuracy ({:.2f})".format(
                    idx + 1, len(gamma_list), test_acc_at_best_val, best_val))
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return z, test_acc_at_best_val


def update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, labels=None, max_iter=5):
    few_shot = labels is not None
    if few_shot:
        shots_labels = F.one_hot(labels).float()
        z = torch.cat((z.clone(), shots_labels))

    num_samples = gmm_likelihood.size(0)

    for it in range(max_iter):
        intermediate = gmm_likelihood.clone()

        intermediate += (50 / (n_neighbors * 2)) * (
                W.T @ z + (W @ z[0:num_samples, :])[0:num_samples, :])

        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_value) * torch.exp(1 / 50 * intermediate)
        z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)

    return z


def update_mu(adapter, gamma_value, query_features, z, support_features, labels):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    weights = (1 / n_query) * affinity_unlabeled

    # Use einsum to compute the new_mu for each class in one pass
    new_mu = torch.einsum('ij,ik->jk', weights, query_features)

    if few_shot:
        weights = (gamma_value * 50 / n_support) * affinity_labeled
        new_mu += torch.einsum('ij,ik->jk', weights, support_features)

        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(
            -1) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled, dim=0).unsqueeze(-1))
    else:
        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(-1))
    new_mu = new_mu.unsqueeze(1)

    new_mu /= new_mu.norm(dim=-1, keepdim=True)

    adapter.mu = new_mu

    return adapter


def update_sigma(adapter, gamma_value, query_features, z, support_features, labels):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    std = 0

    chunk_size = 500  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]

        # Compute the weighted sum of squared differences for the chunk
        chunk_result = (1 / n_query) * torch.einsum(
            'ij,ijk->k',
            affinity_unlabeled[start_idx:end_idx, :],
            # Use a chunk of affinity_unlabeled
            (query_features_chunk[:, None, :] - adapter.mu[None, :,
                                               0, :]) ** 2)

        # If this is the first chunk, initialize std; otherwise, accumulate
        if start_idx == 0:
            std = chunk_result
        else:
            std += chunk_result

    if few_shot and gamma_value > 0:
        # Iterate over query_features in chunks
        for start_idx in range(0, n_support, chunk_size):
            end_idx = min(start_idx + chunk_size, n_support)
            support_features_chunk = support_features[
                                     start_idx:end_idx]

            # Compute the weighted sum of squared differences for the chunk
            chunk_result = (gamma_value * 50 / n_support) * torch.einsum(
                'ij,ijk->k',
                affinity_labeled[start_idx:end_idx, :],
                # Use the relevant part of affinity_unlabeled
                (support_features_chunk[:, None, :] - adapter.mu[
                                                      None, :, 0,
                                                      :]) ** 2
            )

            std += chunk_result

        std /= (1 / n_query * torch.sum(
            affinity_unlabeled[:,
            :]) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled[:, :]))
    else:
        std /= (1 / n_query * torch.sum(
            affinity_unlabeled[:, :]))

    adapter.set_std(std)
    return adapter


def init_z(affinity, softmax=True):
    if softmax:
        y_hat = F.softmax(affinity, dim=1)
        z = F.softmax(affinity, dim=1)
    else:
        y_hat = affinity
        z = affinity
    return y_hat, z


def init_mu(K, d, z, query_features, support_features, support_labels):
    few_shot = support_features is not None
    if few_shot:
        support_labels_one_hot = F.one_hot(support_labels).float()
        t = support_features.cuda().squeeze()
        mu = support_labels_one_hot.t() @ t
        mu = mu.unsqueeze(1)
    else:
        mu = torch.zeros(K, 1, d,
                         device=query_features.device)
        n_most_confident = 8
        topk_values, topk_indices = torch.topk(z, k=n_most_confident, dim=0)  # 8 pseudo-labels per class

        mask = torch.zeros_like(z).scatter_(0, topk_indices, 1)
        filtered_z = z * mask
        for c in range(K):
            class_indices = mask[:, c].nonzero().squeeze(1)
            class_features = query_features[class_indices]
            class_z = filtered_z[
                class_indices, c].unsqueeze(
                1)

            combined = class_features * class_z
            component_mean = combined[:n_most_confident].mean(dim=0)
            mu[c, 0, :] = component_mean
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def init_sigma(d, std_init):
    std = (torch.eye(d).diag() * std_init).cuda()

    return std


def get_parameters(support_labels, shot_features):
    few_shot = shot_features is not None
    if few_shot:
        lambda_value = 0.5
        gamma_list = [0.002, 0.01, 0.02, 0.2]

        support_labels = support_labels.max(dim=1)[1].cuda()

        support_features = shot_features.squeeze().cuda()

    else:
        lambda_value = 10
        gamma_list = [0]
        support_labels = None
        support_features = None

    return lambda_value, gamma_list, support_features, support_labels


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.mu = mu.clone()

        self.K, self.num_components, self.d = self.mu.shape
        self.std = std.clone()

        self.mixing = torch.ones(self.K, self.num_components, device=self.mu.device) / self.num_components

    def forward(self, x, get_components=False, no_exp=False):
        chunk_size = 500
        N = x.shape[0]
        M, D = self.mu.shape[0], self.std.shape[0]

        intermediate = torch.empty((N, M), dtype=x.dtype, device=x.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            intermediate[start_idx:end_idx] = -0.5 * torch.einsum('ijk,ijk->ij',
                                                                  (x[start_idx:end_idx][:, None, :] - self.mu[None, :,
                                                                                                      0, :]) ** 2,
                                                                  1 / self.std[None, None, :])

        if not no_exp:
            intermediate = torch.exp(intermediate)

        if get_components:
            return torch.ones_like(intermediate.unsqueeze(1))

        return intermediate

    def set_std(self, std):
        self.std = std


def build_affinity_matrix(query_features, support_features, num_samples, n_neighbors=3):
    device = query_features.device
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = (support_features.matmul(query_features.t())).cpu()
        affinity_test = (query_features.matmul(query_features.t())).cpu()
        affinity = torch.cat((affinity_labeled, affinity_test), dim=0)
        num_rows = num_samples + support_features.size(0)
        num_cols = num_samples
    else:
        affinity = query_features.matmul(query_features.T).cpu()
        num_rows = num_samples
        num_cols = num_samples

    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols),
                                device=device)
    return W


def prepare_objects(query_features, query_labels, val_features, val_labels, support_features, support_labels,
                    clip_prototypes, initial_prototypes, initial_predictions, verbose=True):

    few_shot = support_features is not None
    query_features = query_features.cuda().float()
    query_labels = query_labels.cuda()
    clip_prototypes = clip_prototypes.cuda().float()
    neighbor_index = None

    if initial_prototypes is not None:
        initial_prototypes = initial_prototypes.cuda().float()

    if initial_predictions is not None:
        initial_predictions = initial_predictions.cuda().float()

    if len(clip_prototypes.shape) == 3:  # use more than 1 template
        clip_prototypes = clip_prototypes[0]  # use only the first one

    if few_shot:
        support_features = support_features.squeeze().float()
        val_features = val_features.cuda().float()
        val_labels = val_labels.cuda()
        # find the nearest neighbor of each validation sample in the query set for prediction on the validation set:
        neighbor_index = torch.argmax(val_features @ query_features.T, dim=1)

    clip_logits = 100 * query_features @ clip_prototypes

    if verbose:
        acc = cls_acc(clip_logits, query_labels)
        print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    if initial_prototypes is not None:
        prototypes_logits = 100 * query_features @ initial_prototypes
        acc = cls_acc(prototypes_logits, query_labels)
        if verbose:
            print("\n**** Prototypes test accuracy: {:.2f}. ****\n".format(acc))
        y_hat = prototypes_logits
    elif initial_predictions is not None:
        y_hat = initial_predictions
        acc = cls_acc(initial_predictions, query_labels)
        if verbose:
            print("\n**** Prototypes test accuracy: {:.2f}. ****\n".format(acc))
    else:
        y_hat = clip_logits

    return y_hat, query_features, query_labels, val_features, val_labels, support_features, support_labels, neighbor_index