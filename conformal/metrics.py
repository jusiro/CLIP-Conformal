
import numpy as np


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


def evaluate_conformal(pred_sets, labels, alpha=0.1):

    size = set_size(pred_sets)
    coverage = empirical_set_coverage(pred_sets, labels)
    class_cov_gap = avg_class_coverage_gap(pred_sets, labels, alpha=alpha)

    return [coverage, size, class_cov_gap]


def set_size(pred_sets):
    """
        Compute the size of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
        returns:
            sizes [numpy.array]: mean size of the predicted sets
    """
    mean_size = np.mean([len(pred_set) for pred_set in pred_sets])
    return mean_size


def empirical_set_coverage(pred_sets, labels):
    """
        Compute the empirical coverage of the predicted sets.
        arguments:
            pred_sets [numpy.array]: predicted sets
            labels [numpy.array]: true labels
        returns:
            coverage [float]: empirical coverage
    """
    coverage = np.mean([label in pred_set for label, pred_set in zip(labels, pred_sets)])
    return coverage


def avg_class_coverage_gap(pred_sets, labels, alpha=0.1):

    # Get sample-wise accuracy
    correct = np.int8([labels[i] in pred_sets[i] for i in range(len(labels))])

    violation = []
    for i_label in list(np.unique(labels)):
        idx = np.argwhere(labels == i_label)
        violation.append(abs(correct[idx].mean() - (1 - alpha)))

    # Get mean violation
    covgap = 100 * np.mean(violation)

    return covgap