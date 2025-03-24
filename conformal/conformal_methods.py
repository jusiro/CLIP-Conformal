import torch
import time

from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import THR
from torchcp.classification.scores.base import BaseScore


def lac(calib_preds, calib_labs, val_preds, alpha):
    """
        Perform the threshold conformal prediction.
        https://arxiv.org/abs/1609.00451
        arguments:
            calib_preds [numpy.array]: calibration logits values
            calib_labs [numpy.array]: calibration labels
            val_preds [numpy.array]: validation logits values
            alpha [float]: value for (1-alpha) coverage
        returns:
            val_pred_sets [numpy.array]: predicted sets on the validation set
    """

    # Prepare conformal score method
    conformal_predictor = SplitPredictor(THR(score_type="Identity"))

    # Fit conformal method
    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    # Predict on validation set
    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def aps(calib_preds, calib_labs, val_preds, alpha):
    """
        Perform the adaptive prediction sets algorithm.
        https://arxiv.org/abs/2006.02544
        arguments:
            calib_preds [numpy.array]: calibration logits values
            calib_labs [numpy.array]: calibration labels
            val_preds [numpy.array]: validation logits values
            alpha [float]: value for (1-alpha) coverage
        returns:
            val_pred_sets [numpy.array]: predicted sets on the validation set
    """

    # Prepare conformal score method
    conformal_predictor = SplitPredictor(APS())

    # Fit conformal method
    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    # Predict on validation set
    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def raps(calib_preds, calib_labs, val_preds, alpha, lambda_raps, k_raps):
    """
        Perform the regularized adaptive prediction sets algorithm.
        https://arxiv.org/abs/2009.14193
        arguments:
            calib_preds [numpy.array]: calibration logits values
            calib_labs [numpy.array]: calibration labels
            val_preds [numpy.array]: validation logits values
            alpha [float]: value for (1-alpha) coverage
            lambda_raps [float]: the penalty multiplier
            k_raps [int]: regulariation hyperparameter
        returns:
            val_pred_sets [numpy.array]: predicted sets on the validation set
    """
    assert lambda_raps is not None, 'lambda_raps can not be None.'
    assert k_raps is not None, 'k_raps can not be None.'

    # Prepare conformal score method
    conformal_predictor = SplitPredictor(RAPS(lambda_raps, k_raps))

    # Fit conformal method
    time_conf_fit_i_1 = time.time()
    conformal_predictor.calculate_threshold(calib_preds, calib_labs, alpha)
    time_conf_fit_i_2 = time.time()
    time_fit = time_conf_fit_i_2 - time_conf_fit_i_1

    # Predict on validation set
    time_conf_infer_i_1 = time.time()
    val_pred_sets = conformal_predictor.predict_with_logits(val_preds)
    time_conf_infer_i_2 = time.time()
    time_infer = time_conf_infer_i_2 - time_conf_infer_i_1

    return val_pred_sets, time_fit, time_infer


def conformal_method(method, calib_preds, calib_labs, val_preds, alpha, lambda_raps=0.001, k_raps=1):
    if method == 'aps':
        return aps(calib_preds, calib_labs, val_preds, alpha)
    elif method == 'raps':
        return raps(calib_preds, calib_labs, val_preds, alpha, lambda_raps, k_raps)
    elif method == 'lac':
        return lac(calib_preds, calib_labs, val_preds, alpha)
    else:
        raise NotImplementedError


class APS(BaseScore):
    """
    Adaptive Prediction Sets (Romano et al., 2020)
    paper :https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    """

    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        if label is None:
            return self._calculate_all_label(logits)
        else:
            return self._calculate_single_label(logits, label)

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        # ordered: the ordered probabilities in descending order
        # indices: the rank of ordered probabilities in descending order
        # cumsum: the accumulation of sorted probabilities
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)


class RAPS(APS):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020)
    paper : https://arxiv.org/abs/2009.14193

    :param penalty: the weight of regularization. When penalty = 0, RAPS=APS.
    :param kreg: the rank of regularization which is an integer in [0,labels_num].
    """

    def __init__(self, penalty, kreg=0):

        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
