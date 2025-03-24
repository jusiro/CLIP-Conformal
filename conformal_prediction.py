"""
Main function for conformal prediction using zero-shot, Conf-OT and baseline methods for transfer learning.
It includes three non-conformity scores: LAC, APS, and RAPS.
"""

import argparse
import torch
import os
import conformal
import time

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime

from conformal.metrics import evaluate_conformal, accuracy
from solvers.transductive import confot, TransCLIP, TIM
from solvers.inductive import Linear

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds for reproducibility
from utils.misc import set_seeds
set_seeds(42, use_cuda=device == 'cuda')


def process(args):

    # Prepare table to save results
    res = pd.DataFrame()

    # %---------------------------------------------------------
    # Calibration + Testing on different domains
    results_detailed = {}
    for i_domain in range(0, len(args.test_datasets)):
        print("  Testing on: [{dataset}]".format(dataset=args.test_datasets[i_domain]))

        # Load data
        id = "./local_data/cache/" + args.test_datasets[i_domain] + "_" + args.backbone.lower().replace("/", "_")
        if os.path.isfile(id + ".npz"):
            print("  Loading features from cache_features")
            cache = np.load(id + ".npz", allow_pickle=True)
            logits_ds, labels_ds = torch.tensor(cache["logits_ds"]), torch.tensor(cache["refs_ds"]).to(torch.long)
        else:
            continue

        # Run for different seeds
        emp_cov, set_size, strat_covgap, class_covgap = [], [], [], []
        top1, top5 = [], []
        time_adapt, time_conf_fit, time_conf_inf = [], [], []

        for _ in tqdm(range(args.seeds), leave=False, desc="  Conformal inference: "):
            torch.cuda.empty_cache()

            # Calibration + validation split
            logits_calib, labels_calib, logits_test, labels_test = conformal.split_data(logits_ds, labels_ds, p=args.p)

            # Combine both sets based again based on new ordering
            logits_ds, labels_ds = torch.cat([logits_calib, logits_test]), torch.cat([labels_calib, labels_test])

            # Conf-OT transductive transfer learning approach
            time_adapt_i_1 = time.time()
            if args.adapt == "confot":
                # Compute codes trough optimal transport
                z = confot.compute_codes(logits_ds, epsilon=args.epsilon, num_iters=args.ot_iters,
                                         observed_marginal=args.observed_marginal,
                                         labels_count=np.bincount(labels_calib)).to("cpu")
            elif args.adapt == "tim":
                that = 100  # Default value for hyper-parameter in TIM
                # Compute codes using Transductive Information Maximization solver
                z, that = TIM.compute_codes(logits_ds, observed_marginal=args.observed_marginal,
                                            labels_count=np.bincount(labels_calib), that=that, hp_search=(_ == 0),
                                            labels_calib=labels_calib)
            elif args.adapt == "transclip":
                # Compute codes using TransCLIP GMM solver
                z = TransCLIP.compute_codes(logits_ds, labels_ds)

            elif args.adapt == "linear_probe":
                # Compute codes using supervised linear probe adjusted on calibration data
                z, that = Linear.compute_codes(logits_calib, labels_calib, logits_test, hp_search=(_ == 0))
            else:
                # No adaptation - temperature scaling softmax of logit scores
                z = torch.softmax(torch.tensor(logits_ds)/args.epsilon, dim=-1)

            # Set transfer learning times
            time_adapt_i_2 = time.time()
            time_adapt_i = time_adapt_i_2 - time_adapt_i_1

            # Retrieve calibration and validation predictions
            preds_calib, preds_test = z[:len(labels_calib), :], z[len(labels_calib):, :]

            # apply the conformal algorithms

            val_sets, time_fit_i, time_infer_i = conformal.conformal_method(
                args.ncscore, preds_calib, labels_calib, preds_test, args.alpha)

            #  Run metrics
            metrics_conformal = evaluate_conformal(val_sets, labels_test, alpha=args.alpha)
            metrics_accuracy = accuracy(preds_test, labels_test, (1, 5))
            # Allocate conformal inference metrics
            emp_cov.append(metrics_conformal[0]), set_size.append(metrics_conformal[1])
            class_covgap.append(metrics_conformal[2])
            # Training times
            time_adapt.append(time_adapt_i), time_conf_fit.append(time_fit_i), time_conf_inf.append(time_infer_i)
            # Allocate accuracy-related metrics
            top1.append(metrics_accuracy[0].item()), top5.append(metrics_accuracy[1].item())

            # Output metrics
            print('  Empirical Coverage: [{cover}] -- Set Size: [{size}] -- '
                  'class_covgap: [{class_covgap}]'.format(cover=np.round(emp_cov[-1], 3),
                                                          size=np.round(set_size[-1], 2),
                                                          class_covgap=np.round(class_covgap[-1], 3)))
            print('  Top-1 Accuracy: [{top1}] -- Top-5 Accuracy: [{top5}]'.format(
                top1=np.round(np.median(top1[-1]), 3),
                top5=np.round(np.median(top5[-1]), 2)))

        # Save detailed results
        results_detailed[args.test_datasets[i_domain]] = {}
        results_detailed[args.test_datasets[i_domain]]["cov"] = emp_cov
        results_detailed[args.test_datasets[i_domain]]["set_size"] = set_size
        results_detailed[args.test_datasets[i_domain]]["class_covgap"] = class_covgap
        results_detailed[args.test_datasets[i_domain]]["top1"] = top1

        # Output metrics
        print("  " + "%" * 100)
        print('  [AVG] Empirical Coverage: [{cover}] -- Set Size: [{size}] -- '
              'class_covgap: [{class_covgap}]'.format(cover=np.round(np.median(emp_cov), 3),
                                                      size=np.round(np.median(set_size), 2),
                                                      class_covgap=np.round(np.median(class_covgap), 3)))
        print('  [AVG] Top-1 Accuracy: [{top1}] -- Top-5 Accuracy: [{top5}]'.format(
            top1=np.round(np.median(top1), 3),
            top5=np.round(np.median(top5), 2)))
        print("  " + "%" * 100)

        # Prepare results
        res_i = {"backbone": args.backbone, "dataset": args.test_datasets[i_domain], "alpha": args.alpha,
                 "adapt": args.adapt, "ncscore": args.ncscore, "epsilon": args.epsilon,
                 "ot_iters": args.ot_iters, "observed_marginal": str(args.observed_marginal),
                 "prop. calib": args.p, "top1": np.round(np.median(top1), 3), "cov":  np.round(np.median(emp_cov), 3),
                 "size": np.round(np.median(set_size), 2), "CCV":  np.round(np.median(class_covgap), 3),
                 "time_adapt": np.round(np.mean(time_adapt), 6), "time_conf_fit": np.round(np.mean(time_conf_fit), 6),
                 "time_conf_inf": np.round(np.mean(time_conf_inf), 6)}
        res = pd.concat([res, pd.DataFrame(res_i, index=[0])])

    # Produce average results
    avg = res[["top1", "cov", "size", "CCV", "time_adapt", "time_conf_fit", "time_conf_inf"]].mean().values
    res_avg = {"backbone": args.backbone, "dataset": "AVG", "alpha": args.alpha, "adapt": args.adapt,
               "ot_iters": args.ot_iters, "observed_marginal": str(args.observed_marginal),
               "ncscore": args.ncscore, "epsilon": args.epsilon, "prop. calib": args.p,
               "top1": np.round(avg[0], 3), "cov": np.round(avg[1], 3), "size": np.round(avg[2], 2),
               "CCV": np.round(avg[3], 3), "time_adapt": np.round(avg[4], 6), "time_conf_fit": np.round(avg[5], 6),
               "time_conf_inf": np.round(avg[6], 6)}
    res = pd.concat([res, pd.DataFrame(res_avg, index=[0])])

    timestamp = datetime.now().strftime("-%m-%d_%H-%M-%S")
    # save summary results
    path = "./local_data/results/{backbone}/{alpha}/{ncscore}/summary/".format(
        backbone=args.backbone.replace("/", ""), alpha=str(args.alpha).replace(".", ""),
        ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame.to_excel(res, path + args.adapt + timestamp + ".xlsx")

    # save detailed results
    path = "./local_data/results/{backbone}/{alpha}/{ncscore}/detailed/".format(
        backbone=args.backbone.replace("/", ""), alpha=str(args.alpha).replace(".", ""),
        ncscore=args.ncscore)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + args.adapt + timestamp + ".npy", results_detailed)


def main():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument('--test_datasets',
                        default='dtd,aircraft',
                        help='imagenet,imagenet-a,imagenetv2,imagenet-r,imagenet-sketch,'
                             'sun397,aircraft,eurosat,stanford_cars,food101,oxford_pets,flowers,caltech,dtd,ucf',
                        type=lambda s: [item for item in s.split(',')])

    # Model to employ
    parser.add_argument('--backbone', default='CLIP-ViT-B/16',
                        help='"CLIP-RN50", "CLIP-RN101", "CLIP-ViT-B/32","CLIP-ViT-B/16", "CLIP-ViT-L/14",'
                             '"MetaCLIP-ViT-B/16", "MetaCLIP-ViT-H/14"')

    # Setting for adaptation (OT hyper-parameters)
    parser.add_argument('--adapt', default='linear_probe', help='TL mode',
                        choices=['none', 'linear_probe', 'confot', 'tim', 'transclip'])
    parser.add_argument('--epsilon', default=1.0, help='Entropic multiplier term (temp. scaling)', type=float)
    parser.add_argument('--ot_iters', default=3, help='Iterations during OT cost matrix adjustment', type=int)
    parser.add_argument('--observed_marginal', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Conformal prediction hyperparameters
    parser.add_argument('--alpha', default=0.1, help='Value for the desired coverage.', type=float)
    parser.add_argument('--ncscore', default='lac', help='Non-conformity score', choices=['lac', 'aps', 'raps'])

    # Experimental setting (data) hyperparameters
    parser.add_argument('--p', default=0.5, type=float, help='Percentage of calibration data')

    # Number of seeds
    parser.add_argument('--seeds', default=20, type=int, help='Batch size')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()
