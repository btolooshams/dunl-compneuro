"""
Copyright (c) 2025 Bahareh Tolooshams

plot pca/nmf data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from scipy import stats
import h5py
import os
import argparse

import sys

sys.path.append("../dunl/")

import utils


def read_data(data_fname):
    try:
        with h5py.File(data_fname, "r") as hf:
            data_dict = {k: np.array(v) for k, v in hf.items()}
            return data_dict
    except IOError:
        print("Cannot open %s for reading." % data_fname)
        raise


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="../data/dopamine-spiking-eshel-uchida/lfads/dopamine_spiking_eshel_uchida_for_lfads_for_inference_dataset_N0",
    )
    parser.add_argument(
        "--lfads-res-path",
        type=str,
        help="lfads res path",
        default="../../lfads/results/dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p1train_dataset_N0_factordim2_for_inference",
    )
    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../figures/lfads/dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p1train_dataset_N0_factordim2_for_inference",
    )
    parser.add_argument(
        "--num-comp",
        type=int,
        help="number of components",
        default=2,
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--n-bin-spearman",
        type=int,
        help="number of bins in spearman histogram",
        default=10,
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "orange",
            "blue",
            "red",
            "black",
        ],
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(4, 2),
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    filename_partial = params["lfads_res_path"].split("/")[-1]

    # ----------------------------------------------------------------#
    # ----------------------------------------------------------------#
    # raw_data_dict = read_data(params["data_path"])
    # res_data_dict = read_data(os.path.join(params["lfads_res_path"], "model_runs__train_posterior_sample_and_average"))

    # spike_counts = np.sum(raw_data_dict["train_data"], axis=1)
    # neurons = raw_data_dict["train_neuron"]
    # labels = raw_data_dict["train_label"]
    # factors = res_data_dict["factors"]
    # factors_counts = np.sum(res_data_dict["factors"], axis=1)

    # data_dict = {
    #     "spike_counts": spike_counts,
    #     "neurons": neurons,
    #     "labels": labels,
    #     "factors": factors,
    #     "factors_counts": factors_counts,
    # }

    # torch.save(data_dict, f"../data/dopamine-spiking-eshel-uchida/lfads/train_data_dict_{filename_partial}.py")
    # exit()
    # ----------------------------------------------------------------#
    # ----------------------------------------------------------------#

    data_dict = torch.load(
        f"../data/dopamine-spiking-eshel-uchida/lfads/train_data_dict_{filename_partial}.py"
    )

    spike_counts = data_dict["spike_counts"]
    neurons = data_dict["neurons"]
    labels = data_dict["labels"]
    factors = data_dict["factors"]
    factors_counts = data_dict["factors_counts"]

    print(data_dict.keys)

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    avg_factors = np.mean(factors, axis=0)

    print(spike_counts.shape)
    print(neurons.shape)
    print(labels.shape)
    print(factors_counts.shape)
    print(factors.shape)

    num_neurons = max(neurons) + 1
    num_reward_types = 6

    sign = -1

    x_corr_sur = np.zeros((num_neurons, params["num_comp"]))
    x_corr_exp = np.zeros((num_neurons, params["num_comp"]))

    y_corr_sur = np.zeros((num_neurons))
    y_corr_exp = np.zeros((num_neurons))

    # factors_per_reward_amount_sur = np.zeros((num_reward_types, factors.shape[-2], factors.shape[-1]))
    # factors_per_reward_amount_exp = np.zeros((num_reward_types, factors.shape[-2], factors.shape[-1]))

    for neuron_ctr in range(num_neurons):
        neuron_indices = neurons == neuron_ctr

        label_neuron = labels[neuron_indices]
        spike_counts_neuron = spike_counts[neuron_indices]

        factor_counts_neuron = np.mean(factors, axis=1)[neuron_indices]

        for sur_or_exp in [True, False]:
            # surprise has negative sign in label
            if sur_or_exp:
                indices = label_neuron < 0
            else:
                indices = label_neuron > 0

            label_curr = np.abs(label_neuron[indices])
            spike_counts_curr = spike_counts_neuron[indices]
            factor_curr = factor_counts_neuron[indices]
            print(factor_curr.shape)

            y_corr, _ = stats.spearmanr(spike_counts_curr, label_curr)
            if sur_or_exp:
                y_corr_sur[neuron_ctr] = y_corr
            else:
                y_corr_exp[neuron_ctr] = y_corr

            for factor_ctr in range(params["num_comp"]):
                x_corr_curr, _ = stats.spearmanr(
                    sign * factor_curr[:, factor_ctr], label_curr
                )

                if sur_or_exp:
                    x_corr_sur[neuron_ctr, factor_ctr] = x_corr_curr
                else:
                    x_corr_exp[neuron_ctr, factor_ctr] = x_corr_curr


if __name__ == "__main__":
    main()
