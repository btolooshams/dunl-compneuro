"""
Copyright (c) 2020 Bahareh Tolooshams

plot pca/nmf data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy as sp
from scipy import stats
import h5py
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys

sys.path.append("../src/")

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
        # default="../data/dopamine-spiking-eshel-uchida/lfads/dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p1train_dataset_N0_factordim2_for_inference",
    )
    parser.add_argument(
        "--lfads-res-path",
        type=str,
        help="lfads res path",
        default="../../lfads/results/dopamine_spiking_eshel_uchida_for_lfads_dataset_N0_factordim2_for_inference",
        # default="../../lfads/results/dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p1train_dataset_N0_factordim2_for_inference",
    )
    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../figures/lfads/dopamine_spiking_eshel_uchida_for_lfads_dataset_N0_factordim2_for_inference",
        # default="../figures/lfads/dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p1train_dataset_N0_factordim2_for_inference",
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

    # # ----------------------------------------------------------------#
    # # ----------------------------------------------------------------#
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
    # # ----------------------------------------------------------------#
    # # ----------------------------------------------------------------#

    data_dict = torch.load(
        f"../data/dopamine-spiking-eshel-uchida/lfads/train_data_dict_{filename_partial}.py"
    )

    spike_counts = data_dict["spike_counts"]
    neurons = data_dict["neurons"]
    labels = data_dict["labels"]
    factors = data_dict["factors"]
    factors_counts = data_dict["factors_counts"]

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

    n_comp = 2

    num_data = factors.shape[0]

    factors_proc = factors - np.mean(factors, axis=1, keepdims=True)
    factors_proc = factors_proc / np.linalg.norm(factors_proc, axis=1, keepdims=True)

    factors_stack = np.concatenate(
        [factors_proc[:, :, 0], factors_proc[:, :, 1]], axis=0
    )
    print("factors_stack", factors_stack.shape)

    if 0:
        pca_transform = PCA(n_components=n_comp)
        factors_pca_coeff = pca_transform.fit_transform(factors_stack)

        print(np.expand_dims(np.mean(avg_factors[:, 0], axis=0), axis=0).shape)
        factors_mean_pca_coeff_1 = pca_transform.transform(
            np.expand_dims(np.mean(factors_stack[:num_data], axis=0), axis=0)
        )
        factors_mean_pca_coeff_2 = pca_transform.transform(
            np.expand_dims(np.mean(factors_stack[num_data:], axis=0), axis=0)
        )

        pc = pca_transform.components_
        print("pc", pc.shape)

        print(factors_pca_coeff.shape)

        # plot new fig
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.scatter(
            factors_pca_coeff[:num_data, 0],
            factors_pca_coeff[:num_data, 1],
            s=0.5,
            color="black",
        )
        plt.scatter(
            factors_pca_coeff[num_data:, 0],
            factors_pca_coeff[num_data:, 1],
            s=0.5,
            color="red",
        )

        plt.scatter(
            factors_mean_pca_coeff_1[:, 0],
            factors_mean_pca_coeff_1[:, 1],
            s=0.5,
            color="blue",
        )
        plt.scatter(
            factors_mean_pca_coeff_2[:, 0],
            factors_mean_pca_coeff_2[:, 1],
            s=0.5,
            color="green",
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        ax.grid(False)

        fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
        plt.show()
        plt.savefig("../tmp.png")
        plt.close()

        # exit()

    avg_factors_proc = np.mean(factors_proc, axis=0)

    plot_kernel(
        -avg_factors_proc, params, out_path, name="lfads_zeromean_normalized_sign_flip"
    )
    plot_kernel(avg_factors_proc, params, out_path, name="lfads_zeromean_normalized")
    plot_kernel(-avg_factors, params, out_path, name="lfads_sign_flip")
    plot_kernel(avg_factors, params, out_path, name="lfads")

    # exit()

    num_neurons = max(neurons) + 1
    num_reward_types = 6

    for sign in [-1]:
        for method in ["mse", "mae", "sum"]:
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
                if method == "mse":
                    factor_counts_neuron = np.mean(factors**2, axis=1)[neuron_indices]
                elif method == "mae":
                    factor_counts_neuron = np.mean(np.abs(factors), axis=1)[
                        neuron_indices
                    ]
                elif method == "sum":
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

            dunl_res = torch.load(
                "../data/dopamine-spiking-eshel-uchida/lfads/eshel_spearman_for_lfads_comparison.pt"
            )
            dunl_corr_sur = dunl_res["code_sur_corr"]
            dunl_corr_exp = dunl_res["code_exp_corr"]
            print(dunl_corr_exp.shape)

            print("sur")
            print(np.mean(x_corr_sur[:, 0]), np.mean(x_corr_sur[:, 1]))
            print(torch.mean(dunl_corr_sur[:, 2]))

            print("exp")
            print(np.mean(x_corr_exp[:, 0]), np.mean(x_corr_exp[:, 1]))
            print(torch.mean(dunl_corr_exp[:, 2]))

            for factor_ctr in range(params["num_comp"]):
                plot_spearman_correlation(
                    dunl_corr_exp,
                    torch.tensor(x_corr_exp[:, factor_ctr]),
                    dunl_corr_sur,
                    torch.tensor(x_corr_sur[:, factor_ctr]),
                    params,
                    out_path_name=os.path.join(
                        out_path,
                        "lfads_spearman_correlation_factor{}_sign{}_{}.svg".format(
                            factor_ctr, sign, method
                        ),
                    ),
                    color_ctr=factor_ctr + 1,
                )

                plot_spearman_correlation_hist(
                    dunl_corr_exp[:, 2],
                    torch.tensor(x_corr_exp[:, factor_ctr]),
                    params,
                    out_path_name=os.path.join(
                        out_path,
                        "spearman_correlation_distance_exp_factor{}_sign{}_{}.svg".format(
                            factor_ctr, sign, method
                        ),
                    ),
                )
                plot_spearman_correlation_hist(
                    dunl_corr_sur[:, 2],
                    torch.tensor(x_corr_sur[:, factor_ctr]),
                    params,
                    out_path_name=os.path.join(
                        out_path,
                        "spearman_correlation_distance_sur_factor{}_sign{}_{}.svg".format(
                            factor_ctr, sign, method
                        ),
                    ),
                )

            print(f"plotting of lfads is done. plots are saved at {out_path}")


def plot_kernel(factors, params, out_path, name="lfads"):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    kernel_length = 600

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    # plot -------------------------------------------------------#
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(1.6, 2))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.axhline(0, color="gray", lw=0.3)

    t = np.linspace(
        0,
        kernel_length,
        kernel_length,
    )

    plt.subplot(1, 1, 1)
    for ctr in range(factors.shape[1]):
        plt.plot(t, factors[:, ctr], color=params["color_list"][ctr + 1])
        xtic = np.array([0, 0.5, 1]) * kernel_length
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)
        plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "kernels_{}.svg".format(name)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        os.path.join(out_path, "kernels_{}.png".format(name)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_spearman_correlation(
    code_exp, lfads_exp, code_sur, lfads_sur, params, out_path_name, color_ctr
):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    markersize = 4
    fontfamily = "sans-serif"

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")

    for code_ctr in range(1, code_sur.shape[1] - 1):
        plt.subplot(1, 2, code_ctr)
        plt.scatter(
            code_sur[:, code_ctr],
            lfads_sur,
            marker=".",
            color=params["color_list"][color_ctr],
            s=markersize,
        )
        plt.scatter(
            code_exp[:, code_ctr],
            lfads_exp,
            marker=".",
            color=params["color_list"][color_ctr],
            s=markersize,
        )
        plt.plot([-0.5, 1], [-0.5, 1], "--", color="gray", linewidth=1)

        if code_ctr == 1:
            axn[code_ctr - 1].set_ylabel("Spearman(LFADS, Reward)")
            axn[code_ctr].set_xlabel("Spearman(Code, Reward)")

        if code_ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        elif code_ctr == 2:
            plt.title(r"$\textbf{Reward\ II}$")

        plt.scatter(
            torch.mean(code_sur[:, code_ctr]),
            torch.mean(lfads_sur),
            marker="x",
            color="green",
            lw=1.5,
            s=markersize * 6,
        )
        print(
            "sur",
            "y",
            torch.mean(lfads_sur),
            "code",
            torch.mean(code_sur[:, code_ctr]),
            code_ctr,
        )
        plt.scatter(
            torch.mean(code_exp[:, code_ctr]),
            torch.mean(lfads_exp),
            marker="x",
            color="yellow",
            lw=1.5,
            s=markersize * 6,
        )
        print(
            "exp",
            "y",
            torch.mean(lfads_exp),
            "code",
            torch.mean(code_exp[:, code_ctr]),
            code_ctr,
        )
        xtic = np.array([0, 0.5, 1])
        xtic = [x for x in xtic]
        plt.xticks(xtic, xtic)
        plt.yticks(xtic, xtic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close()


def plot_spearman_correlation_hist(code_one_kernel, y_window, params, out_path_name):
    axes_fontsize = 20
    legend_fontsize = 8
    tick_fontsize = 20
    title_fontsize = 25
    markersize = 40
    fontfamily = "sans-serif"
    lw = 2

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    cmap = "YlGn"

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    distance_from_diag = utils.compute_signed_distance_from_diag(
        code_one_kernel,
        y_window,
    )

    hist_heatmap = []
    distance_mean_list = []

    a = np.max(np.abs(distance_from_diag)) * 1.1
    hist, bin_edges = np.histogram(
        distance_from_diag, bins=params["n_bin_spearman"], range=(-a, a), density=True
    )
    distance_mean = np.mean(distance_from_diag)

    plt.hist(distance_from_diag, bins=bin_edges)
    # y, x = np.meshgrid(
    #     params["window_start_list"],
    #     np.linspace(-a, a, params["n_bin_spearman"]),
    # )
    # c = ax.pcolormesh(x, y, hist_heatmap.T, cmap=cmap)
    # plt.plot(bin_edges[:-1], hist, color="black")

    ax.axvline(0, color="black", lw=lw)

    ax.set_xlabel("Distance to Diagonal from Right")
    ax.grid(False)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
