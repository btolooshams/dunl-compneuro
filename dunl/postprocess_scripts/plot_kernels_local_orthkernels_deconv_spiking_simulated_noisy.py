"""
Copyright (c) 2025 Bahareh Tolooshams

plot rec data kernel

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

sys.path.append("../dunl/")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-list",
        type=str,
        help="res path list",
        default=["../results"],
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
        ],  # 2 kernels
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 2),
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
    params_init = init_params()

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "01noise_kernels",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    epoch_type = "best_val"

    res_folder = "../results"
    filename_list = os.listdir(res_folder)
    filename_list = [f"{x}" for x in filename_list if "20unrolling" in x]
    filename_list = [f"{x}" for x in filename_list if "01noise" in x]
    res_path_list = [f"{res_folder}/{x}" for x in filename_list if "orth_" in x]

    data_folder = "../data/local-orthkernels-simulated"
    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

    for res_path in res_path_list:
        num_trials = int(res_path.split("_")[1].split("trials")[0])

        num_kernels = int(res_path.split("_")[2].split("kernel")[0])

        # take parameters from the result path
        params = pickle.load(open(os.path.join(res_path, "params.pickle"), "rb"))
        params["time_bin_resolution"] = 5
        for key in params_init.keys():
            params[key] = params_init[key]

        # model
        model_path = os.path.join(
            res_path,
            "model",
            f"model_{epoch_type}.pt",
        )
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()
        kernels = net.get_param("H").clone().detach()
        kernels = np.squeeze(kernels.cpu().numpy())

        outname = res_path.split("/")[-1]
        plot_kernel_est(
            kernels,
            params,
            out_path,
            f"{outname}_onlyest.png",
        )
        plot_kernel_est(
            kernels,
            params,
            out_path,
            f"{outname}_onlyest.svg",
        )


def plot_kernel_est(kernels, params, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
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

    row = 1
    col = kernels.shape[0]
    fig, axn = plt.subplots(
        row, col, sharex=True, sharey=True, figsize=params["figsize"]
    )

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0,
        kernels.shape[-1] * params["time_bin_resolution"],
        kernels.shape[-1],
    )

    for ctr in range(col):
        plt.subplot(row, col, ctr + 1)

        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])

        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)

        plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
