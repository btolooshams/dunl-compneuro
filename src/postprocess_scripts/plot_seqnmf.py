"""
Copyright (c) 2020 Bahareh Tolooshams

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
import scipy.io

import sys

sys.path.append("../src/")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/seqnmf"
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "blue",
            "red",
        ],  # 2 kernels
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
    params_init = init_params()

    res_path = params_init["res_path"]

    H = scipy.io.loadmat(os.path.join(res_path, "H_numtrials_5.mat"))["H"]
    W = scipy.io.loadmat(os.path.join(res_path, "W_numtrials_5.mat"))["W"]
    X = scipy.io.loadmat(os.path.join(res_path, "X_numtrials_5.mat"))["X"]

    kernels_true = np.load("../data/local-2kernelfornmf-simulated/kernels.npy")

    kernels = np.mean(W, axis=0)[:-1]
    kernels = kernels / (np.linalg.norm(kernels, axis=-1, keepdims=True) + 1e-8)

    kernels = np.nan_to_num(kernels)

    # take parameters from the result path
    dunl_path = "../results/2kernelfornmf_30trials_3kernel_num_20unrolling_2024_08_22_20_26_11"
    params = pickle.load(
        open(os.path.join(dunl_path, "params.pickle"), "rb")
    )
    params["time_bin_resolution"] = 5
    for key in params_init.keys():
        params[key] = params_init[key]
    
    # model
    model_path = os.path.join(
        dunl_path,
        "model",
        f"model_best_val.pt",
    )
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()
    kernels_dunl = net.get_param("H").clone().detach()
    kernels_dunl = np.squeeze(kernels_dunl.cpu().numpy())
    kernels_dunl = kernels_dunl[[2,1]]

    print(kernels_true.shape, kernels.shape, kernels_dunl.shape)

    plot_kernel_one_plot(
        kernels,
        kernels_true,
        params,
        res_path,
        name="seqnmf"
    )

    plot_kernel_one_plot(
        kernels_dunl,
        kernels_true,
        params,
        res_path,
        name="dunl"
    )

    print(X.shape, H.shape)

    x_one_trial = np.mean(X[:, :3000], axis=0)
    code_one_trial = H[0, :3000]

    plot_data_and_code(x_one_trial, code_one_trial, params, res_path)

def plot_data_and_code(
    x, 
    code,
    params,
    out_path,
):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

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

    fig, axn = plt.subplots(2, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(2,1,1)
    plt.plot(
        np.linspace(0, x.shape[0]-1, x.shape[0]) * 5,
        x,
        lw=0.7,
        color="black",
        label="data",
    )
    plt.ylabel("Average data across neurons")
    plt.subplot(2,1,2)
    plt.plot(
        np.linspace(0, code.shape[0]-1, code.shape[0]) * 5,
        code,
        lw=1,
        color="blue",
        label="code",
    )
    plt.ylabel("SeqNMF Code")

    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, f"h__seqnmf.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        os.path.join(out_path, f"h_seqnmf.png"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()



def plot_kernel_one_plot(
    kernels, kernels_true, params, out_path, name="",
):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 15
    fontfamily = "sans-serif"
    lw_true = 1

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

    fig, ax = plt.subplots(
        1,
        1,
        sharex=True,
        sharey=True,
        figsize=(params["figsize"][-1], params["figsize"][-1]),
    )

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    t = np.linspace(
        0,
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    ax.axhline(0, color="gray", lw=0.3)
    for ctr in range(2):
        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])
        if ctr == 0:
            label = "True"
        else:
            label = None
        plt.plot(t, kernels_true[ctr], color="gray", label=label, linewidth=lw_true)

    xtic = (
        np.array([0, 0.5, 1]) * params["kernel_length"] * params["time_bin_resolution"]
    )
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    ax.set_yticks([])

    plt.legend()
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path, f"kernels_{name}.png"
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        os.path.join(
            out_path, f"kernels_{name}.svg"
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
