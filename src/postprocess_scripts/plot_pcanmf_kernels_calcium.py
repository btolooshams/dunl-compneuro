"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy as sp
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
    )
    parser.add_argument(
        "--num-comp",
        type=int,
        help="number of components",
        default=2,
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=15,
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "blue",
            "red",
            "green",
        ],
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(1.6, 2),
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

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # load data -------------------------------------------------------#
    pca_transform = pickle.load(
        open(
            os.path.join(
                postprocess_path, "pca_transform_{}.pkl".format(params["num_comp"])
            ),
            "rb",
        )
    )
    nmf_transform = pickle.load(
        open(
            os.path.join(
                postprocess_path, "nmf_transform_{}.pkl".format(params["num_comp"])
            ),
            "rb",
        )
    )

    pca_W = pca_transform.components_.T
    nmf_W = nmf_transform.components_.T

    plot_pca_nmf(pca_W, params, out_path, "pca_{}".format(params["num_comp"]))
    plot_pca_nmf(nmf_W, params, out_path, "nmf_{}".format(params["num_comp"]))

    print(f"plotting of kernels is done. plots are saved at {out_path}")


def plot_pca_nmf(W, params, out_path, name):
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

    # plot -------------------------------------------------------#
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=params["figsize"])

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.axhline(0, color="gray", lw=0.3)

    t = np.linspace(
        0,
        params["kernel_length"] * params["sampling_rate"],
        params["kernel_length"],
    )

    plt.subplot(1, 1, 1)
    for ctr in range(W.shape[1]):
        plt.plot(t, W[:, ctr], color=params["color_list"][ctr])
        xtic = np.array([0, 0.5, 1]) * params["kernel_length"] * params["sampling_rate"]
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)
        plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "kernels_{}.svg".format(name)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
