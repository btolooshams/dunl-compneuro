"""
Copyright (c) 2020 Bahareh Tolooshams

plot rec data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


import sys

sys.path.append("../src/")

import model


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/xxx",
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
            "black",
            "orange",
            "blue",
            "red",
            "green",
        ],  # cue reg, cue exp, 3 rewards
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 2),
    )
    parser.add_argument(
        "--swap-kernel-list", type=bool, help="bool to swap kernels", default=[]
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

    neuron_file_list = os.listdir(params_init["res_path"])
    neuron_file_list = [x for x in neuron_file_list if "figure" not in x]
    neuron_file_list = [x for x in neuron_file_list if ".DS_Store" not in x]

    kernels_list = list()
    for neuron_file in neuron_file_list:
        neuron_folder = os.path.join(params_init["res_path"], neuron_file)

        # take parameters from the result path
        params = pickle.load(open(os.path.join(neuron_folder, "params.pickle"), "rb"))
        for key in params_init.keys():
            params[key] = params_init[key]

        # create folders -------------------------------------------------------#
        model_path = os.path.join(
            neuron_folder,
            "model",
            "model_final.pt",
        )

        out_path = os.path.join(
            params["res_path"],
            "figures",
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # load model ------------------------------------------------------#
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()

        kernels = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())

        name_path = os.path.join(out_path, f"kernel_{neuron_file}.svg")
        plot_kernel_individual(kernels, params, name_path)

        for k in range(kernels.shape[0]):
            if np.mean(kernels[k]) < 0:
                kernels[k] *= -1

        for k in range(kernels.shape[0]):
            if np.mean(kernels[k][:30]) < 0:
                kernels[k] *= -1

        kernels_list.append(kernels)

    # (neurons, cue/rewardI/rewardII, time length
    kernels_list = np.stack(kernels_list, axis=0)

    name_path = os.path.join(out_path, f"0_kernel_all.svg")
    plot_kernel_all(kernels_list, params, name_path)


def plot_kernel_individual(kernels, params, out_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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

    fig, axn = plt.subplots(1, 4, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0, params["kernel_length"] / params["sampling_rate"], params["kernel_length"]
    )

    for ctr in range(params["kernel_num"] - 1):
        plt.subplot(1, 4, ctr + 1)
        axn[ctr].axhline(0, color="gray", lw=0.3)

        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])
        if ctr == 3:
            plt.plot(t, kernels[ctr + 1], color=params["color_list"][ctr + 1])

        if ctr == 0:
            plt.title(r"$\textbf{Cue\ Regret}$")
        elif ctr == 1:
            plt.title(r"$\textbf{Cue\ Expected}$")
        elif ctr == 2:
            plt.title(r"$\textbf{Reward}$")
        else:
            plt.title(r"$\textbf{Reward\ Coupled}$")
        xtic = np.array([0, 0.5, 1]) * params["kernel_length"] / params["sampling_rate"]
        plt.xticks(xtic, xtic)

        if ctr == 1:
            plt.xlabel("Time [s]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"plotting of kernels is done. plots are saved at {out_path}")


def plot_kernel_all(kernels_list, params, out_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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

    fig, axn = plt.subplots(1, 4, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0, params["kernel_length"] / params["sampling_rate"], params["kernel_length"]
    )

    for ctr in range(params["kernel_num"] - 1):
        plt.subplot(1, 4, ctr + 1)
        axn[ctr].axhline(0, color="gray", lw=0.3)

        for i in range(kernels_list.shape[0]):
            plt.plot(t, kernels_list[i, ctr], color="gray", linewidth=0.1)
        plt.plot(
            t, np.mean(kernels_list[:, ctr], axis=0), color=params["color_list"][ctr]
        )
        if ctr == 3:
            for i in range(kernels_list.shape[0]):
                plt.plot(t, kernels_list[i, ctr + 1], color="gray", linewidth=0.1)
            plt.plot(
                t,
                np.mean(kernels_list[:, ctr + 1], axis=0),
                color=params["color_list"][ctr + 1],
            )

        if ctr == 0:
            plt.title(r"$\textbf{Cue\ Regret}$")
        elif ctr == 1:
            plt.title(r"$\textbf{Cue\ Expected}$")
        elif ctr == 2:
            plt.title(r"$\textbf{Reward}$")
        else:
            plt.title(r"$\textbf{Reward\ Coupled}$")
        xtic = np.array([0, 0.5, 1]) * params["kernel_length"] / params["sampling_rate"]
        plt.xticks(xtic, xtic)

        if ctr == 1:
            plt.xlabel("Time [s]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
