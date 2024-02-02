"""
Copyright (c) 2020 Bahareh Tolooshams

plot experimental setup

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_14_12_37_30",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers for dataloader",
        default=4,
    )
    parser.add_argument(
        "--reward-delay",
        type=int,
        help="reward delay from the cue onset",
        default=60,  # this is after the bining
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="duration",
        default=120,  # this is after the bining
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

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # plot configuration -------------------------------------------------------#

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
    fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    dot_loc = 1

    cue_regret_dot = 0
    cue_expected_dot = 0
    reward_expected_dot = params["reward_delay"]
    reward_surprise_dot = 0

    plt.subplot(2, 1, 1)
    plt.title(r"$\textbf{Surprise\ Trials}$")
    plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
    plt.plot(
        reward_surprise_dot,
        dot_loc,
        ".",
        markersize=10,
        color="Blue",
    )

    plt.subplot(2, 1, 2)
    plt.title(r"$\textbf{Expected\ Trials}$")
    plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
    plt.axvline(x=params["reward_delay"], linestyle="--", linewidth=0.5, color="black")
    plt.plot(
        cue_expected_dot,
        dot_loc,
        ".",
        markersize=10,
        color="Orange",
    )
    plt.plot(
        reward_expected_dot,
        dot_loc,
        ".",
        markersize=10,
        color="Blue",
    )
    xtic = np.array([0, 0.5, 1]) * params["duration"]
    xtic_figure = [int(x * params["time_bin_resolution"]) for x in xtic]
    plt.xticks(xtic, xtic_figure)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "experiment_setup.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"plotting of experimental setup is done. plots are saved at {out_path}")


if __name__ == "__main__":
    main()
