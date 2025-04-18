"""
Copyright (c) 2025 Bahareh Tolooshams

plot experimental setup

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt



def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
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
        "--regret-dur",
        type=int,
        help="regret duration after onset in samples",
        default=60,
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=15,
    )
    parser.add_argument(
        "--reward-delay",
        type=int,
        help="reward delay from the cue onset",
        default=45,
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="duration",
        default=90,
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
    fig, axn = plt.subplots(3, 1, sharex=True, sharey=True)

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

    plt.subplot(3, 1, 1)
    plt.title(r"$\textbf{Regret\ Trials}$")
    plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
    ctr = -1
    plt.plot(
        cue_regret_dot,
        dot_loc,
        ".",
        markersize=10,
        color="Orange",
    )

    plt.subplot(3, 1, 2)
    plt.title(r"$\textbf{Surprise\ Trials}$")
    plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
    plt.plot(
        reward_surprise_dot,
        dot_loc,
        ".",
        markersize=10,
        color="Blue",
    )

    plt.subplot(3, 1, 3)
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
    plt.xticks(xtic, xtic / params["sampling_rate"])
    plt.xlabel("Time [s]", labelpad=0)

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
