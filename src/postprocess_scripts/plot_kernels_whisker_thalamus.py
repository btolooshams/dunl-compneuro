"""
Copyright (c) 2020 Bahareh Tolooshams

plot kernel whisker data

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

sys.path.append("../src/")


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/whisker_05msbinres_lamp03_topk18_smoothkernelp003_2023_07_19_00_03_18",
        # default="../results//whisker_05msbinres_lamp03_topk16_smoothkernelp003_2023_07_20_23_11_21",
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "cyan",
        ],  # learning one kernel
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

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]
    # create folders -------------------------------------------------------#
    model_path = os.path.join(
        params["res_path"],
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

    kernels = np.squeeze(net.get_param("H").clone().detach().cpu().numpy(), axis=1)
    t = np.linspace(
        0,
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

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
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=params["figsize"])

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for ctr in range(params["kernel_num"]):
        plt.subplot(1, 1, ctr + 1)
        ax.axhline(0, color="gray", lw=0.3)

        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])

        print(t)
        stim = np.sin(2 * np.pi * (1 / 125 * t))
        stim /= np.linalg.norm(stim)
        plt.plot(t, stim, color="gray", lw=0.5)
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
        os.path.join(out_path, "kernels.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"plotting of kernels is done. plots are saved at {out_path}")


if __name__ == "__main__":
    main()
