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

import sys

sys.path.append("../src/")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-list",
        type=str,
        help="res path list",
        default=[
            "../results/simulated_dopaminespiking_40neurons_14trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_23_42_31",
            "../results/simulated_dopaminespiking_40neurons_25trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_23_30_02",
            "../results/simulated_dopaminespiking_40neurons_50trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_22_02_49",
            "../results/simulated_dopaminespiking_40neurons_100trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_44_03",
            "../results/simulated_dopaminespiking_40neurons_200trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_18_24",
            "../results/simulated_dopaminespiking_40neurons_300trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_19_37",
            "../results/simulated_dopaminespiking_40neurons_400trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_48_04",
        ],
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "orange",
            "blue",
            "red",
        ],  # cue exp, 2 rewards
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(6, 2),
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

    kernel_err_list = list()
    num_trials_list = list()

    for res_path in params_init["res_path_list"]:
        # take parameters from the result path
        params = pickle.load(open(os.path.join(res_path, "params.pickle"), "rb"))
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

        # set time bin resolution -----------------------------------------------#
        data_dict = torch.load(data_path_list[0])
        params["time_bin_resolution"] = data_dict["time_bin_resolution"]

        # create folders -------------------------------------------------------#
        model_path = os.path.join(
            res_path,
            "model",
            "model_final.pt",
        )

        out_path = os.path.join(
            res_path,
            "figures",
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # load true kernel  ------------------------------------------------#
        data_folder = data_path_list[0].split(data_path_list[0].split("/")[-1])[0]

        kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

        num_trials = int(data_path_list[0].split("nuerons_")[-1].split("trials")[0])
        num_trials_list.append(num_trials)

        # load model ------------------------------------------------------#
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()

        kernel_err = 1.0
        for swap_kernel in [True, False]:
            kernels = net.get_param("H").clone().detach()

            if swap_kernel:
                kernels = utils.swap_kernel(kernels, 1, 2)
            kernels = np.squeeze(kernels.cpu().numpy())

            plot_kernel(kernels, kernels_true, params, out_path, swap_kernel)
            plot_kernel_one_plot(kernels, kernels_true, params, out_path, swap_kernel)

            # compute kernel error
            kernel_err_tmp = utils.compute_dictionary_error(kernels_true, kernels)

            kernel_err = np.minimum(kernel_err, np.mean(kernel_err_tmp))

        kernel_err_list.append(kernel_err)

    plot_kernelerr_vs_numtrials(kernel_err_list, num_trials_list, params, out_path)


def plot_kernelerr_vs_numtrials(kernel_err, num_trials, params, out_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    fontfamily = "sans-serif"
    lw_true = 1
    color = "black"

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

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 2))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.plot(num_trials, kernel_err, color=color)
    plt.xlabel("Number of Trials")
    plt.ylabel("Kernel Error")

    xtic = np.array([num_trials[0], num_trials[3], num_trials[-2], num_trials[-1]])
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)

    ytic = np.array([0, 0.25, 0.5, 0.75, 1])
    ytic = [x for x in ytic]
    plt.yticks(ytic, ytic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "kernelerr_vs_numtrials.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print("plotting done!")


def plot_kernel(kernels, kernels_true, params, out_path, swap_kernel):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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

    fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0,
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    for ctr in range(params["kernel_num"]):
        plt.subplot(1, 3, ctr + 1)
        axn[ctr].axhline(0, color="gray", lw=0.3)

        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])

        plt.plot(t, kernels_true[ctr], color="gray", label="True", linewidth=lw_true)

        if ctr == 0:
            plt.title(r"$\textbf{Cue}$")
        elif ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        else:
            plt.title(r"$\textbf{Reward\ II}$")
        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)

        if ctr == 0:
            plt.legend()
        if ctr == 1:
            plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "kernels_swap{}.svg".format(swap_kernel)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_kernel_one_plot(kernels, kernels_true, params, out_path, swap_kernel):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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
    for ctr in range(params["kernel_num"]):
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
        os.path.join(out_path, "kernels_swap{}_oneplot.svg".format(swap_kernel)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
