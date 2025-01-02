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

    epoch_type = "best_val"

    res_folder = "../results"
    filename_list = os.listdir(res_folder)
    filename_list = [f"{x}" for x in filename_list if "20unrolling" in x]
    res_path_list = [f"{res_folder}/{x}" for x in filename_list if "orth_" in x]

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data_folder = "../data/local-orthkernels-simulated"
    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

    for res_path in res_path_list:
        num_trials = int(res_path.split("_")[1].split("trials")[0])

        num_kernels = int(res_path.split("_")[2].split("kernel")[0])

        if num_trials != 100:
            continue

        # take parameters from the result path
        params = pickle.load(open(os.path.join(res_path, "params.pickle"), "rb"))
        params["time_bin_resolution"] = 5
        for key in params_init.keys():
            params[key] = params_init[key]

        if res_path == res_path_list[0]:
            plot_kernel_true(
                kernels_true,
                params,
                out_path,
            )

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
        plot_kernel(
            kernels,
            kernels_true,
            params,
            out_path,
            f"{outname}.svg",
        )
        plot_kernel_est(
            kernels,
            params,
            out_path,
            f"{outname}_onlyest.svg",
        )

    for err_type in ["cross_corr", "corr"]:
        for errmode in ["mean", "max"]:
            kernel_err_list = list()
            num_trials_list = list()

            for res_path in params_init["res_path_list"]:
                # take parameters from the result path
                params = pickle.load(
                    open(os.path.join(res_path, "params.pickle"), "rb")
                )
                for key in params_init.keys():
                    params[key] = params_init[key]

                if params["data_path"] == "":
                    data_folder = params["data_folder"]
                    filename_list = os.listdir(data_folder)
                    data_path_list = [
                        f"{data_folder}/{x}"
                        for x in filename_list
                        if "trainready.pt" in x
                    ]
                else:
                    data_path_list = params["data_path"]

                # set time bin resolution -----------------------------------------------#
                data_dict = torch.load(data_path_list[0])
                params["time_bin_resolution"] = data_dict["time_bin_resolution"]

                # create folders -------------------------------------------------------#
                for epoch_type in ["best_val"]:
                    model_path = os.path.join(
                        res_path,
                        "model",
                        f"model_{epoch_type}.pt",
                    )

                    out_path = os.path.join(
                        res_path,
                        "figures",
                    )
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # load true kernel  ------------------------------------------------#
                    data_folder = data_path_list[0].split(
                        data_path_list[0].split("/")[-1]
                    )[0]

                    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

                    num_trials = int(
                        data_path_list[0].split("neurons_")[-1].split("trials")[0]
                    )

                    num_trials_list.append(num_trials)

                    # load model ------------------------------------------------------#
                    net = torch.load(model_path, map_location=device)
                    net.to(device)
                    net.eval()

                    kernel_err = 1.0
                    for swap_kernel in [True, False]:
                        kernels = net.get_param("H").clone().detach()

                        if swap_kernel:
                            kernels = utils.swap_kernel(kernels, 0, 1)
                        kernels = np.squeeze(kernels.cpu().numpy())

                        plot_kernel(
                            kernels,
                            kernels_true,
                            params,
                            out_path,
                            swap_kernel,
                            epoch_type,
                        )
                        plot_kernel_one_plot(
                            kernels,
                            kernels_true,
                            params,
                            out_path,
                            swap_kernel,
                            epoch_type,
                        )

                        # compute kernel error
                        if err_type == "cross_corr":
                            kernel_err_tmp = (
                                utils.compute_dictionary_error_with_cross_correlation(
                                    kernels_true, kernels
                                )
                            )
                        else:
                            kernel_err_tmp = utils.compute_dictionary_error(
                                kernels_true, kernels
                            )

                        if errmode == "mean":
                            kernel_err = np.minimum(kernel_err, np.mean(kernel_err_tmp))
                        elif errmode == "max":
                            kernel_err = np.minimum(kernel_err, np.max(kernel_err_tmp))

                    kernel_err_list.append(kernel_err)

            print("num_trials_list", num_trials_list)
            print("kernel_err_list", kernel_err_list)
            plot_kernelerr_vs_numtrials(
                kernel_err_list,
                num_trials_list,
                params,
                out_path,
                f"{errmode}_{err_type}",
            )


def plot_kernelerr_vs_numtrials(kernel_err, num_trials, params, out_path, errmode):
    num_trials_unique = np.unique(num_trials)
    kernel_err_avg = list()
    kernel_err_std = list()
    for trial in num_trials_unique:
        indices = list(np.where(num_trials == trial)[0])
        kernel_err_avg.append(np.mean(np.array(kernel_err)[indices]))
        kernel_err_std.append(np.std(np.array(kernel_err)[indices]))

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

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 2))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.errorbar(num_trials_unique, kernel_err_avg, yerr=kernel_err_std, color=color)
    plt.xlabel("Number of Trials")
    plt.ylabel("Kernel Error")

    xtic = np.array(
        [
            num_trials_unique[0],
            num_trials_unique[-3],
            num_trials_unique[-2],
            num_trials_unique[-1],
        ]
    )
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)

    ytic = np.array([0, 0.25, 0.5, 0.75, 1])
    ytic = [x for x in ytic]
    plt.yticks(ytic, ytic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, f"kernelerr_vs_numtrials_errmode{errmode}.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print("plotting done!")


def plot_kernel(kernels, kernels_true, params, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"
    lw_true = 1

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
    col = np.maximum(kernels.shape[0], kernels_true.shape[0])
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
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    for ctr in range(col):
        plt.subplot(row, col, ctr + 1)

        if ctr < kernels_true.shape[0]:
            plt.plot(
                t, kernels_true[ctr], color="gray", label="True", linewidth=lw_true
            )

            cor = list()
            for i in range(kernels.shape[0]):
                cor.append(np.correlate(kernels_true[ctr], kernels[i]))

            best_kernel_index = np.argmax(np.abs(cor))

        if ctr < kernels.shape[0]:
            plt.plot(t, kernels[best_kernel_index], color=params["color_list"][ctr])

        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)

        if ctr == 0:
            plt.legend()

        plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_kernel_est(kernels, params, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"
    lw_true = 1

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
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
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


def plot_kernel_true(kernels_true, params, out_path):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"
    lw_true = 2

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
    col = kernels_true.shape[0]
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
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    for ctr in range(col):
        plt.subplot(row, col, ctr + 1)

        plt.plot(
            t, kernels_true[ctr], color="black", label="Ground Truth", linewidth=lw_true
        )

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
        os.path.join(out_path, "kernel_true.png"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.savefig(
        os.path.join(out_path, "kernel_true.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_kernel_one_plot(
    kernels, kernels_true, params, out_path, swap_kernel, epoch_type
):
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
        if ctr < kernels_true.shape[0]:
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
            out_path, "kernels_swap{}_{}_oneplot.svg".format(swap_kernel, epoch_type)
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
