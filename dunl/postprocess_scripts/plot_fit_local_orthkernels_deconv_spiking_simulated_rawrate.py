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

import utils, datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-list",
        type=str,
        help="res path list",
        default=["../results"],
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

    data_folder = "../data/local-orthkernels-simulated"

    ###################################
    filename_list = os.listdir(data_folder)
    filename_list = [f"{x}" for x in filename_list if "testvis" in x]
    data_path_list_vis = [f"{data_folder}/{x}" for x in filename_list if ".pt" in x]

    # take parameters from the result path
    params = pickle.load(open(os.path.join(res_path_list[0], "params.pickle"), "rb"))
    params["time_bin_resolution"] = 5
    for key in params_init.keys():
        params[key] = params_init[key]

    out_path_raster = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "raster",
    )
    if not os.path.exists(out_path_raster):
        os.makedirs(out_path_raster)

    test_dataset = datasetloader.DUNLdatasetwithRaster(data_path_list_vis[0])
    test_raster = test_dataset.raster
    test_y = test_dataset.y
    test_rate = test_dataset.rate

    ###################################
    out_path_rec = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "rec",
    )
    if not os.path.exists(out_path_rec):
        os.makedirs(out_path_rec)

    csv_path = os.path.join("../", "figures", "orthkernels", "csv")
    csv_filename_list = os.listdir(csv_path)
    csv_filename_list = [f"{csv_path}/{x}" for x in csv_filename_list if ".csv" in x]

    kernels_list = [3, 4, 5, 6]

    for res_path in res_path_list:
        if (
            "orth_500trials_5kernel_num_20unrolling_1000epochs_2024_08_12_11_02_48"
            not in res_path
        ):
            continue

        num_trials = int(res_path.split("_")[1].split("trials")[0])

        num_kernels = int(res_path.split("_")[2].split("kernel")[0])

        # model
        model_path = os.path.join(
            res_path,
            "model",
            f"model_{epoch_type}.pt",
        )
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()

        outname = res_path.split("/")[-1]

        if num_kernels not in kernels_list:
            continue

        for i in range(9):
            y = test_dataset.y[[i]]
            a = test_dataset.a[[i]]
            rate = test_dataset.rate[[i]]
            y_in = y.unsqueeze(dim=2)
            a_in = a.unsqueeze(dim=2)

            y_in = y_in.to(device)
            a_in = a_in.to(device)

            print(y_in.shape, a_in.shape, rate.shape)

            print(a_in)
            # forward encoder
            xhat_out, a_est = net.encode(y_in, a_in, None)
            for k in range(xhat_out.shape[2]):
                xhat_out_0 = xhat_out.clone()
                xhat_out_0[:, :, 1:] = 0

                xhat_out_1 = xhat_out.clone()
                xhat_out_1[:, :, :1] = 0
                xhat_out_1[:, :, 2:] = 0

                xhat_out_2 = xhat_out.clone()
                xhat_out_2[:, :, :2] = 0
                xhat_out_2[:, :, 3:] = 0

                xhat_out_3 = xhat_out.clone()
                xhat_out_3[:, :, :3] = 0
                xhat_out_3[:, :, 4:] = 0

                xhat_out_4 = xhat_out.clone()
                xhat_out_4[:, :, :4] = 0

            # forward decoder
            hxmu = net.decode(xhat_out, a_est)
            hxmu_0 = net.decode(xhat_out_0, a_est).detach().cpu().squeeze(dim=2)
            hxmu_1 = net.decode(xhat_out_1, a_est).detach().cpu().squeeze(dim=2)
            hxmu_2 = net.decode(xhat_out_2, a_est).detach().cpu().squeeze(dim=2)
            hxmu_3 = net.decode(xhat_out_3, a_est).detach().cpu().squeeze(dim=2)
            hxmu_4 = net.decode(xhat_out_4, a_est).detach().cpu().squeeze(dim=2)

            if params["model_distribution"] == "binomial":
                yhat = torch.sigmoid(hxmu)
            else:
                raise NotImplementedError("model distribution is not implemented")

            yhat = torch.squeeze(yhat, dim=2).detach().cpu()

            print("num_trials", num_trials, "num_kernels", num_kernels)

            neuron = 5
            plot_orthkernels_singleneuron_rateest(
                rate[0, neuron],
                yhat[0, neuron],
                os.path.join(out_path_rec, f"orthkernels_rate_est_{i}.png"),
            )
            plot_orthkernels_singleneuron_rateest(
                rate[0, neuron],
                yhat[0, neuron],
                os.path.join(out_path_rec, f"orthkernels_rate_est_{i}.svg"),
            )

            plot_orthkernels_singleneuron_decomposition(
                hxmu_0[0, neuron],
                hxmu_1[0, neuron],
                hxmu_2[0, neuron],
                hxmu_3[0, neuron],
                hxmu_4[0, neuron],
                os.path.join(out_path_rec, f"orthkernels_dec_{i}.png"),
            )
            plot_orthkernels_singleneuron_decomposition(
                hxmu_0[0, neuron],
                hxmu_1[0, neuron],
                hxmu_2[0, neuron],
                hxmu_3[0, neuron],
                hxmu_4[0, neuron],
                os.path.join(out_path_rec, f"orthkernels_dec_{i}.svg"),
            )

        break


def plot_orthkernels_neuron_time(raster, plot_filename):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

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
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(111)
    plt.imshow(-raster, aspect="auto", cmap="gray")
    plt.xlabel("Time")
    plt.ylabel("Neurons", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_orthkernels_singleneuron_decomposition(
    hxmu_0, hxmu_1, hxmu_2, hxmu_3, hxmu_4, plot_filename
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
            "text.usetex": False,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    row = 5
    col = 1
    fig, axn = plt.subplots(5, 1, sharex=True, sharey=True)
    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for ctr in range(row):
        plt.subplot(row, col, ctr + 1)
        that = np.linspace(0, hxmu_0.shape[-1] - 1, hxmu_0.shape[-1])
        if ctr == 0:
            plt.plot(that * 5, hxmu_0, color="blue")
        if ctr == 1:
            plt.plot(that * 5, hxmu_1, color="blue")
        if ctr == 2:
            plt.plot(that * 5, hxmu_2, color="blue")
        if ctr == 3:
            plt.plot(that * 5, hxmu_3, color="blue")
        if ctr == 4:
            plt.plot(that * 5, hxmu_4, color="blue")
        if ctr == row - 1:
            plt.xlabel("Time")

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_orthkernels_singleneuron_rateest(rate, rate_hat, plot_filename):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

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
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(111)
    t = np.linspace(0, rate.shape[-1] - 1, rate.shape[-1])
    that = np.linspace(0, rate_hat.shape[-1] - 1, rate_hat.shape[-1])
    plt.plot(t, rate, color="gray", label="Ground Truth")
    plt.plot(that * 5, rate_hat, color="blue", label="Est")
    plt.legend()
    plt.ylabel("Rate")
    plt.xlabel("Time")

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
