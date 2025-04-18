"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import torch.nn.functional as F
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

sys.path.append("../dunl/")


import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        # default="../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35",
        default="../results/2000_12sparse_local_deconv_calscenario_shorttrial_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse2period10_kernelsmooth0.015_knownsuppFalse_2023_10_28_00_14_22",
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
        "--onset-color-list",
        type=list,
        help="onset color list",
        default=["Blue", "Red"],
    )
    parser.add_argument(
        "--raster-color",
        type=str,
        help="raster color",
        default="Black",
    )
    parser.add_argument(
        "--raster-markersize",
        type=int,
        help="raster markersize",
        default=0.2,
    )
    parser.add_argument(
        "--raw-color",
        type=str,
        help="raw color",
        default="Black",
    )
    parser.add_argument(
        "--smoothing-tau",
        type=int,
        help="smoothing tau",
        default=20,
    )
    parser.add_argument(
        "--num-trials-to-plot",
        type=int,
        help="num trials to plot",
        default=50,
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

    # create datasets -------------------------------------------------------#
    dataset = datasetloader.DUNLdatasetwithRaster(params["data_path"][0])

    data_folder = params["data_path"][0].split(data_path_list[0].split("/")[-1])[0]
    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))
    kernels_true = torch.tensor(np.expand_dims(kernels_true, axis=1))

    # create folders -------------------------------------------------------#
    out_path = os.path.join(params["res_path"], "figures", "raster_code")
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

    # go over data -------------------------------------------------------#

    num_neurons = dataset.raster.shape[1]

    for neuron_ctr in range(num_neurons):
        print(f"plot for neuron {neuron_ctr}")

        train_raster_neuron = dataset.raster[:, [neuron_ctr]]
        train_x = dataset.x
        train_a = dataset.a
        train_codes = dataset.codes
        train_rate = dataset.rate

        train_smooth_raster = utils.smooth_raster(
            train_raster_neuron, params["smoothing_tau"]
        )

        neuron_out_path = os.path.join(out_path, "neuron{}".format(neuron_ctr))
        if not os.path.exists(neuron_out_path):
            os.makedirs(neuron_out_path)

        # train_num_trials = len(dataset)

        plot_raster_smooth_code_several(
            train_x[: params["num_trials_to_plot"]],
            train_raster_neuron[: params["num_trials_to_plot"], neuron_ctr],
            train_smooth_raster[: params["num_trials_to_plot"], 0],
            params,
            os.path.join(
                neuron_out_path,
                "raster_smooth_code_train_neuron{}_t_smoothingtau{}.svg".format(
                    neuron_ctr,
                    params["smoothing_tau"],
                ),
            ),
        )

        for trial in range(10):
            plot_raster_smooth_code_one(
                train_x[trial],
                train_raster_neuron[trial],
                train_smooth_raster[trial, 0],
                train_a[trial],
                kernels_true,
                params,
                os.path.join(
                    neuron_out_path,
                    "raster_smooth_code_train_neuron{}_trial{}_smoothingtau{}.svg".format(
                        neuron_ctr,
                        trial,
                        params["smoothing_tau"],
                    ),
                ),
            )

    print(f"plotting of rec is done. plots are saved at {out_path}")


def plot_raster_smooth_code_several(
    x,
    raster,
    smooth_raster,
    params,
    plot_filename,
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

    print("x", x.shape)
    print("raster", raster.shape)
    print("smooth_raster", smooth_raster.shape)

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]
    t_raster = np.arange(0, smooth_raster.shape[-1])

    fig, axn = plt.subplots(3, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(3, 1, 1)
    raster_y, raster_x = np.where(raster != 0)
    plt.scatter(
        raster_x,
        raster_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.ylabel(r"$\textbf{Trials}$")
    plt.title(r"$\textbf{Rasters}$")

    plt.subplot(3, 1, 2)
    plt.imshow(
        smooth_raster,
        aspect="auto",
    )
    # plt.colorbar()
    plt.ylabel(r"$\textbf{Trials}$")
    plt.title(r"$\textbf{Smoothed Raster}$")

    plt.subplot(3, 1, 3)

    for conv in range(x.shape[1]):
        x_conv = x[:, conv]
        code_y, code_x = np.where(x_conv != 0)

        plt.scatter(
            code_x * params["time_bin_resolution"],
            code_y,
            s=params["raster_markersize"],
            color=params["onset_color_list"][conv],
            label=f"Onset {conv+1}",
        )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )
    plt.ylabel(r"$\textbf{Trials}$")

    xtic = [0, 500, 1500, 2000]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{Code}$")
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_raster_smooth_code_one(
    x,
    raster,
    smooth_raster,
    a,
    kernels,
    params,
    plot_filename,
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

    x = torch.squeeze(x, dim=0)

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]
    t_raster = np.arange(0, smooth_raster.shape[-1])

    fig, axn = plt.subplots(4, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(4, 1, 1)
    raster_y, raster_x = np.where(raster != 0)
    plt.scatter(
        raster_x,
        raster_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.title(r"$\textbf{Rasters}$")

    plt.subplot(4, 1, 2)
    plt.plot(
        t_raster,
        smooth_raster,
        lw=0.7,
        color=params["raw_color"],
        label="smooth raster",
    )
    plt.title(r"$\textbf{Raw}$")

    plt.subplot(4, 1, 3)

    for conv in range(x.shape[0]):
        x_conv = x[conv]
        x_onsets = t[x_conv > 0]
        plt.vlines(
            x_onsets,
            ymin=0,
            ymax=torch.max(x),
            color=params["onset_color_list"][conv],
            lw=0.7,
            label=f"Code {conv+1}",
        )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )
    plt.ylim(0)

    plt.subplot(4, 1, 4)

    for conv in range(x.shape[0]):
        x_conv = x.clone()
        x_conv[-conv + 1] = 0
        print(x_conv.shape)
        print(kernels.shape)
        Hx_conv = F.conv_transpose1d(x_conv, kernels)
        rate_conv = torch.sigmoid(Hx_conv + a)[0]
        rate_length = rate_conv.shape[-1]
        rate_t = np.arange(0, rate_length) * params["time_bin_resolution"]
        plt.plot(
            rate_t,
            rate_conv,
            color=params["onset_color_list"][conv],
            lw=0.7,
            label=f"Deconv {conv+1}",
        )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )

    xtic = [0, 500, 1500, 2000]

    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{Code}$")
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    # plt.show()
    # exit()
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
