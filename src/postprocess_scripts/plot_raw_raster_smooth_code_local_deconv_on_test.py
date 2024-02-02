"""
Copyright (c) 2020 Bahareh Tolooshams

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

sys.path.append("../src/")


import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        # default="../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35",
        # default="../results/2000_12sparse_local_deconv_calscenario_shorttrial_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse2period10_kernelsmooth0.015_knownsuppFalse_2023_10_28_00_14_22",
        default="../results/6000_3sparse_local_deconv_calscenario_longtrial_1600trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18",
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

    if (
        params_init["res_path"]
        == "../results/2000_12sparse_local_deconv_calscenario_shorttrial_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse2period10_kernelsmooth0.015_knownsuppFalse_2023_10_28_00_14_22"
    ):
        params_init[
            "data_path"
        ] = "../data/local-deconv-calscenario-shorttrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"

    elif (
        params_init["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35"
    ):
        params_init[
            "data_path"
        ] = "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"

    elif (
        params_init["res_path"]
        == "../results/6000_3sparse_local_deconv_calscenario_longtrial_1600trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18"
    ):
        params_init[
            "data_path"
        ] = "../data/local-deconv-calscenario-longtrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(params["data_path"])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

    # create datasets -------------------------------------------------------#
    dataset = datasetloader.DUNLdatasetwithRasterWithCodeRate(params["data_path"])

    data_folder = params["data_path"].split(params["data_path"].split("/")[-1])[0]
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

        print(
            train_raster_neuron.shape,
            train_x.shape,
            train_codes.shape,
            train_rate.shape,
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
                train_rate[trial, 0],
                train_codes[trial, 0],
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
    raster_y, raster_x = np.where(raster[:, 500:] != 0)
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
        smooth_raster[:, 500:],
        aspect="auto",
    )
    # plt.colorbar()
    plt.ylabel(r"$\textbf{Trials}$")
    plt.title(r"$\textbf{Smoothed Raster}$")

    plt.subplot(3, 1, 3)

    for conv in range(x.shape[1]):
        x_conv = x[:, conv][:, 20:]
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

    xtic_org = [0, 500, 1500]
    xtic = [int(x) + 500 for x in xtic_org]
    plt.xticks(xtic_org, xtic)
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
    rate,
    codes,
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
    t_rate_org = np.arange(0, rate.shape[-1])
    t_code = np.arange(0, codes.shape[-1])

    fig, axn = plt.subplots(5, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(5, 1, 1)
    raster_y, raster_x = np.where(raster[:, 500:] != 0)
    plt.scatter(
        raster_x + 500,
        raster_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.title(r"$\textbf{Rasters}$")

    plt.subplot(5, 1, 2)
    plt.plot(
        t_raster[500:],
        smooth_raster[500:],
        lw=0.7,
        color=params["raw_color"],
        label="smooth raster",
    )
    plt.title(r"$\textbf{Smooth\ Raster}$")

    plt.subplot(5, 1, 3)

    rate_org_length = rate.shape[-1]
    rate_org_t = np.arange(0, rate_org_length) * params["time_bin_resolution"]
    print(rate.shape, rate_org_t.shape)
    plt.plot(
        t_rate_org[500:],
        rate[500:],
        color="black",
        lw=0.7,
    )
    plt.title(r"$\textbf{Underlying\ Rate}$")
    plt.subplot(5, 1, 4)

    for conv in range(x.shape[0]):
        print(x.shape, codes.shape)
        # exit()
        codes_conv = codes[conv]

        plt.plot(
            t_code[codes_conv > 0][500:],
            codes_conv[codes_conv > 0][500:],
            ".",
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

    plt.subplot(5, 1, 5)

    for conv in range(x.shape[0]):
        code_conv = codes.clone()
        code_conv = torch.unsqueeze(code_conv, dim=0)
        print("here", "conv", conv)
        print(code_conv.shape)
        code_conv[:, -conv + 1] = 0

        print(code_conv.shape)
        print(kernels.shape)
        kernels = torch.tensor(kernels)
        kernels_org_resolution = torch.repeat_interleave(
            kernels, params["time_bin_resolution"], dim=-1
        )
        kernels_org_resolution = F.normalize(kernels_org_resolution, p=2, dim=-1)
        Hx_conv = F.conv_transpose1d(code_conv, kernels_org_resolution)
        rate_conv = torch.sigmoid(Hx_conv + a)[0]
        rate_length = rate_conv.shape[-1]
        rate_t = np.arange(0, rate_length)
        plt.plot(
            rate_t[500:],
            rate_conv[0][500:],
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

    xtic = [500, 1500, 2000]

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
