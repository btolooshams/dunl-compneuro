"""
Copyright (c) 2025 Bahareh Tolooshams

plot code local deconv

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

sys.path.append("../src/")

import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-partial",
        type=str,
        help="res path partial",
        # default="../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured",
        default="../results/6000_3sparse_local_deconv_calscenario_longtrial",
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
        "--color-list",
        type=list,
        help="color decomposition list",
        default=[
            "blue",
            "red",
        ],  #
    )
    parser.add_argument(
        "--onset-color-list",
        type=list,
        help="onset color list",
        default=["Blue", "Red"],
    )
    parser.add_argument(
        "--raster-markersize",
        type=int,
        help="raster markersize",
        default=0.2,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def compute_r2_score(spikes, rate_hat):
    # compute r2 score
    ss_res = np.mean((spikes - rate_hat), axis=1) ** 2
    ss_tot = np.var(spikes)

    r2_fit = 1 - ss_res / ss_tot

    return np.mean(r2_fit)


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    for trials in [25, 50, 100, 200, 400, 800, 1600]:
        # init parameters -------------------------------------------------------#
        params_init = init_params()

        # this is make sure the inference would be on full eshel data
        if (
            params_init["res_path_partial"]
            == f"../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured"
        ):
            params_init["res_path_list"] = [
                f"../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_{trials}trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35",
                # f"../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_{trials}trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_31",
            ]

            params_init["test_data_path"] = [
                "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"
            ]

        elif (
            params_init["res_path_partial"]
            == f"../results/6000_3sparse_local_deconv_calscenario_longtrial"
        ):
            params_init["res_path_list"] = [
                f"../results/6000_3sparse_local_deconv_calscenario_longtrial_{trials}trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18",
                f"../results/6000_3sparse_local_deconv_calscenario_longtrial_{trials}trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_26",
            ]
            params_init["test_data_path"] = [
                "../data/local-deconv-calscenario-longtrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"
            ]

        hit_rate_xhat_tol_avg = list()
        for res_path in params_init["res_path_list"]:
            print("res_path", res_path)

            params_init["res_path"] = res_path

            # take parameters from the result path
            params = pickle.load(
                open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
            )
            for key in params_init.keys():
                params[key] = params_init[key]

            data_path_list = params["test_data_path"]

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

            postprocess_path = os.path.join(
                params["res_path"],
                "postprocess",
            )

            datafile_name = params["test_data_path"][0].split("/")[-1].split(".pt")[0]

            dataset = datasetloader.DUNLdatasetwithRasterWithCodeRate(
                params["test_data_path"][0]
            )
            codes = dataset.codes

            xhat = torch.load(
                os.path.join(postprocess_path, "test_xhat_{}.pt".format(datafile_name))
            )
            x = torch.load(
                os.path.join(postprocess_path, "test_x_{}.pt".format(datafile_name))
            )

            plot_code_est_several(
                xhat[:, 0],
                params,
                os.path.join(out_path, f"codes_dunl_est.svg"),
            )
            plot_code_true_several(
                codes[:, 0],
                params,
                os.path.join(out_path, f"codes_true.svg"),
            )

            print(f"plotting of decomposition is done. plots are saved at {out_path}")

            #### for xhat
            hit_rate_xhat_tol = list()
            for tol in [1, 2, 3]:
                ####################
                hit_rate_xhat_0, false_rate_xhat = utils.compute_hit_rate(
                    x[:, [0]], xhat[:, 0, [0]], tol=tol
                )
                hit_rate_xhat_0 = np.minimum(hit_rate_xhat_0, 1)

                hit_rate_xhat_1, false_rate_xhat = utils.compute_hit_rate(
                    x[:, [1]], xhat[:, 0, [1]], tol=tol
                )
                hit_rate_xhat_1 = np.minimum(hit_rate_xhat_1, 1)

                hit_rate_xhat = 0.5 * (hit_rate_xhat_0 + hit_rate_xhat_1)

                ####################
                hit_rate_xhat_0_swap, false_rate_xhat = utils.compute_hit_rate(
                    x[:, [0]], xhat[:, 0, [1]], tol=tol
                )
                hit_rate_xhat_0_swap = np.minimum(hit_rate_xhat_0_swap, 1)

                hit_rate_xhat_1_swap, false_rate_xhat = utils.compute_hit_rate(
                    x[:, [1]], xhat[:, 0, [0]], tol=tol
                )
                hit_rate_xhat_1_swap = np.minimum(hit_rate_xhat_1_swap, 1)

                hit_rate_xhat_swap = 0.5 * (hit_rate_xhat_0_swap + hit_rate_xhat_1_swap)

                ############### take the max, this for swap
                hit_rate_xhat_final = np.maximum(hit_rate_xhat, hit_rate_xhat_swap)

                # print("trials", trials, "tol", tol, hit_rate_xhat)

                hit_rate_xhat_tol.append(hit_rate_xhat_final)

            hit_rate_xhat_tol_avg.append(hit_rate_xhat_tol)

        hit_rate_xhat_tol_avg = np.stack(hit_rate_xhat_tol_avg, axis=0)
        # print("before", hit_rate_xhat_tol_avg)
        hit_rate_xhat_tol_avg = np.mean(hit_rate_xhat_tol_avg, axis=0)

        print("trials", trials, "hit rate", hit_rate_xhat_tol_avg)


def plot_code_est_several(
    x,
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

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False)

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(1, 1, 1)

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

    xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    # xtic = [0, 500, 1500, 2500, 3500, 4500, 5500, 6500]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{Estimated\ Code}$")
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_code_true_several(
    x,
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

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False)

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(1, 1, 1)

    for conv in range(x.shape[1]):
        x_conv = x[:, conv]
        code_y, code_x = np.where(x_conv != 0)

        plt.scatter(
            code_x,
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

    xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    # xtic = [0, 500, 1500, 2500, 3500, 4500, 5500, 6500]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{True\ Code}$")
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
