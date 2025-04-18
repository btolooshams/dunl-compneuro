"""
Copyright (c) 2025 Bahareh Tolooshams

plot recompose data

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

import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
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
        "--raw-color",
        type=str,
        help="raw color",
        default="Black",
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
        "--smoothing-tau",
        type=int,
        help="smoothing tau",
        default=20,
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

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # this is make sure the inference would be on full eshel data
    if (
        params_init["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35"
    ):
        params_init["test_data_path"] = [
            "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"
        ]

    elif (
        params_init["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_31"
    ):
        params_init["test_data_path"] = [
            "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"
        ]

    elif (
        params_init["res_path"]
        == "../results/6000_3sparse_local_deconv_calscenario_longtrial_1600trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18"
    ):
        params_init["test_data_path"] = [
            "../data/local-deconv-calscenario-longtrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"
        ]

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    data_path_list = params["test_data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

    # create datasets -------------------------------------------------------#
    dataset = datasetloader.DUNLdatasetwithRasterWithCodeRate(
        params["test_data_path"][0]
    )
    datafile_name = params["test_data_path"][0].split("/")[-1].split(".pt")[0]

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
        "figures",
        "decompose",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    codes = dataset.codes
    rate = dataset.rate

    # train_num_trials = len(dataset)

    y = torch.load(os.path.join(postprocess_path, "test_y_{}.pt".format(datafile_name)))
    y = y[:, 0, :]

    raster = dataset.raster[:, [0]]

    smooth_raster = utils.smooth_raster(raster, params["smoothing_tau"])

    xhat_org = torch.load(
        os.path.join(postprocess_path, "test_xhat_{}.pt".format(datafile_name))
    )
    xhat = codes * 0
    xhat[:, :, :, :: params["time_bin_resolution"]] = xhat_org
    print(xhat.shape)

    rate_hat = torch.load(
        os.path.join(postprocess_path, "test_ratehat_{}.pt".format(datafile_name))
    )
    print(rate_hat.shape)
    rate_hat = rate_hat[:, 0, :]

    deconv_0 = torch.load(
        os.path.join(postprocess_path, "test_deconv_0_{}.pt".format(datafile_name))
    )
    deconv_0 = deconv_0[:, 0, :]
    deconv_1 = torch.load(
        os.path.join(postprocess_path, "test_deconv_1_{}.pt".format(datafile_name))
    )
    deconv_1 = deconv_1[:, 0, :]

    for i in range(20):
        plot_deconv(
            y[i],
            rate_hat[i],
            deconv_0[i],
            deconv_1[i],
            rate[i, 0],
            codes[i, 0],
            xhat[i, 0],
            params,
            os.path.join(out_path, f"rate_and_deconv_example_{i}.png"),
        )
        plot_deconv(
            y[i],
            rate_hat[i],
            deconv_0[i],
            deconv_1[i],
            rate[i, 0],
            codes[i, 0],
            xhat[i, 0],
            params,
            os.path.join(out_path, f"rate_and_deconv_example_{i}.svg"),
        )
        plot_deconv_combined_rate_code(
            smooth_raster[i, 0],
            rate_hat[i],
            deconv_0[i],
            deconv_1[i],
            xhat[i, 0],
            params,
            os.path.join(out_path, f"rate_and_deconv_example_{i}_combined.svg"),
        )

    print(f"plotting of decomposition is done. plots are saved at {out_path}")


def plot_deconv(
    y,
    rate_hat,
    deconv_0,
    deconv_1,
    rate,
    codes,
    xhat,
    params,
    plot_filename,
):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    a = 0

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

    trial_length = rate_hat.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]
    t_rate_org = np.arange(0, rate.shape[-1])
    t_code = np.arange(0, codes.shape[-1])
    xhat_t_code = np.arange(0, xhat.shape[-1])

    fig, axn = plt.subplots(5, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(5, 1, 1)
    rate_org_length = rate.shape[-1]
    rate_org_t = np.arange(0, rate_org_length) * params["time_bin_resolution"]
    print(rate.shape, rate_org_t.shape)
    plt.plot(
        t_rate_org,
        rate,
        color="black",
        lw=0.7,
    )
    plt.title(r"$\textbf{Underlying\ Rate}$")

    plt.subplot(5, 1, 2)
    plt.plot(
        t[a:],
        y[a:],
        color="black",
        lw=0.7,
    )
    plt.title(r"$\textbf{Binned Spikes}$")

    plt.subplot(5, 1, 3)
    plt.plot(
        t[a:],
        rate_hat[a:],
        color="black",
        lw=0.7,
    )
    plt.title(r"$\textbf{Rate Estimate}$")

    plt.subplot(5, 1, 4)
    for conv in range(codes.shape[0]):
        codes_conv = codes[conv]

        plt.vlines(
            t_code[codes_conv > 0],
            ymin=0,
            ymax=np.max(xhat.numpy()) * 1.1,
            color=params["color_list"][conv],
            lw=0.7,
            label=f"True Onset {conv+1}",
        )
    for conv in range(xhat.shape[0]):
        xhat_conv = xhat[conv]

        plt.plot(
            xhat_t_code[xhat_conv > 0],
            xhat_conv[xhat_conv > 0],
            ".",
            color=params["color_list"][conv],
            lw=0.1,
            label=f"Estimated Code {conv+1}",
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
    plt.title(r"$\textbf{Code}$")

    plt.subplot(5, 1, 5)
    plt.plot(
        t[a:],
        deconv_0[a:],
        color=params["color_list"][0],
        lw=0.7,
        label=f"Deconv {1}",
    )
    plt.plot(
        t[a:],
        deconv_1[a:],
        color=params["color_list"][1],
        lw=0.7,
        label=f"Deconv {2}",
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
    plt.title(r"$\textbf{Estimated\ Deconvolution}$")

    if (
        params["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35"
    ):
        xtic = [0, 500, 1000, 1500, 2000]
    elif (
        params["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_31"
    ):
        xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    elif (
        params["res_path"]
        == "../results/6000_3sparse_local_deconv_calscenario_longtrial_1600trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18"
    ):
        xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]

    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{Deconvolution}$")
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


def plot_deconv_combined_rate_code(
    smooth_raster,
    rate_hat,
    deconv_0,
    deconv_1,
    xhat,
    params,
    plot_filename,
):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    a = 0

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

    trial_length = rate_hat.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]
    t_raster = np.arange(0, smooth_raster.shape[-1])
    xhat_t_code = np.arange(0, xhat.shape[-1])

    fig, axn = plt.subplots(3, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(3, 1, 1)
    plt.plot(
        t_raster,
        smooth_raster,
        lw=0.7,
        color=params["raw_color"],
        label="smooth raster",
    )

    plt.subplot(3, 1, 2)
    plt.plot(
        t[a:],
        rate_hat[a:],
        color="black",
        lw=0.7,
    )
    ax2 = axn[1].twinx()
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", direction="out")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    for conv in range(xhat.shape[0]):
        xhat_conv = xhat[conv]
        ax2.plot(
            xhat_t_code[xhat_conv > 0],
            xhat_conv[xhat_conv > 0],
            ".",
            color=params["color_list"][conv],
            lw=0.1,
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

    plt.subplot(3, 1, 3)
    plt.plot(
        t[a:],
        deconv_0[a:],
        color=params["color_list"][0],
        lw=0.7,
        label=f"Deconv {1}",
    )
    plt.plot(
        t[a:],
        deconv_1[a:],
        color=params["color_list"][1],
        lw=0.7,
        label=f"Deconv {2}",
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

    if (
        params["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35"
    ):
        xtic = [0, 500, 1000, 1500, 2000]
    elif (
        params["res_path"]
        == "../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_1600trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_31"
    ):
        xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    elif (
        params["res_path"]
        == "../results/6000_3sparse_local_deconv_calscenario_longtrial_1600trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18"
    ):
        xtic = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000]

    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
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
