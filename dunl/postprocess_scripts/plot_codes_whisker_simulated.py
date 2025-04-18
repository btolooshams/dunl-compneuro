"""
Copyright (c) 2025 Bahareh Tolooshams

plot whisker simulation - model characterization

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
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
        "--out-path", type=str, help="out path", default="../results/a-model"
    )
    parser.add_argument(
        "--code-sparse-regularization",
        type=float,
        help="lam reg",
        default=0.03,
    )
    parser.add_argument(
        "--tol-list",
        type=list,
        help="tol list for hit rate",
        default=[1, 2, 3, 5, 6, 7, 10, 15, 20, 25, 50],  # this is in ms
    )

    parser.add_argument(
        "--kernel-length-ms", type=int, help="kernel length in ms", default=500
    )
    parser.add_argument(
        "--time-bin-resolution-list",
        type=int,
        help="time bin resolution list",  # see main
        default=[
            5,
            10,
            25,
            50,
        ],  # I assume the original resolution of data is 1 ms, so this would be 25 ms.
    )
    parser.add_argument(
        "--num-trials-list",
        type=int,
        help="number of trials list",  # see main
        default=[25, 50, 100, 250, 500, 1000],
    )
    parser.add_argument(
        "--baseline-mean-list",
        type=float,
        help="baseline mean across neurons",  # see main
        default=[
            -6.2126,  # 2 Hz
            -5.2933,  # 5 Hz
            -4.8203,  # 8 Hz
            -4.4988,  # 11 Hz
            -4.2546,  # 14 Hz
            -4.0574,  # 17 Hz
            # -3.8918, # 20 Hz
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
            "green",
            "yellow",
            "black",
        ],
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(4, 4),
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

    out_path = params_init["out_path"]

    baseline_in_Hz = dict()
    baseline_in_Hz["-6.2126"] = 2
    baseline_in_Hz["-5.2933"] = 5
    baseline_in_Hz["-4.8203"] = 8
    baseline_in_Hz["-4.4988"] = 11
    baseline_in_Hz["-4.2546"] = 14
    baseline_in_Hz["-4.0574"] = 17
    baseline_in_Hz["-3.8918"] = 20

    baseline_in_Hz_list = [2, 5, 8, 11, 14, 17]

    res_path_dict_unknown = get_result_path(
        params_init, baseline_in_Hz, code_supp=False
    )
    res_path_dict_known = get_result_path(params_init, baseline_in_Hz, code_supp=True)

    # trials, bin, baseline

    for tol in params_init["tol_list"]:
        print("tol:", tol)

        code_hitrate_unknown = get_code_err_matrix(
            params_init, res_path_dict_unknown, baseline_in_Hz, baseline_in_Hz_list, tol
        )
        code_hitrate_known = get_code_err_matrix(
            params_init, res_path_dict_known, baseline_in_Hz, baseline_in_Hz_list, tol
        )

        for bin_ctr in range(len(params_init["time_bin_resolution_list"])):
            for legend in [True, False]:
                plot_code_hitrate_one_per_timebinresolution(
                    code_hitrate_unknown,
                    params_init,
                    baseline_in_Hz_list,
                    bin_ctr,
                    "unknown_timing_tol{}".format(tol),
                    out_path,
                    legend,
                )

        for trial_ctr in range(len(params_init["num_trials_list"])):
            for legend in [True, False]:
                plot_code_hitrate_one_per_numtrials(
                    code_hitrate_unknown,
                    params_init,
                    baseline_in_Hz_list,
                    trial_ctr,
                    "unknown_timing_tol{}".format(tol),
                    out_path,
                    legend,
                )

    print("done!")


def get_result_path(params, baseline_in_Hz, code_supp):
    res_path_dict = dict()

    if code_supp:
        random_date = "2023_07_25_02_23_25"
    else:
        random_date = "2023_07_24_23_12_31"

    if code_supp:
        event_status = "knownEvent"
    else:
        event_status = "unknownEventKnownNumEvent"

    lam = str(params["code_sparse_regularization"]).replace(".", "p")

    for num_trials in params["num_trials_list"]:
        res_path_dict["num_trials_{}".format(num_trials)] = dict()

        for time_bin_resolution in params["time_bin_resolution_list"]:
            res_path_dict["num_trials_{}".format(num_trials)][
                "time_bin_resolution_{}".format(time_bin_resolution)
            ] = dict()

            kernel_length = int(params["kernel_length_ms"] / time_bin_resolution)

            for baseline_mean in params["baseline_mean_list"]:
                baseline_in_Hz_curr = baseline_in_Hz["{}".format(baseline_mean)]

                res_path = f"../results/simulated_1neurons_{num_trials}trials_{time_bin_resolution}msbinres_{baseline_in_Hz_curr}Hzbaseline_kernellength{kernel_length}_{event_status}_lam{lam}_{random_date}"
                res_path_dict["num_trials_{}".format(num_trials)][
                    "time_bin_resolution_{}".format(time_bin_resolution)
                ]["baseline_mean_{}".format(baseline_mean)] = res_path

    return res_path_dict


def get_code_err_matrix(
    params_init, res_path_dict, baseline_in_Hz, baseline_in_Hz_list, tol, device="cpu"
):
    hit_rate_matrix = np.zeros(
        (
            len(params_init["num_trials_list"]),
            len(params_init["time_bin_resolution_list"]),
            len(params_init["baseline_mean_list"]),
        )
    )  # trials, bin, baseline

    for num_trials in params_init["num_trials_list"]:
        for time_bin_resolution in params_init["time_bin_resolution_list"]:
            for baseline_mean in params_init["baseline_mean_list"]:
                baseline_in_Hz_curr = baseline_in_Hz["{}".format(baseline_mean)]

                res_path = res_path_dict["num_trials_{}".format(num_trials)][
                    "time_bin_resolution_{}".format(time_bin_resolution)
                ]["baseline_mean_{}".format(baseline_mean)]

                i = np.where(np.array(params_init["num_trials_list"]) == num_trials)[0][
                    0
                ]
                j = np.where(
                    np.array(params_init["time_bin_resolution_list"])
                    == time_bin_resolution
                )[0][0]
                k = np.where(np.array(baseline_in_Hz_list) == baseline_in_Hz_curr)[0][0]

                # create folders -------------------------------------------------------#
                postprocess_path = os.path.join(
                    res_path,
                    "postprocess",
                )

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

                data_folder = data_path_list[0].split(data_path_list[0].split("/")[-1])[
                    0
                ]

                # load test folder  ------------------------------------------------#
                data_train_filename = data_path_list[0].split("/")[-1]
                data_test_filename_first = data_train_filename.split(
                    "_{}trials".format(num_trials)
                )[0]
                data_test_filename_second = data_train_filename.split("trials")[-1]
                data_test_filename = (
                    f"{data_test_filename_first}_100trials{data_test_filename_second}"
                )
                test_path = os.path.join(
                    data_folder, "test_{}".format(data_test_filename)
                )

                numpy_test_path = "{}_format_processed.npy".format(
                    test_path.split("_format_processed")[0]
                )
                numpy_test_dict = torch.load(numpy_test_path)
                x = numpy_test_dict["codes"][:, 0]

                # load x and xhat -------------------------------------------------------#
                xhat_load = torch.load(os.path.join(postprocess_path, "xhat.pt"))
                xhat_load = xhat_load[
                    :, 0
                ]  # take the neuron 1 (there is only one neuron)
                xhat = torch.repeat_interleave(xhat_load, time_bin_resolution, dim=-1)
                xhat = xhat[:, :, : x.shape[-1]]  # match the dim

                # compute code error
                hit_rate_xhat, false_rate_xhat = utils.compute_hit_rate(
                    x, xhat, tol=tol
                )
                hit_rate_xhat = np.minimum(hit_rate_xhat, 1)

                # trials, bin, baseline
                hit_rate_matrix[i, j, k] = hit_rate_xhat

    return hit_rate_matrix


def plot_code_hitrate_one_per_numtrials(
    code_hitrate,
    params,
    baseline_in_Hz_list,
    trial_ctr,
    scenario,
    out_path,
    legend=False,
):
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

    num_trials = params["num_trials_list"][trial_ctr]

    # trials, bin, baseline
    for ctr in range(len(params["time_bin_resolution_list"])):
        label = "Bin {} ms".format(params["time_bin_resolution_list"][ctr])
        plt.plot(
            baseline_in_Hz_list,
            code_hitrate[trial_ctr, ctr, :],
            label=label,
            color=params["color_list"][ctr],
        )

    if legend:
        plt.legend()

    plt.xlabel("Baseline activity")
    plt.ylabel("Code Hit Rate")
    plt.title("Num Trials {}".format(num_trials))

    xtic = baseline_in_Hz_list
    xtic = [x for x in xtic]
    plt.xticks(xtic, xtic)

    ytic = np.array([0, 0.5, 1])
    ytic = [x for x in ytic]
    plt.yticks(ytic, ytic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path,
            "codehitrate_numtrials{}_{}_legend{}.svg".format(
                num_trials, scenario, legend
            ),
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_code_hitrate_one_per_timebinresolution(
    code_hitrate, params, baseline_in_Hz_list, bin_ctr, scenario, out_path, legend=False
):
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

    bin_res = params["time_bin_resolution_list"][bin_ctr]

    # trials, bin, baseline
    for ctr in range(len(params["num_trials_list"])):
        label = "Num Trials {}".format(params["num_trials_list"][ctr])
        plt.plot(
            baseline_in_Hz_list,
            code_hitrate[ctr, bin_ctr, :],
            label=label,
            color=params["color_list"][ctr],
        )

    if legend:
        plt.legend()

    plt.xlabel("Baseline activity")
    plt.ylabel("Code Hit Rate")
    plt.title("Bin {} ms".format(bin_res))

    xtic = baseline_in_Hz_list
    xtic = [x for x in xtic]
    plt.xticks(xtic, xtic)

    ytic = np.array([0, 0.5, 1])
    ytic = [x for x in ytic]
    plt.yticks(ytic, ytic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path,
            "codehitrate_binres{}ms_{}_legend{}.svg".format(bin_res, scenario, legend),
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
