"""
Copyright (c) 2020 Bahareh Tolooshams

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

sys.path.append("../src/")

import utils


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
    kernel_err_unknown = get_dict_err_matrix(
        params_init,
        res_path_dict_unknown,
        baseline_in_Hz,
        baseline_in_Hz_list,
        "unknown_timing",
    )
    kernel_err_known = get_dict_err_matrix(
        params_init,
        res_path_dict_known,
        baseline_in_Hz,
        baseline_in_Hz_list,
        "known_timing",
    )

    for trial_ctr in range(len(params_init["num_trials_list"])):
        plot_kernelerr(
            kernel_err_unknown,
            params_init,
            baseline_in_Hz_list,
            trial_ctr,
            "unknown_timing",
            out_path,
        )
        plot_kernelerr(
            kernel_err_known,
            params_init,
            baseline_in_Hz_list,
            trial_ctr,
            "known_timing",
            out_path,
        )


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


def get_dict_err_matrix(
    params_init,
    res_path_dict,
    baseline_in_Hz,
    baseline_in_Hz_list,
    scenario,
    device="cpu",
):
    kernel_err_matrix = np.zeros(
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

                # load true kernel  ------------------------------------------------#
                data_folder = data_path_list[0].split(data_path_list[0].split("/")[-1])[
                    0
                ]
                kernels_true_org_res = torch.load(
                    os.path.join(data_folder, "kernels.pt")
                )
                kernels_true_org_res = kernels_true_org_res.cpu().numpy()

                # create folders -------------------------------------------------------#
                out_path = params["out_path"]
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                best_epoch = np.load(
                    os.path.join(res_path, "best_epoch_based_on_true_kernel.npy")
                )

                model_path = os.path.join(
                    res_path,
                    "model",
                    "model_epoch{}.pt".format(best_epoch),
                )
                net = torch.load(model_path, map_location=device)
                net.to(device)
                net.eval()

                kernels = net.get_param("H").clone().detach()
                kernels = np.squeeze(kernels.cpu().numpy(), axis=1)

                length = kernels.shape[-1]
                a = kernels_true_org_res.shape[-1] / length
                coordinates_inter = np.linspace(1, int(a * length), int(a * length))
                coordinates_x = np.linspace(1, int(a * length), int(length))
                best_kernels_on_org_resolution = np.interp(
                    coordinates_inter, coordinates_x, kernels[0]
                )
                best_kernels_on_org_resolution = torch.tensor(
                    best_kernels_on_org_resolution, dtype=torch.float
                )
                best_kernels_on_org_resolution = F.normalize(
                    best_kernels_on_org_resolution, p=2, dim=-1
                )
                best_kernels_on_org_resolution = torch.unsqueeze(
                    best_kernels_on_org_resolution, dim=0
                )

                best_kernels_on_org_resolution = (
                    best_kernels_on_org_resolution.cpu().numpy()
                )

                kernel_err_best_epoch = utils.compute_dictionary_error(
                    best_kernels_on_org_resolution, kernels_true_org_res
                )[0]

                plot_kernel(
                    best_kernels_on_org_resolution,
                    kernels_true_org_res,
                    params,
                    num_trials,
                    time_bin_resolution,
                    scenario,
                    out_path,
                )

                # trials, bin, baseline
                kernel_err_matrix[i, j, k] = kernel_err_best_epoch

    return kernel_err_matrix


def plot_kernelerr(
    kernel_err, params, baseline_in_Hz_list, trial_ctr, scenario, out_path
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

    ax.axhline(0.7943382175935327, color="gray", lw=0.3)

    # trials, bin, baseline
    for ctr in range(len(params["time_bin_resolution_list"])):
        label = "Bin {} ms".format(params["time_bin_resolution_list"][ctr])
        plt.plot(
            baseline_in_Hz_list,
            kernel_err[trial_ctr, ctr, :],
            label=label,
            color=params["color_list"][ctr],
        )

    # plt.legend()

    plt.xlabel("Baseline activity")
    plt.ylabel("Kernel Error")
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
            out_path, "kernelerr_numtrials{}_{}.svg".format(num_trials, scenario)
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print("plotting done!")


def plot_kernel(
    kernels_est, kernels_true, params, num_trials, bin_res, scenario, out_path
):
    t = np.linspace(
        0,
        kernels_est.shape[-1],
        kernels_est.shape[-1],
    )

    # plot configuration -------------------------------------------------------#

    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 15
    fontfamily = "sans-serif"

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

    # plot -------------------------------------------------------#
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(2, 2))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for ctr in range(params["kernel_num"]):
        plt.subplot(1, 1, ctr + 1)
        ax.axhline(0, color="gray", lw=0.3)

        plt.plot(t, kernels_est[ctr], color="Blue")
        plt.plot(t, kernels_true[ctr], color="Black")
        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)
        plt.xlabel("Time [ms]", labelpad=0)
    plt.title(f"{bin_res} ms, {num_trials} Tri")

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path,
            "kernel_numtrials{}_binres{}ms{}.svg".format(num_trials, bin_res, scenario),
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
