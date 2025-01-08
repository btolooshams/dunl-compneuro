"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy as sp
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
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_fixedq_2p5_firstshrinkage_2023_09_27_01_17_09",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_fixedq_2p5_firstshrinkage_2023_09_27_00_58_18",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_15_33_55",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_kernelinitsmootherTrue_2023_07_13_08_23_18",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_kernelinitsmootherTrue_2023_07_13_17_01_14",
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.0, 0.3, 0.5, 1.2, 2.5, 5.0, 8.0, 11.0],
    )
    parser.add_argument(
        "--window-dur",
        type=int,
        help="window duration to get average activity",
        default=60,  # 15 * 4 s = 60
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=15,  # 15 Hz
    )
    parser.add_argument(
        "--n-bin-spearman",
        type=int,
        help="number of bins in spearman histogram",
        default=10,
    )

    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "orange",
            "blue",
            "brown",
            "black",
        ],
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(13, 2),
    )
    parser.add_argument(
        "--window-start_list",
        type=list,
        help="window start list",
        default=list(np.arange(0, 24)),  # 15 samples is 1 second (24 is 1.6 seconds)
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Predict.")
    print("WARNING! This script assumes that each code is 1-sparse.")

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

    # load data -------------------------------------------------------#

    code_exp_corr_from_all_datasets = list()
    code_sur_corr_from_all_datasets = list()
    y_exp_corr_from_all_datasets = list()
    y_sur_corr_from_all_datasets = list()

    y_exp_corr_from_all_datasets_window = dict()
    y_sur_corr_from_all_datasets_window = dict()
    for window_start in params["window_start_list"]:
        y_exp_corr_from_all_datasets_window["{}".format(window_start)] = list()
        y_sur_corr_from_all_datasets_window["{}".format(window_start)] = list()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        y = torch.load(os.path.join(postprocess_path, "y_{}.pt".format(datafile_name)))
        x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        label = torch.load(
            os.path.join(postprocess_path, "label_{}.pt".format(datafile_name))
        )

        num_trials = xhat.shape[0]
        num_neurons = xhat.shape[1]

        code_expected = list()
        code_surprise = list()
        yavg_expected = list()
        yavg_surprise = list()
        expected_rew_amount = list()
        surprise_rew_amount = list()

        yavg_expected_window = dict()
        yavg_surprise_window = dict()
        for window_start in params["window_start_list"]:
            yavg_expected_window["{}".format(window_start)] = list()
            yavg_surprise_window["{}".format(window_start)] = list()

        # go over all trials
        for i in range(num_trials):
            yi = y[i]
            xi = x[i]
            xihat = xhat[i]
            labeli = label[i]

            if labeli == 0:
                continue
            else:
                # reward presence
                cue_flag = torch.sum(torch.abs(xi[1]), dim=-1).item()
                reward_onset = np.where(xi[2] > 0)[-1][0]
                if cue_flag:
                    # expected trial
                    y_expected_curr = torch.sum(
                        yi[:, reward_onset : reward_onset + params["window_dur"]],
                        dim=-1,
                    )
                    yavg_expected.append(y_expected_curr)
                    expected_rew_amount.append(labeli)
                    code_expected.append(torch.sum(xihat, dim=-1))

                    for window_start in params["window_start_list"]:
                        y_expected_curr_window = torch.sum(
                            yi[
                                :,
                                reward_onset
                                + window_start : reward_onset
                                + params["window_dur"],
                            ],
                            dim=-1,
                        )
                        yavg_expected_window["{}".format(window_start)].append(
                            y_expected_curr_window
                        )
                else:
                    # surprise trial
                    y_surprise_curr = torch.sum(
                        yi[:, reward_onset : reward_onset + params["window_dur"]],
                        dim=-1,
                    )
                    yavg_surprise.append(y_surprise_curr)
                    surprise_rew_amount.append(labeli)
                    code_surprise.append(torch.sum(xihat, dim=-1))

                    for window_start in params["window_start_list"]:
                        y_surprise_curr_window = torch.sum(
                            yi[
                                :,
                                reward_onset
                                + window_start : reward_onset
                                + params["window_dur"],
                            ],
                            dim=-1,
                        )
                        yavg_surprise_window["{}".format(window_start)].append(
                            y_surprise_curr_window
                        )

        code_expected = torch.stack(code_expected, dim=0)
        code_surprise = torch.stack(code_surprise, dim=0)
        yavg_expected = torch.stack(yavg_expected, dim=0)
        yavg_surprise = torch.stack(yavg_surprise, dim=0)
        for window_start in params["window_start_list"]:
            yavg_expected_window["{}".format(window_start)] = torch.stack(
                yavg_expected_window["{}".format(window_start)], dim=0
            )
            yavg_surprise_window["{}".format(window_start)] = torch.stack(
                yavg_surprise_window["{}".format(window_start)], dim=0
            )

        expected_rew_amount = torch.stack(expected_rew_amount, dim=0)
        surprise_rew_amount = torch.stack(surprise_rew_amount, dim=0)

        # go over all neurons and compute spearman correlation
        code_exp_corr = torch.zeros(num_neurons, 4)
        code_sur_corr = torch.zeros(num_neurons, 4)
        y_exp_corr = torch.zeros(num_neurons)
        y_sur_corr = torch.zeros(num_neurons)

        y_sur_corr_window = dict()
        y_exp_corr_window = dict()
        for window_start in params["window_start_list"]:
            y_sur_corr_window["{}".format(window_start)] = torch.zeros(num_neurons)
            y_exp_corr_window["{}".format(window_start)] = torch.zeros(num_neurons)

        for neuron_ctr in range(num_neurons):
            yavg_expected_neuron = yavg_expected[:, neuron_ctr]
            yavg_surprise_neuron = yavg_surprise[:, neuron_ctr]
            code_expected_neuron = code_expected[:, neuron_ctr]
            code_surprise_neuron = code_surprise[:, neuron_ctr]

            y_exp_corr_neuron, _ = sp.stats.spearmanr(
                yavg_expected_neuron, expected_rew_amount
            )
            y_sur_corr_neuron, _ = sp.stats.spearmanr(
                yavg_surprise_neuron, surprise_rew_amount
            )
            y_exp_corr[neuron_ctr] = y_exp_corr_neuron
            y_sur_corr[neuron_ctr] = y_sur_corr_neuron

            for window_start in params["window_start_list"]:
                yavg_expected_window_neuron = yavg_expected_window[
                    "{}".format(window_start)
                ][:, neuron_ctr]
                yavg_surprise_window_neuron = yavg_surprise_window[
                    "{}".format(window_start)
                ][:, neuron_ctr]

                y_exp_corr_neuron_window, _ = sp.stats.spearmanr(
                    yavg_expected_window_neuron, expected_rew_amount
                )
                y_sur_corr_neuron_window, _ = sp.stats.spearmanr(
                    yavg_surprise_window_neuron, surprise_rew_amount
                )

                y_exp_corr_window["{}".format(window_start)][
                    neuron_ctr
                ] = y_exp_corr_neuron_window
                y_sur_corr_window["{}".format(window_start)][
                    neuron_ctr
                ] = y_sur_corr_neuron_window

            for code_ctr in range(1, params["kernel_num"]):
                if code_ctr < 3:
                    # for cue expected, and reward I
                    (
                        code_exp_corr_neuron_curr_code,
                        code_exp_pvalue,
                    ) = sp.stats.spearmanr(
                        code_expected_neuron[:, code_ctr], expected_rew_amount
                    )
                    (
                        code_sur_corr_neuron_curr_code,
                        code_sur_pvalue,
                    ) = sp.stats.spearmanr(
                        code_surprise_neuron[:, code_ctr], surprise_rew_amount
                    )
                if code_ctr == 3:
                    # for reward coupled together
                    (
                        code_exp_corr_neuron_curr_code,
                        code_exp_pvalue,
                    ) = sp.stats.spearmanr(
                        code_expected_neuron[:, 3] + code_expected_neuron[:, 4],
                        expected_rew_amount,
                    )
                    (
                        code_sur_corr_neuron_curr_code,
                        code_sur_pvalue,
                    ) = sp.stats.spearmanr(
                        code_surprise_neuron[:, 3] + code_surprise_neuron[:, 4],
                        surprise_rew_amount,
                    )
                elif code_ctr == 4:
                    # for all rewards (reward I + reward coupled together)
                    (
                        code_exp_corr_neuron_curr_code,
                        code_exp_pvalue,
                    ) = sp.stats.spearmanr(
                        code_expected_neuron[:, 2]
                        + code_expected_neuron[:, 3]
                        + code_expected_neuron[:, 4],
                        expected_rew_amount,
                    )
                    (
                        code_sur_corr_neuron_curr_code,
                        code_sur_pvalue,
                    ) = sp.stats.spearmanr(
                        code_surprise_neuron[:, 2]
                        + code_surprise_neuron[:, 3]
                        + code_surprise_neuron[:, 4],
                        surprise_rew_amount,
                    )

                code_exp_corr[neuron_ctr, code_ctr - 1] = code_exp_corr_neuron_curr_code
                code_sur_corr[neuron_ctr, code_ctr - 1] = code_sur_corr_neuron_curr_code

        code_exp_corr_from_all_datasets.append(code_exp_corr)
        code_sur_corr_from_all_datasets.append(code_sur_corr)
        y_exp_corr_from_all_datasets.append(y_exp_corr)
        y_sur_corr_from_all_datasets.append(y_sur_corr)
        for window_start in params["window_start_list"]:
            y_exp_corr_from_all_datasets_window["{}".format(window_start)].append(
                y_exp_corr_window["{}".format(window_start)]
            )
            y_sur_corr_from_all_datasets_window["{}".format(window_start)].append(
                y_sur_corr_window["{}".format(window_start)]
            )

    code_exp_corr_from_all_datasets = torch.cat(code_exp_corr_from_all_datasets, dim=0)
    code_sur_corr_from_all_datasets = torch.cat(code_sur_corr_from_all_datasets, dim=0)
    y_exp_corr_from_all_datasets = torch.cat(y_exp_corr_from_all_datasets, dim=0)
    y_sur_corr_from_all_datasets = torch.cat(y_sur_corr_from_all_datasets, dim=0)
    for window_start in params["window_start_list"]:
        y_exp_corr_from_all_datasets_window["{}".format(window_start)] = torch.cat(
            y_exp_corr_from_all_datasets_window["{}".format(window_start)], dim=0
        )
        y_sur_corr_from_all_datasets_window["{}".format(window_start)] = torch.cat(
            y_sur_corr_from_all_datasets_window["{}".format(window_start)], dim=0
        )

    for code_ctr in range(code_exp_corr_from_all_datasets.shape[1]):
        print(code_ctr)
        stat, pvalue = sp.stats.ttest_rel(
            code_exp_corr_from_all_datasets[:, code_ctr], y_exp_corr_from_all_datasets
        )
        print("exp", stat, pvalue)

        stat, pvalue = sp.stats.ttest_rel(
            code_sur_corr_from_all_datasets[:, code_ctr], y_sur_corr_from_all_datasets
        )
        print("sur", stat, pvalue)

    plot_spearman_correlation(
        code_exp_corr_from_all_datasets,
        y_exp_corr_from_all_datasets,
        code_sur_corr_from_all_datasets,
        y_sur_corr_from_all_datasets,
        params,
        out_path_name=os.path.join(
            out_path,
            "spearman_correlation.svg",
        ),
    )

    for code_ctr in range(code_exp_corr_from_all_datasets.shape[1]):
        print(code_ctr, "exp")
        plot_spearman_correlation_hist_window(
            code_exp_corr_from_all_datasets[:, code_ctr],
            y_exp_corr_from_all_datasets_window,
            params,
            out_path_name=os.path.join(
                out_path,
                "spearman_correlation_distance_subplot{}_exp.svg".format(code_ctr),
            ),
        )
        if code_ctr > 0:
            print(code_ctr, "sur")
            plot_spearman_correlation_hist_window(
                code_sur_corr_from_all_datasets[:, code_ctr],
                y_sur_corr_from_all_datasets_window,
                params,
                out_path_name=os.path.join(
                    out_path,
                    "spearman_correlation_distance_subplot{}_sur.svg".format(code_ctr),
                ),
            )


def plot_spearman_correlation(code_exp, y_exp, code_sur, y_sur, params, out_path_name):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    markersize = 4
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

    fig, axn = plt.subplots(1, 4, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")

    for code_ctr in range(code_sur.shape[1]):
        plt.subplot(1, 4, code_ctr + 1)
        plt.scatter(
            code_sur[:, code_ctr],
            y_sur,
            marker=".",
            color=params["color_list"][code_ctr],
            s=markersize,
        )
        plt.scatter(
            code_exp[:, code_ctr],
            y_exp,
            marker=".",
            color=params["color_list"][code_ctr],
            s=markersize,
        )
        plt.plot([-0.5, 1], [-0.5, 1], "--", color="gray", linewidth=1)

        if code_ctr == 0:
            axn[code_ctr].set_ylabel("Spearman(Raw, Reward)")
            axn[code_ctr + 1].set_xlabel("Spearman(Code, Reward)")

        if code_ctr == 0:
            plt.title(r"$\textbf{Cue\ Expected}$")
        elif code_ctr == 1:
            plt.title(r"$\textbf{Reward}$")
        elif code_ctr == 2:
            plt.title(r"$\textbf{Reward\ Coupled\ Combined}$")
        else:
            plt.title(r"$\textbf{Reward\ Combined} $")

        plt.scatter(
            torch.mean(code_sur[:, code_ctr]),
            torch.mean(y_sur),
            marker="x",
            color="green",
            lw=1.5,
            s=markersize * 6,
        )
        plt.scatter(
            torch.mean(code_exp[:, code_ctr]),
            torch.mean(y_exp),
            marker="x",
            color="yellow",
            lw=1.5,
            s=markersize * 6,
        )
        xtic = np.array([0, 0.5, 1])
        xtic = [x for x in xtic]
        plt.xticks(xtic, xtic)
        plt.yticks(xtic, xtic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close()


def plot_spearman_correlation_hist_window(
    code_one_kernel, y_window, params, out_path_name, a=1.5556
):
    axes_fontsize = 20
    legend_fontsize = 8
    tick_fontsize = 20
    title_fontsize = 25
    markersize = 40
    fontfamily = "sans-serif"
    lw = 2

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

    cmap = "YlGn"

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    distance_from_diag_list = list()
    for window_start in params["window_start_list"]:
        distance_from_diag = utils.compute_signed_distance_from_diag(
            code_one_kernel,
            y_window["{}".format(window_start)],
        )
        distance_from_diag_list.append(distance_from_diag)
    best_scenario_index = np.argmin(np.mean(np.array(distance_from_diag_list), axis=1))

    print("best_scenario_index", best_scenario_index)
    print("window", params["window_start_list"][best_scenario_index])

    hist_heatmap = []
    distance_mean_list = []

    # a = np.max(np.abs(distance_from_diag_list)) * 1.1
    for plot_ctr in range(len(distance_from_diag_list)):
        hist, bin_edges = np.histogram(
            distance_from_diag_list[plot_ctr],
            bins=params["n_bin_spearman"],
            range=(-a, a),
            density=True,
        )
        distance_mean = np.mean(distance_from_diag_list[plot_ctr])
        hist_heatmap.append(hist)
        distance_mean_list.append(distance_mean)

    hist_heatmap = np.array(hist_heatmap)
    distance_mean_list = np.array(distance_mean_list)

    y, x = np.meshgrid(
        params["window_start_list"],
        np.linspace(-a, a, params["n_bin_spearman"]),
    )
    c = ax.pcolormesh(x, y, hist_heatmap.T, cmap=cmap)
    ax.scatter(
        distance_mean_list,
        params["window_start_list"],
        marker=".",
        color="red",
        s=markersize,
    )
    ax.axvline(0, color="black", lw=lw)
    plt.colorbar(c)

    tic = np.array(
        [
            0,
            params["window_start_list"][2],
            params["window_start_list"][5],
            params["window_start_list"][10],
            params["window_start_list"][20],
            params["window_start_list"][-1],
        ]
    )
    tic = [x for x in tic]
    tic_show = [np.round(x / params["sampling_rate"], 2) for x in tic]
    plt.yticks(tic, tic_show)

    ax.set_xlabel("Distance to Diagonal from Right")
    ax.set_ylabel("Window Start Time [s]")
    ax.grid(False)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
