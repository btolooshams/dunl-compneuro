"""
Copyright (c) 2025 Bahareh Tolooshams

plot recompose data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy as sp

import sys

sys.path.append("../dunl/")


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        # default="../results/local_raised_cosine_5_bases_res25ms",
        # default="../results/local_raised_cosine_2_bases_res25ms",
        # default="../results/local_raised_nonlin_cosine_5_bases_res25ms",
        # default="../results/local_raised_nonlin_cosine_2_bases_res25ms",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="../data/dopamine-spiking-eshel-uchida/neuralgml_matlab/dopamine_eshel_reward_onset.mat",
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "darkred",
            "peru",
            "yellowgreen",
            "mediumaquamarine",
            "cornflowerblue",
            "mediumblue",
            "navy",
        ],
    )
    parser.add_argument(
        "--color-dec-list",
        type=list,
        help="color decomposition list",
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
        default=(4, 2),
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Predict for Neural GLM.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    print(params["res_path"])

    if params["res_path"] == "../results/local_raised_cosine_5_bases_res25ms":
        params["bases_num"] = 6
    elif params["res_path"] == "../results/local_raised_cosine_2_bases_res25ms":
        params["bases_num"] = 3
    elif params["res_path"] == "../results/local_raised_nonlin_cosine_5_bases_res25ms":
        params["bases_num"] = 6
    elif params["res_path"] == "../results/local_raised_nonlin_cosine_2_bases_res25ms":
        params["bases_num"] = 3

    params["time_bin_resolution"] = 25

    data = sio.loadmat(params["data_path"])
    # print(data)

    y_from_data = data["y"]
    label_data = np.squeeze(data["label"])
    neuron_data = np.squeeze(data["neuron"])
    print("y_from_data", y_from_data.shape)
    print("label_data", label_data.shape)
    print("neuron_data", neuron_data.shape)

    num_neurons = 40

    res_path = params["res_path"]

    out_path = os.path.join(res_path, "figures/rec")
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
    num_trials = y_from_data.shape[0]

    num_bases = params["bases_num"]
    code_exp_corr = np.zeros((num_neurons, num_bases))
    code_sur_corr = np.zeros((num_neurons, num_bases))
    for neuron_ctr in range(num_neurons):
        y_surprise = list()
        y_expected = list()

        code_surprise = list()
        code_expected = list()

        yhat_surprise = list()
        yhat_expected = list()

        y_surprise_rew_amount = list()
        y_expected_rew_amount = list()

        for trial_ctr in range(num_trials):
            if neuron_data[trial_ctr] != neuron_ctr:
                continue

            filename = os.path.join(res_path, f"res_trial_{trial_ctr+1}.mat")
            data_mat = sio.loadmat(filename)

            y = data_mat["y"].astype(np.float32).todense()
            rate_hat = data_mat["rate_hat"].astype(np.float32).todense()
            trial_label = data_mat["trial_label"][0, 0]
            trial_type = data_mat["trial_type"][0, 0]
            wml = data_mat["wml"].todense()
            time_bin_resolution = data_mat["binSize"]
            x_matrix = data_mat["x_matrix"].todense()

            # print(trial_label, label_data[trial_ctr])

            # print("y", y.shape)
            # print("rate_hat", rate_hat.shape)
            # print("trial_label", trial_label.shape)
            # print("trial_type", trial_type.shape)
            # print("wml", wml.shape)
            # print("x_matrix", x_matrix.shape)
            # print("time_bin_resolution", time_bin_resolution)

            # print(abs(trial_label), label_data[trial_ctr])

            if abs(trial_label - label_data[trial_ctr]) > 0:
                print("Error: mismatch in trials")
                exit()

            if trial_label > 0:  # 1 exp, -1 sur
                y_expected.append(y)
                yhat_expected.append(rate_hat)
                y_expected_rew_amount.append(abs(trial_label))
                code_expected.append(wml)
            else:
                y_surprise.append(y)
                yhat_surprise.append(rate_hat)
                y_surprise_rew_amount.append(abs(trial_label))
                code_surprise.append(wml)

        code_surprise = np.concatenate(code_surprise, axis=1)
        code_expected = np.concatenate(code_expected, axis=1)

        # print(np.mean(code_surprise), np.mean(code_surprise))

        y_expected = np.concatenate(y_expected, axis=1)
        y_surprise = np.concatenate(y_surprise, axis=1)
        yhat_expected = np.concatenate(yhat_expected, axis=1)
        yhat_surprise = np.concatenate(yhat_surprise, axis=1)
        y_expected_rew_amount = np.stack(y_expected_rew_amount, axis=0)
        y_surprise_rew_amount = np.stack(y_surprise_rew_amount, axis=0)
        # print(np.max(y_surprise_rew_amount), np.max(y_expected_rew_amount))

        for code_ctr in range(num_bases):
            # for each
            (
                code_exp_corr_neuron_curr_code,
                code_exp_pvalue,
            ) = sp.stats.spearmanr(code_expected[code_ctr, :].T, y_expected_rew_amount)
            (
                code_sur_corr_neuron_curr_code,
                code_sur_pvalue,
            ) = sp.stats.spearmanr(code_surprise[code_ctr, :].T, y_surprise_rew_amount)

            code_exp_corr[neuron_ctr, code_ctr] = code_exp_corr_neuron_curr_code
            code_sur_corr[neuron_ctr, code_ctr] = code_sur_corr_neuron_curr_code

        ##############
        # plot the rec -------------------------------------------------------#
        y_surprise_neuron = y_surprise
        y_expected_neuron = y_expected

        yhat_surprise_neuron = yhat_surprise
        yhat_expected_neuron = yhat_expected

        fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)

        for ax in axn.flat:
            ax.tick_params(axis="x", direction="out")
            ax.tick_params(axis="y", direction="out")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        plt.subplot(2, 1, 1)
        plt.title(r"$\textbf{Surprise\ Trials}$")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
        ctr = -1
        for reward in params["reward_amount_list"]:
            ctr += 1
            indices = np.where(y_surprise_rew_amount == reward)
            plt.plot(
                np.mean(y_surprise_neuron[:, indices], axis=2).T,
                color=params["color_list"][ctr],
                lw=0.7,
                label="{}".format(reward),
            )
            plt.plot(
                np.mean(yhat_surprise_neuron[:, indices], axis=2).T,
                "--",
                color=params["color_list"][ctr],
                lw=0.7,
            )
            plt.legend(
                loc="upper right",
                ncol=4,
                borderpad=0.1,
                labelspacing=0.2,
                handletextpad=0.4,
                columnspacing=0.2,
            )

        plt.subplot(2, 1, 2)
        plt.title(r"$\textbf{Expected\ Trials}$")
        ctr = -1
        for reward in params["reward_amount_list"]:
            ctr += 1
            indices = np.where(y_expected_rew_amount == reward)

            plt.plot(
                np.mean(y_expected_neuron[:, indices], axis=2).T,
                color=params["color_list"][ctr],
                lw=0.7,
            )
            plt.plot(
                np.mean(yhat_expected_neuron[:, indices], axis=2).T,
                "--",
                color=params["color_list"][ctr],
                lw=0.7,
            )

        xtic = np.array([0, 0.5, 1, 1.5, 2]) * 12
        xtic_figure = [int(x * params["time_bin_resolution"]) for x in xtic]
        plt.xticks(xtic, xtic_figure)
        plt.xlabel("Time [ms]", labelpad=0)

        fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
        datafile_name = filename.split("/")[-1].split(".mat")[0]
        plt.savefig(
            os.path.join(out_path, "rec_{}_neuron.svg".format(neuron_ctr)),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close()


if __name__ == "__main__":
    main()
