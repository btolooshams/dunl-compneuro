"""
Copyright (c) 2025 Bahareh Tolooshams

plot code lowdim dopamine

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys

sys.path.append("../src/")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_14_12_37_30",
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
            "orange",
            "blue",
            "red",
        ],  # cue exp, 2 rewards
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(6, 4),
    )
    parser.add_argument(
        "--swap-code",
        type=bool,
        help="bool to swap code",
        default=True,
    )
    parser.add_argument(
        "--reg-mode",
        type=str,
        help="regression mode",
        # default="linreg",
        default="hillfit",
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

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

    # create folders -------------------------------------------------------#

    out_path = os.path.join(
        params["res_path"],
        "figures",
        "codes_vs_reward",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    ################################
    axes_fontsize = 20
    legend_fontsize = 20
    tick_fontsize = 20
    title_fontsize = 20
    fontfamily = "sans-serif"
    markersize = 0.1

    kcolor = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "pink",
        "brown",
        "olive",
        "purple",
        "gray",
    ]

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

    # plot new fig
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    code_combined_all = list()
    rew_amount_combined_all = list()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        label_int = torch.load(
            os.path.join(postprocess_path, "label_{}.pt".format(datafile_name))
        )

        label = label_int.clone()
        tmp_ctr = 0
        for reward in params["reward_amount_list"]:
            tmp_ctr += 1
            label[label == tmp_ctr] = reward

        num_trials = xhat.shape[0]
        num_neurons = xhat.shape[1]

        code_expected = list()
        code_surprise = list()
        expected_rew_amount = list()
        surprise_rew_amount = list()

        for i in range(num_trials):
            xi = x[i]
            xihat = xhat[i]
            labeli = label[i]

            # reward presence
            cue_flag = torch.sum(torch.abs(xi[0]), dim=-1).item()
            if cue_flag:
                # expected trial
                expected_rew_amount.append(labeli)
                code_expected.append(torch.sum(xihat, dim=-1))
            else:
                # surprise trial
                surprise_rew_amount.append(labeli)
                code_surprise.append(torch.sum(xihat, dim=-1))

        # (trials, 1, num_kernels)
        code_expected = torch.stack(code_expected, dim=0)
        code_surprise = torch.stack(code_surprise, dim=0)
        # (trials)
        expected_rew_amount = torch.stack(expected_rew_amount, dim=0)
        surprise_rew_amount = torch.stack(surprise_rew_amount, dim=0)

        if params["swap_code"]:
            code_expected = utils.swap_code(code_expected, 1, 2)
            code_surprise = utils.swap_code(code_surprise, 1, 2)

        code_combined = torch.cat([code_expected, code_surprise], dim=0)
        code_combined_all.append(code_combined)
        rew_amount_combined = torch.cat(
            [expected_rew_amount, surprise_rew_amount], dim=0
        )
        rew_amount_combined_all.append(rew_amount_combined)

        for c in range(len(params["reward_amount_list"])):
            cur_rew = params["reward_amount_list"][c]
            cluster_i = np.where(rew_amount_combined == cur_rew)[0]

            plt.scatter(
                torch.mean(code_combined[cluster_i, 0, 1]),
                torch.mean(code_combined[cluster_i, 0, 2]),
                color=kcolor[c],
                s=2,
                label=f"{cur_rew}",
            )
            plt.axis("equal")

    code_combined_all = torch.squeeze(torch.cat(code_combined_all, dim=0), dim=1)
    rew_amount_combined_all = torch.cat(rew_amount_combined_all, dim=0)
    print(code_combined_all.shape, rew_amount_combined_all.shape)

    n_comp = 2
    pca_transform = PCA(n_components=n_comp)
    code_pca_coeff = pca_transform.fit_transform(code_combined_all)

    plt.xlabel("K1")
    plt.ylabel("K2")

    ax.grid(False)
    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.show()
    plt.close()


def plot_code_vs_rew_single_neuron(
    fig,
    code_expected_neuron,
    expected_rew_amount,
    code_surprise_neuron,
    surprise_rew_amount,
    params,
    reg_mode="linreg",
):
    lw = 0.5

    for kernel_ctr in range(params["kernel_num"]):
        (
            _,
            expected_rew_amount_linreg,
            code_expected_neuron_linreg,
        ) = utils.perform_sorted_regression(
            code_expected_neuron[:, kernel_ctr],
            expected_rew_amount,
            mode=reg_mode,
            y_fit=params["reward_amount_list"],
        )
        if kernel_ctr != 0:
            (
                _,
                surprise_rew_amount_linreg,
                code_surprise_neuron_linreg,
            ) = utils.perform_sorted_regression(
                code_surprise_neuron[:, kernel_ctr],
                surprise_rew_amount,
                mode=reg_mode,
                y_fit=params["reward_amount_list"],
            )

        plt.subplot(2, params["kernel_num"], kernel_ctr + 1)

        if kernel_ctr != 0:
            if (
                surprise_rew_amount_linreg is not None
            ):  # this might be all 0 becasue of an errror in hilt
                plt.plot(
                    surprise_rew_amount_linreg,
                    code_surprise_neuron_linreg,
                    color=params["color_list"][kernel_ctr],
                    label="Surprise",
                    lw=lw,
                )

        if kernel_ctr == 0:
            plt.title(r"$\textbf{Cue}$")
        elif kernel_ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        elif kernel_ctr == 2:
            plt.title(r"$\textbf{Reward\ II}$")

        plt.subplot(2, params["kernel_num"], params["kernel_num"] + kernel_ctr + 1)
        if expected_rew_amount_linreg is not None:
            plt.plot(
                expected_rew_amount_linreg,
                code_expected_neuron_linreg,
                "--",
                color=params["color_list"][kernel_ctr],
                label="Expected",
                lw=lw,
            )

        if kernel_ctr == 2:
            pass
            # plt.legend()
        if kernel_ctr == 0:
            plt.ylabel("Code", labelpad=0)

        plt.xlabel("Reward amount [$\mu l$]", labelpad=0)

    return fig


if __name__ == "__main__":
    main()
