"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
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
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.0, 0.3, 0.5, 1.2, 2.5, 5.0, 8.0, 11.0],
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "black",
            "orange",
            "blue",
            "red",
            "green",
        ],  # cue reg, cue exp, 3 rewards
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(12, 2),
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
        "codes_vs_reward",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # load data -------------------------------------------------------#

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

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
        expected_rew_amount = list()
        surprise_rew_amount = list()

        for i in range(num_trials):
            xi = x[i]
            xihat = xhat[i]
            labeli = label[i]

            if labeli == 0:
                continue
            else:
                # reward presence
                cue_flag = torch.sum(torch.abs(xi[1]), dim=-1).item()
                if cue_flag:
                    # expected trial
                    expected_rew_amount.append(labeli)
                    code_expected.append(torch.sum(xihat, dim=-1))
                else:
                    # surprise trial
                    surprise_rew_amount.append(labeli)
                    code_surprise.append(torch.sum(xihat, dim=-1))

        code_expected = torch.stack(code_expected, dim=0)
        code_surprise = torch.stack(code_surprise, dim=0)
        expected_rew_amount = torch.stack(expected_rew_amount, dim=0)
        surprise_rew_amount = torch.stack(surprise_rew_amount, dim=0)

        for neuron_ctr in range(num_neurons):
            out_path_name = os.path.join(
                out_path,
                "codes_vs_reward_{}_neuron{}.svg".format(datafile_name, neuron_ctr),
            )

            plot_code_vs_rew_single_neuron(
                code_expected[:, neuron_ctr],
                expected_rew_amount,
                code_surprise[:, neuron_ctr],
                surprise_rew_amount,
                params,
                out_path_name,
            )


def plot_code_vs_rew_single_neuron(
    code_expected_neuron,
    expected_rew_amount,
    code_surprise_neuron,
    surprise_rew_amount,
    params,
    out_path_name,
):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    markersize = 4
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

    fig, axn = plt.subplots(1, 4, sharex=True, sharey=False, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for kernel_ctr in range(1, params["kernel_num"]):
        (
            _,
            expected_rew_amount_linreg,
            code_expected_neuron_linreg,
        ) = utils.perform_sorted_regression(
            code_expected_neuron[:, kernel_ctr],
            expected_rew_amount,
            mode="linreg",
            y_fit=params["reward_amount_list"][1:],
        )
        (
            _,
            surprise_rew_amount_linreg,
            code_surprise_neuron_linreg,
        ) = utils.perform_sorted_regression(
            code_surprise_neuron[:, kernel_ctr],
            surprise_rew_amount,
            mode="linreg",
            y_fit=params["reward_amount_list"][1:],
        )
        plt.subplot(1, params["kernel_num"] - 1, kernel_ctr)

        if kernel_ctr != 1:
            plt.plot(
                surprise_rew_amount_linreg,
                code_surprise_neuron_linreg,
                color=params["color_list"][kernel_ctr],
                label="Surprise",
                lw=lw,
            )
            plt.scatter(
                surprise_rew_amount,
                code_surprise_neuron[:, kernel_ctr],
                marker=".",
                color=params["color_list"][kernel_ctr],
                s=markersize,
            )

        plt.plot(
            expected_rew_amount_linreg,
            code_expected_neuron_linreg,
            "--",
            color=params["color_list"][kernel_ctr],
            label="Expected",
            lw=lw,
        )
        plt.scatter(
            expected_rew_amount,
            code_expected_neuron[:, kernel_ctr],
            marker=".",
            color="gray",
            s=markersize,
        )

        if kernel_ctr == 1:
            plt.title(r"$\textbf{Cue\ Expected}$")
        elif kernel_ctr == 2:
            plt.title(r"$\textbf{Reward}$")
        elif kernel_ctr == 3:
            plt.title(r"$\textbf{Reward\ Coupled } +$")
        else:
            plt.title(r"$\textbf{Reward\ Coupled } -$")

        if kernel_ctr == 4:
            plt.legend()
        if kernel_ctr == 1:
            plt.ylabel("Code", labelpad=0)

        plt.xlabel("Reward amount [$\mu l$]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
