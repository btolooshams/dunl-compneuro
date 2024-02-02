"""
Copyright (c) 2020 Bahareh Tolooshams

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
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.0, 0.3, 0.5, 1.2, 2.5, 5.0, 8.0, 11.0],
    )
    parser.add_argument(
        "--num-neurons-to-plot",
        type=int,
        help="number of neurons to plot",
        default=56,
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(13, 3),
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

    code_expected_avg_from_all_datasets = list()
    code_surprise_avg_from_all_datasets = list()

    zc_list_from_all_datasets = list()

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

        zc_list = list()

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

        ### -------------------------------------------------------------------
        # #### this is calculate zero-crossing

        rpe_code_exp = torch.sum(code_expected[:, :, 3:], dim=-1)
        # rpe_code_sur = torch.sum(code_surprise[:,:,3:], dim=-1)

        # rpe_code = torch.cat((rpe_code_exp, rpe_code_sur), dim=0)
        # reward_for_rpe = torch.cat((expected_rew_amount, surprise_rew_amount), dim=0)
        rpe_code = rpe_code_exp
        reward_for_rpe = expected_rew_amount

        for neuron in range(rpe_code.shape[1]):
            rpe_code_neuron = rpe_code[:, neuron]

            zc = utils.compute_zc(
                rpe_code_neuron, reward_for_rpe, params["reward_amount_list"]
            )

            zc_list.append(zc)

        zc_list_from_all_datasets.append(torch.unsqueeze(torch.tensor(zc_list), dim=1))

        # ### -------------------------------------------------------------------

        # take the mean across trials
        code_expected_avg = list()
        code_surprise_avg = list()
        # (reward level, neurons, kernels)
        for reward in params["reward_amount_list"][1:]:
            code_expected_avg.append(
                torch.mean(
                    code_expected[np.where(expected_rew_amount == reward)], dim=0
                )
            )
            code_surprise_avg.append(
                torch.mean(
                    code_surprise[np.where(surprise_rew_amount == reward)], dim=0
                )
            )
        code_expected_avg = torch.stack(code_expected_avg, dim=0)
        code_surprise_avg = torch.stack(code_surprise_avg, dim=0)

        code_expected_avg_from_all_datasets.append(code_expected_avg)
        code_surprise_avg_from_all_datasets.append(code_surprise_avg)

    # cat across neurons
    code_expected_avg_from_all_datasets = torch.cat(
        code_expected_avg_from_all_datasets, dim=1
    )
    code_surprise_avg_from_all_datasets = torch.cat(
        code_surprise_avg_from_all_datasets, dim=1
    )

    zc_list_from_all_datasets = torch.cat(zc_list_from_all_datasets, dim=0)

    zc_list_from_all_datasets = torch.squeeze(zc_list_from_all_datasets, dim=1).numpy()

    for normalize in [True, False]:
        plot_code_vs_rew_all_neuron(
            code_expected_avg_from_all_datasets,
            code_surprise_avg_from_all_datasets,
            params["reward_amount_list"][1:],
            params,
            out_path,
            name="exp",
            normalize=normalize,
            color_mean=["black", "red"],
            zc=zc_list_from_all_datasets,
        )

        plot_code_vs_rew_all_neuron(
            code_surprise_avg_from_all_datasets,
            code_expected_avg_from_all_datasets,
            params["reward_amount_list"][1:],
            params,
            out_path,
            name="sur",
            normalize=normalize,
            color_mean=["red", "black"],
            zc=zc_list_from_all_datasets,
        )


def plot_code_vs_rew_all_neuron(
    x,
    x_other,
    reward_amount_list,
    params,
    out_path,
    name="",
    normalize=False,
    color_mean=["black", "black"],
    zc=None,
):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    fontfamily = "sans-serif"
    lw = 0.3

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
    cmap = mpl.colormaps["plasma"]
    cmap = cmap.reversed()

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    num_neurons = x.shape[1]

    print("there are total of {} neurons".format(num_neurons))
    print(
        "{} neurons are included in the plot. Those with highest curvature".format(
            params["num_neurons_to_plot"]
        )
    )

    x_curr_list = list()
    x_curr_other_list = list()
    x_min_list = list()
    x_max_list = list()
    for code_ctr in range(1, params["kernel_num"]):
        if code_ctr == 1:
            x_curr = x[:, :, 1]
            x_curr_other = x_other[:, :, 1]
        elif code_ctr == 2:
            x_curr = x[:, :, 2]
            x_curr_other = x_other[:, :, 2]
        elif code_ctr == 3:
            x_curr = x[:, :, 3] + x[:, :, 4]
            x_curr_other = x_other[:, :, 3] + x_other[:, :, 4]
        else:
            x_curr = x[:, :, 2] + x[:, :, 3] + x[:, :, 4]
            x_curr_other = x_other[:, :, 2] + x_other[:, :, 3] + x_other[:, :, 4]

        x_curr_list.append(x_curr)
        x_curr_other_list.append(x_curr_other)

        x_min_list.append(torch.min(x_curr, dim=0, keepdim=True)[0])
        x_max_list.append(torch.max(x_curr, dim=0, keepdim=True)[0])

        x_min_list.append(torch.min(x_curr_other, dim=0, keepdim=True)[0])
        x_max_list.append(torch.max(x_curr_other, dim=0, keepdim=True)[0])

    global_min = torch.min(torch.cat(x_min_list), dim=0, keepdim=False)[0]
    global_max = torch.max(torch.cat(x_max_list), dim=0, keepdim=False)[0]

    zc_sort_ids = np.flip(np.argsort(zc))
    for code_ctr in range(1, params["kernel_num"]):
        x_curr = x_curr_list[code_ctr - 1]
        x_curr_other = x_curr_other_list[code_ctr - 1]

        plt.subplot(1, params["kernel_num"] - 1, code_ctr)

        if code_ctr == 1:
            plt.title(r"$\textbf{Cue\ Expected}$")
        elif code_ctr == 2:
            plt.title(r"$\textbf{Reward}$")
        elif code_ctr == 3:
            plt.title(r"$\textbf{Reward\ Coupled\ Combined}$")
        else:
            plt.title(r"$\textbf{Reward\ Combined} $")

        if normalize:
            x_sameminmax = (x_curr - global_min) / (global_max - global_min)
            x_sameminmax_other = (x_curr_other - global_min) / (global_max - global_min)
        else:
            x_sameminmax = x_curr
            x_sameminmax_other = x_curr_other

        for i in range(params["num_neurons_to_plot"]):
            plt.plot(
                reward_amount_list,
                x_sameminmax[:, zc_sort_ids[i]],
                linewidth=lw,
                color=cmap(i / params["num_neurons_to_plot"]),
            )

        if color_mean[0] == "black":
            plt.plot(
                reward_amount_list,
                torch.mean(x_sameminmax, dim=-1),
                linewidth=2,
                color=color_mean[0],
            )

            plt.plot(
                reward_amount_list,
                torch.mean(x_sameminmax_other, dim=-1),
                "--",
                linewidth=2,
                color=color_mean[1],
            )
        else:
            plt.plot(
                reward_amount_list,
                torch.mean(x_sameminmax, dim=-1),
                "--",
                linewidth=2,
                color=color_mean[0],
            )

            plt.plot(
                reward_amount_list,
                torch.mean(x_sameminmax_other, dim=-1),
                linewidth=2,
                color=color_mean[1],
            )

        if code_ctr == 1:
            plt.ylabel("Code", labelpad=0)

        if code_ctr == 2:
            plt.xlabel("Reward amount [$\mu l$]", labelpad=0)

        xtic = np.array(
            [
                reward_amount_list[0],
                reward_amount_list[-3],
                reward_amount_list[-2],
                reward_amount_list[-1],
            ]
        )
        xtic = [x for x in xtic]
        plt.xticks(xtic, xtic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path,
            "codes_vs_reward_curves_{}_normalize{}.svg".format(name, normalize),
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
