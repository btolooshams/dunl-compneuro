"""
Copyright (c) 2025 Bahareh Tolooshams

plot rec data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

import sys

sys.path.append("../src/")

import utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        ##### these are after initial submission
        default="../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_independentkernels_kernelsmoothing_0p0005_2024_05_14_11_15_13",
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
        default=(6, 2),
    )
    parser.add_argument(
        "--swap-kernel-list", type=bool, help="bool to swap kernels", default=[]
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

    neuron_file_list = os.listdir(params_init["res_path"])
    neuron_file_list = [x for x in neuron_file_list if "figure" not in x]
    neuron_file_list = [x for x in neuron_file_list if ".DS_Store" not in x]

    kernels_list = list()
    for neuron_file in neuron_file_list:
        neuron_folder = os.path.join(params_init["res_path"], neuron_file)

        # take parameters from the result path
        params = pickle.load(open(os.path.join(neuron_folder, "params.pickle"), "rb"))
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

        # set time bin resolution -----------------------------------------------#
        data_dict = torch.load(data_path_list[0])
        params["time_bin_resolution"] = data_dict["time_bin_resolution"]

        # create folders -------------------------------------------------------#
        model_path = os.path.join(
            neuron_folder,
            "model",
            "model_final.pt",
        )

        out_path = os.path.join(
            params["res_path"],
            "figures",
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # load model ------------------------------------------------------#
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()

        kernels = net.get_param("H").clone().detach()
        kernels = np.squeeze(kernels.cpu().numpy())

        name_path = os.path.join(out_path, f"kernel_{neuron_file}.svg")
        plot_kernel_individual(kernels, params, name_path)

        kernels_list.append(kernels)

    # (neurons, cue/rewardI/rewardII, time length
    kernels_list = np.stack(kernels_list, axis=0)

    cue_kernel = np.mean(kernels_list[:, 0, :], axis=0)

    reward_kernels = kernels_list[:, 1:].reshape(-1, 24)
    print(reward_kernels.shape)

    specclus = SpectralClustering(
        n_clusters=2,
        eigen_solver="arpack",
        affinity="rbf",
        assign_labels="kmeans",
        gamma=1,
    )
    cluster_ids = specclus.fit_predict(reward_kernels)

    salience_list = list()
    value_list = list()
    cue_list = list()
    for neuron in range(40):
        neuron_file = neuron_file_list[neuron]

        cue = kernels_list[neuron, 0]
        if cluster_ids[neuron * 2] == cluster_ids[neuron * 2 + 1]:
            print(neuron_file)
            salience = kernels_list[neuron, 1]
            value = kernels_list[neuron, 2]
        else:
            salience = kernels_list[neuron, cluster_ids[neuron * 2] + 1]
            value = kernels_list[neuron, cluster_ids[neuron * 2 + 1] + 1]

        salience_list.append(salience)
        value_list.append(value)

        cue_list.append(cue)

        kernels = np.stack([cue, salience, value])

        name_path = os.path.join(out_path, f"kernel_{neuron_file}_sorted.svg")
        plot_kernel_individual(kernels, params, name_path)

    name_path = os.path.join(out_path, f"0_kernel_all.svg")
    plot_kernel_all(cue_list, salience_list, value_list, params, name_path)


def plot_kernel_individual(kernels, params, name_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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

    fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0,
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    for ctr in range(params["kernel_num"]):
        plt.subplot(1, 3, ctr + 1)
        axn[ctr].axhline(0, color="gray", lw=0.3)

        plt.plot(t, kernels[ctr], color=params["color_list"][ctr])

        if ctr == 0:
            plt.title(r"$\textbf{Cue}$")
        elif ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        else:
            plt.title(r"$\textbf{Reward\ II}$")
        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)

        if ctr == 1:
            plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        name_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_kernel_all(cue_list, salience_list, value_list, params, name_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
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

    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0,
        params["kernel_length"] * params["time_bin_resolution"],
        params["kernel_length"],
    )

    for ctr in range(1, params["kernel_num"]):
        plt.subplot(1, 2, ctr)
        axn[ctr - 1].axhline(0, color="gray", lw=0.3)

        if ctr == 0:
            for i in range(len(cue_list)):
                plt.plot(t, cue_list[i], color="gray", linewidth=0.1)
            plt.plot(t, np.mean(cue_list, axis=0), color=params["color_list"][ctr])
        elif ctr == 1:
            for i in range(len(salience_list)):
                plt.plot(t, salience_list[i], color="gray", linewidth=0.1)
            plt.plot(t, np.mean(salience_list, axis=0), color=params["color_list"][ctr])
        else:
            for i in range(len(value_list)):
                plt.plot(t, value_list[i], color="gray", linewidth=0.1)
            plt.plot(t, np.mean(value_list, axis=0), color=params["color_list"][ctr])

        if ctr == 0:
            plt.title(r"$\textbf{Cue}$")
        elif ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        else:
            plt.title(r"$\textbf{Reward\ II}$")
        xtic = (
            np.array([0, 0.5, 1])
            * params["kernel_length"]
            * params["time_bin_resolution"]
        )
        xtic = [int(x) for x in xtic]
        plt.xticks(xtic, xtic)

        if ctr == 1:
            plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        name_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"plotting of kernels is done. plots are saved at {name_path}")


if __name__ == "__main__":
    main()
