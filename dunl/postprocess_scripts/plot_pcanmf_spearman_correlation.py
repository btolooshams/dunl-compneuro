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


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_14_12_37_30",
        # default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
    )
    parser.add_argument(
        "--num-comp",
        type=int,
        help="number of components",
        default=2,
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "blue",
            "red",
            "green",
        ],
    )
    parser.add_argument(
        "--method",
        type=str,
        help="nmf or pca",
        default="pca",
    )
    parser.add_argument(
        "--plot-only-sur",
        type=bool,
        help="plot only surprise trials",
        default=True,
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(4, 2),
        # default=(6, 2),
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

    # get transform -------------------------------------------------------#
    method_transform = pickle.load(
        open(
            os.path.join(
                postprocess_path,
                "{}_transform_{}.pkl".format(params["method"], params["num_comp"]),
            ),
            "rb",
        )
    )
    scaler_transform = pickle.load(
        open(os.path.join(postprocess_path, "scaler_transform.pkl"), "rb")
    )

    # load data -------------------------------------------------------#
    x_corr_from_all_datasets = list()
    y_corr_from_all_datasets = list()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        if params["plot_only_sur"]:
            # (neuron, time, trials)
            y = np.load(
                os.path.join(
                    postprocess_path,
                    "y_for_pcanmf_{}_only_sur.npy".format(datafile_name),
                )
            )
            label = np.load(
                os.path.join(
                    postprocess_path,
                    "label_for_pcanmf_{}_only_sur.npy".format(datafile_name),
                )
            )
        else:
            # (neuron, time, trials)
            y = np.load(
                os.path.join(
                    postprocess_path, "y_for_pcanmf_{}.npy".format(datafile_name)
                )
            )
            label = np.load(
                os.path.join(
                    postprocess_path, "label_for_pcanmf_{}.npy".format(datafile_name)
                )
            )

        print(y.shape)

        y = np.transpose(y, (0, 2, 1))

        num_neurons = y.shape[0]

        x_corr = torch.zeros(num_neurons, params["num_comp"])
        y_corr = torch.zeros(num_neurons)

        for neuron_ctr in range(num_neurons):
            y_neuron = y[neuron_ctr]

            y_corr_neuron, _ = sp.stats.spearmanr(np.sum(y_neuron, axis=-1), label)
            y_corr[neuron_ctr] = y_corr_neuron

            if params["method"] == "pca":
                y_neuron_ready = scaler_transform.transform(y_neuron)
            elif params["method"] == "nmf":
                y_neuron_ready = y_neuron - np.min(y_neuron)

            x_neuron = method_transform.transform(y_neuron_ready)

            for code_ctr in range(params["num_comp"]):
                x_corr_curr, _ = sp.stats.spearmanr(x_neuron[:, code_ctr], label)

                x_corr[neuron_ctr, code_ctr] = x_corr_curr

        x_corr_from_all_datasets.append(x_corr)
        y_corr_from_all_datasets.append(y_corr)

    x_corr_from_all_datasets = np.concatenate(x_corr_from_all_datasets, axis=0)
    y_corr_from_all_datasets = np.concatenate(y_corr_from_all_datasets, axis=0)

    if params["plot_only_sur"]:
        out_path_name = os.path.join(
            out_path,
            "{}_spearman_correlation_{}_only_sur.svg".format(
                params["method"], params["num_comp"]
            ),
        )
    else:
        out_path_name = os.path.join(
            out_path,
            "{}_spearman_correlation_{}.svg".format(
                params["method"], params["num_comp"]
            ),
        )

    plot_spearman_correlation(
        x_corr_from_all_datasets,
        y_corr_from_all_datasets,
        params,
        out_path_name=out_path_name,
    )

    print(f"plotting of kernels is done. plots are saved at {out_path}")


def plot_spearman_correlation(x, y, params, out_path_name):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    markersize = 2

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

    fig, axn = plt.subplots(
        1, params["num_comp"], sharex=True, sharey=True, figsize=params["figsize"]
    )

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for code_ctr in range(params["num_comp"]):
        plt.subplot(1, params["num_comp"], code_ctr + 1)
        plt.scatter(
            x[:, code_ctr],
            y,
            marker=".",
            color=params["color_list"][code_ctr],
            s=markersize,
        )
        plt.plot([-0.5, 1], [-0.5, 1], "--", color="gray", linewidth=0.1)

        axn[code_ctr].set_xlabel("Spearman(Code, Reward)")

        if code_ctr == 0:
            axn[code_ctr].set_ylabel("Spearman(Raw, Reward)")

        plt.scatter(
            np.mean(x[:, code_ctr]),
            np.mean(y),
            marker="x",
            color="green",
            lw=1,
            s=markersize * 6,
        )

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
