"""
Copyright (c) 2020 Bahareh Tolooshams

plot rec data kernel

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

import sys

sys.path.append("../src/")

import utils, datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-list",
        type=str,
        help="res path list",
        default=["../results"],
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 2),
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

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "01noise_kernel_err",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    epoch_type = "best_val"

    res_folder = "../results"
    filename_list = os.listdir(res_folder)
    filename_list = [f"{x}" for x in filename_list if "20unrolling" in x]
    filename_list = [f"{x}" for x in filename_list if "01noise" in x]
    filename_list = [f"{x}" for x in filename_list if "forunrolling" not in x]
    res_path_list = [f"{res_folder}/{x}" for x in filename_list if "orth_" in x]

    data_folder = "../data/local-orthkernels-simulated"

    filename_list = os.listdir(data_folder)
    filename_list = [f"{x}" for x in filename_list if "test" in x]
    filename_list = [f"{x}" for x in filename_list if "testvis" not in x]
    data_path_list = [f"{data_folder}/{x}" for x in filename_list if ".pt" in x]

    ###################################

    dataset_list = list()
    dataloader_list = list()
    for data_path_cur in data_path_list:
        print(data_path_cur)
        dataset = datasetloader.DUNLdataset(data_path_cur)
        dataset_list.append(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=5,
            num_workers=1,
        )
        dataloader_list.append(dataloader)

    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

    kernels_dict = dict()
    for res_path in res_path_list:
        noise_level = float(res_path.split("01noise")[-1].split("_")[0])

        print(noise_level)
        if f"noise{noise_level}" not in kernels_dict.keys():
            kernels_dict[f"noise{noise_level}"] = list()

        # model
        model_path = os.path.join(
            res_path,
            "model",
            f"model_{epoch_type}.pt",
        )
        net = torch.load(model_path, map_location=device)
        net.to(device)
        net.eval()
        kernels = net.get_param("H").clone().detach()
        kernels = np.squeeze(kernels.cpu().numpy())

        # take parameters from the result path
        params = pickle.load(open(os.path.join(res_path, "params.pickle"), "rb"))

        outname = res_path.split("/")[-1]

        kernels_dict[f"noise{noise_level}"].append(kernels)

    ################################
    num_noise_list_global = list()
    ker_err_list_global = list()
    for noise_level_str in kernels_dict.keys():
        noise_level = float(noise_level_str.split("oise")[-1])

        ker_err_list = list()
        num_noise_list = list()

        min_corr_list = list()
        for cur in range(len(kernels_dict[f"noise{noise_level}"])):
            cur_kernel = kernels_dict[f"noise{noise_level}"][cur]

            # print(last_kernel.shape, cur_kernel.shape)]
            kTk = np.zeros((cur_kernel.shape[0], cur_kernel.shape[0]))
            for kernel_index in range(cur_kernel.shape[0]):
                for kernel_index_true in range(kernels_true.shape[0]):
                    corrr = np.correlate(
                        cur_kernel[kernel_index],
                        kernels_true[kernel_index_true],
                        mode="full",
                    )
                    kTk[kernel_index, kernel_index_true] = np.max(np.abs(corrr))

            # print(kTk)

            # exit()
            kTk = np.max(kTk, axis=0)  # taking the best corresponding kernel
            # rm the last column of zero
            ker_err = kTk
            ker_err = np.mean(ker_err[:-1])

            ker_err_list.append(ker_err)
            num_noise_list.append(noise_level)

        num_noise_list_global.append(np.array(num_noise_list))
        ker_err_list_global.append(np.array(ker_err_list))

    num_noise_list_global = np.stack(num_noise_list_global).T
    ker_err_list_global = np.stack(ker_err_list_global).T
    print(num_noise_list_global.shape)
    print(num_noise_list_global.shape)
    outname = f"noisy_kernelerr_with_errbar.png"
    plot_kernelerr_wih_errbar(
        ker_err_list_global, num_noise_list_global, out_path, outname
    )
    outname = f"noisy_kernelerr_with_errbar.svg"
    plot_kernelerr_wih_errbar(
        ker_err_list_global, num_noise_list_global, out_path, outname
    )


def plot_kernelerr_wih_errbar(k_list, num_noise_list, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"
    lw_true = 2
    markersize = 5

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
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

    row = 1
    col = 1
    fig, ax = plt.subplots(row, col, sharex=True, sharey=True, figsize=(4, 3))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(row, col, 1)

    plt.axhline(0.5, color="gray", linewidth=0.5)

    sorted_indices = np.argsort(np.mean(num_noise_list, axis=0))
    ax.errorbar(
        np.mean(num_noise_list, axis=0)[sorted_indices],
        np.mean(k_list, axis=0)[sorted_indices],
        yerr=np.std(k_list, axis=0)[sorted_indices],
        color="black",
    )

    plt.xticks(
        np.mean(num_noise_list, axis=0)[sorted_indices[1:]],
        np.mean(num_noise_list, axis=0)[sorted_indices[1:]],
    )
    plt.ylabel("Mean Correlation(True, Est)", labelpad=0)
    plt.xlabel("Noise Level", labelpad=0)
    plt.ylim([0, 1])

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
