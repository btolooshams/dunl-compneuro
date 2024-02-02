"""
Copyright (c) 2020 Bahareh Tolooshams

plot rec data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys

sys.path.append("../src/")

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/whisker_05msbinres_lamp03_topk18_smoothkernelp003_2023_07_19_00_03_18",
        # default="../results/whisker_05msbinres_lamp05_topk17_smoothkernelp003_2023_07_19_00_02_10",
        # default="../results/whisker_05msbinres_lamp05_topk18_smoothkernelp001_2023_07_18_23_32_40",
        # default="../results/whisker_05msbinres_lamp05_topk18_smoothkernelp003_2023_07_18_23_42_17",
        # default="../results/whisker_05msbinres_lamp05_topk18_smoothkernelp005_2023_07_18_23_44_26",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers for dataloader",
        default=4,
    )
    parser.add_argument(
        "--psth-color",
        type=str,
        help="psth color",
        default="Black",
    )
    parser.add_argument(
        "--rec-color",
        type=str,
        help="rec color",
        default="Cyan",
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

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

    # create datasets -------------------------------------------------------#
    dataset = datasetloader.DUNLdatasetwithRaster(params["data_path"][0])
    test_dataset = datasetloader.DUNLdatasetwithRaster(params["test_data_path"][0])

    # create folders -------------------------------------------------------#
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "figures",
        "rec",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    # go over data -------------------------------------------------------#

    train_raster = dataset.raster
    test_raster = test_dataset.raster

    num_neurons = train_raster.shape[1]
    for neuron_ctr in range(num_neurons):
        print(f"plot for neuron {neuron_ctr}")

        train_raster_neuron = train_raster[:, neuron_ctr]
        test_raster_neuron = test_raster[:, neuron_ctr]

        train_yhat = get_yhat(net, dataset, params, neuron_ctr)

        test_yhat = get_yhat(net, test_dataset, params, neuron_ctr)

        plot_whisker_rec(
            train_raster_neuron,
            train_yhat,
            test_raster_neuron,
            test_yhat,
            params,
            plot_filename=os.path.join(
                out_path,
                "rec_neuron{}.svg".format(
                    neuron_ctr,
                ),
            ),
        )

    print(f"plotting of rec is done. plots are saved at {out_path}")


def plot_whisker_rec(
    train_raster_neuron,
    train_yhat,
    test_raster_neuron,
    test_yhat,
    params,
    plot_filename,
):
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

    fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    trial_length = train_raster_neuron.shape[-1]
    t_raster = np.arange(0, trial_length)
    t_y = np.arange(0, train_yhat.shape[-1]) * params["time_bin_resolution"]

    plt.subplot(2, 1, 1)
    plt.title(r"$\textbf{Train}$")
    plt.plot(
        t_raster,
        torch.mean(train_raster_neuron, dim=0),
        lw=0.7,
        color=params["psth_color"],
        label="raw",
    )
    plt.plot(
        t_y,
        torch.mean(train_yhat, dim=0),
        lw=1,
        color=params["rec_color"],
        label="rec",
    )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )

    plt.subplot(2, 1, 2)
    plt.title(r"$\textbf{Test}$")
    plt.plot(
        t_raster,
        torch.mean(test_raster_neuron, dim=0),
        lw=0.7,
        color=params["psth_color"],
    )
    plt.plot(
        t_y,
        torch.mean(test_yhat, dim=0),
        lw=1,
        color=params["rec_color"],
    )
    xtic = np.array([0, 0.25, 0.5, 0.75, 1]) * trial_length
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    # plt.show()
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()


def get_yhat(net, dataset, params, neuron_ctr):
    y = dataset.y[:, [neuron_ctr]]
    x = dataset.x
    a = dataset.a[:, [neuron_ctr]]

    if params["code_supp"]:
        x_code_supp = x
    else:
        x_code_supp = None

    # forward encoder
    xhat, a_est = net.encode(y, a, x_code_supp)
    # forward decoder
    hxmu = net.decode(xhat, a_est)

    if params["model_distribution"] == "binomial":
        yhat = torch.sigmoid(hxmu)
    else:
        raise NotImplementedError("model distribution is not implemented")

    return torch.squeeze(yhat.clone().detach(), dim=1)


if __name__ == "__main__":
    main()
