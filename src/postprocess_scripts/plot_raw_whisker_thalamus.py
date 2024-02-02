"""
Copyright (c) 2020 Bahareh Tolooshams

plot raw data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
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
        "--raster-color",
        type=str,
        help="raster color",
        default="Black",
    )
    parser.add_argument(
        "--raster-markersize",
        type=int,
        help="raster markersize",
        default=0.2,
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

    # create datasets -------------------------------------------------------#
    dataset = datasetloader.DUNLdatasetwithRaster(params["data_path"][0])
    test_dataset = datasetloader.DUNLdatasetwithRaster(params["test_data_path"][0])

    # load stimulus from the numpy file
    numpy_data_path_first = params["data_path"][0].split("_kernellength")[0]
    numpy_data_path = f"{numpy_data_path_first}.npy"
    stimulus = torch.load(numpy_data_path)["stim"]

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
        "figures",
        "raw",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_raster = dataset.raster
    test_raster = test_dataset.raster

    num_neurons = train_raster.shape[1]
    for neuron_ctr in range(num_neurons):
        train_raster_neuron = train_raster[:, neuron_ctr]
        test_raster_neuron = test_raster[:, neuron_ctr]
        stimulus_neuron = stimulus[:, neuron_ctr]

        if 0:
            plot_whisker_psth(
                train_raster_neuron,
                test_raster_neuron,
                params,
                plot_filename=os.path.join(
                    out_path,
                    "psth_neuron{}.svg".format(
                        neuron_ctr,
                    ),
                ),
            )

            plot_whisker_raster(
                train_raster_neuron,
                test_raster_neuron,
                params,
                plot_filename=os.path.join(
                    out_path,
                    "raster_neuron{}.svg".format(
                        neuron_ctr,
                    ),
                ),
            )

        plot_whisker_raster_with_stimulus_zoomin(
            train_raster_neuron,
            stimulus_neuron,
            params,
            plot_filename=os.path.join(
                out_path,
                "stimulus_raster_zoomin_neuron{}.svg".format(
                    neuron_ctr,
                ),
            ),
        )

    print(f"plotting of raw is done. plots are saved at {out_path}")


def plot_whisker_psth(train_raster_neuron, test_raster_neuron, params, plot_filename):
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
    t = np.arange(0, trial_length)

    plt.subplot(2, 1, 1)
    plt.title(r"$\textbf{Train}$")
    plt.plot(
        t,
        torch.mean(train_raster_neuron, dim=0),
        lw=0.7,
        color=params["psth_color"],
    )
    plt.ylabel("PSTH", labelpad=0)

    plt.subplot(2, 1, 2)
    plt.title(r"$\textbf{Test}$")
    plt.plot(
        t,
        torch.mean(test_raster_neuron, dim=0),
        lw=0.7,
        color=params["psth_color"],
    )
    xtic = np.array([0, 0.25, 0.5, 0.75, 1]) * trial_length
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.ylabel("PSTH", labelpad=0)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    # plt.show()
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()


def plot_whisker_raster(train_raster_neuron, test_raster_neuron, params, plot_filename):
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
    train_raster_neuron_y, train_raster_neuron_x = np.where(train_raster_neuron != 0)
    test_raster_neuron_y, test_raster_neuron_x = np.where(test_raster_neuron != 0)

    plt.subplot(2, 1, 1)
    plt.title(r"$\textbf{Train}$")
    plt.scatter(
        train_raster_neuron_x,
        train_raster_neuron_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.ylabel("Trials", labelpad=0)

    plt.subplot(2, 1, 2)
    plt.title(r"$\textbf{Test}$")
    plt.scatter(
        test_raster_neuron_x,
        test_raster_neuron_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    xtic = np.array([0, 0.25, 0.5, 0.75, 1]) * trial_length
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.ylabel("Trials", labelpad=0)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    # plt.show()
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()


def plot_whisker_raster_with_stimulus_zoomin(
    raster_neuron, stimulus_neuron, params, plot_filename
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

    fig, axn = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(2, 3))

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    time_range_a = 500
    time_range_b = 1000

    raster_neuron = raster_neuron[:, time_range_a:time_range_b]
    stimulus_neuron = stimulus_neuron[:, time_range_a:time_range_b]

    trial_length = raster_neuron.shape[-1]
    raster_neuron_y, raster_neuron_x = np.where(raster_neuron != 0)

    plt.subplot(2, 1, 1)
    plt.scatter(
        raster_neuron_x,
        raster_neuron_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.ylabel("Trials", labelpad=0)

    plt.subplot(2, 1, 2)
    plt.plot(
        stimulus_neuron[0],
        color="cyan",
    )
    xtic = np.array([0, 0.5, 1]) * trial_length
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.ylabel("Stimulus", labelpad=0)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()


if __name__ == "__main__":
    main()
