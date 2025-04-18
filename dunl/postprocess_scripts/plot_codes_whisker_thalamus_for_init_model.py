"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

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

sys.path.append("../dunl/")

import model, datasetloader, utils


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
        "--onset-color",
        type=str,
        help="onset color",
        default="Gray",
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

    parser.add_argument(
        "--raw-color",
        type=str,
        help="raw color",
        default="Black",
    )
    parser.add_argument(
        "--rec-color",
        type=str,
        help="rec color",
        default="Brown",
    )
    parser.add_argument(
        "--smoothing-tau",
        type=int,
        help="smoothing tau",
        default=20,
    )
    parser.add_argument(
        "--infer-init-model-code-sparse-regularization-list",
        type=list,
        help="list of lam to infer the init model",
        default=[0, 0.005],
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

    kernel_init = torch.load(params["kernel_initialization"])

    out_path = os.path.join(
        params["res_path"],
        "figures",
        "codes",
    )
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

    # load model ------------------------------------------------------#
    # initial model before training
    net_init = model.DUNL1D(params, kernel_init)
    net_init.eval()

    # for code_topk_sparse in [18, 24, 32]:
    #     net_init.code_topk_sparse = code_topk_sparse
    for code_supp in [True]:
        code_topk_sparse = 32
        net_init.code_topk_sparse = code_topk_sparse
        params["code_supp"] = code_supp
        net_init.code_supp = code_supp
        net_init.code_topk = False

        for code_sparse_regularization in params[
            "infer_init_model_code_sparse_regularization_list"
        ]:
            net_init.code_sparse_regularization = torch.tensor(
                code_sparse_regularization, device=device
            )

            # go over data -------------------------------------------------------#

            num_neurons = dataset.raster.shape[1]

            for neuron_ctr in range(num_neurons):
                print(f"plot for neuron {neuron_ctr}")

                print("only run for neuron 4")
                if neuron_ctr != 4:
                    continue

                test_raster_neuron = test_dataset.raster[:, [neuron_ctr]]

                test_smooth_raster = utils.smooth_raster(
                    test_raster_neuron, params["smoothing_tau"]
                )

                neuron_out_path = os.path.join(out_path, "neuron{}".format(neuron_ctr))
                if not os.path.exists(neuron_out_path):
                    os.makedirs(neuron_out_path)

                test_x_init, test_xhat_init, test_yhat_init = get_xxhat(
                    net_init, dataset, params, neuron_ctr
                )

                test_num_trials = len(test_dataset)

                for trial in range(test_num_trials):
                    plot_whisker_code(
                        test_x_init[trial, 0],
                        test_xhat_init[trial, 0],
                        params,
                        os.path.join(
                            neuron_out_path,
                            "code_test_neuron{}_trial{}_withinitkernel_reg{}_topk{}_code_supp{}.svg".format(
                                neuron_ctr,
                                trial,
                                code_sparse_regularization,
                                code_topk_sparse,
                                code_supp,
                            ),
                        ),
                    )

                    plot_whisker_code_and_raw(
                        test_x_init[trial, 0],
                        test_xhat_init[trial, 0],
                        test_raster_neuron[trial],
                        test_smooth_raster[trial, 0],
                        test_yhat_init[trial],
                        params,
                        os.path.join(
                            neuron_out_path,
                            "a_code_test_neuron{}_trial{}_smoothingtau{}_withinitkernel_reg{}_topk{}_code_supp{}.svg".format(
                                neuron_ctr,
                                trial,
                                params["smoothing_tau"],
                                code_sparse_regularization,
                                code_topk_sparse,
                                code_supp,
                            ),
                        ),
                    )

    print(f"plotting of rec is done. plots are saved at {out_path}")


def plot_whisker_code(
    x,
    xhat,
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

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(1, 1, 1)

    x_onsets = t[x > 0]
    plt.vlines(
        x_onsets,
        ymin=0,
        ymax=torch.max(xhat),
        color=params["onset_color"],
        lw=0.7,
        label="Onset",
    )
    if torch.sum(xhat):
        plt.stem(
            t[xhat > 0],
            xhat[xhat > 0],
            linefmt="r--",
            markerfmt="r.",
            basefmt=" ",
            label="Code",
        )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )
    plt.ylim(0)

    xtic = [0, 500, 1000, 1500, 2000, 2500, 3000]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_whisker_code_and_raw(
    x,
    xhat,
    raster,
    smooth_raster,
    yhat,
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

    x = torch.squeeze(x, dim=0)
    xhat = torch.squeeze(xhat, dim=0)

    trial_length = x.shape[-1]
    t = np.arange(0, trial_length) * params["time_bin_resolution"]
    t_raster = np.arange(0, smooth_raster.shape[-1])
    t_y = np.arange(0, yhat.shape[-1]) * params["time_bin_resolution"]

    fig, axn = plt.subplots(3, 1, sharex=True, sharey=False)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplot(3, 1, 1)
    raster_y, raster_x = np.where(raster != 0)
    plt.scatter(
        raster_x,
        raster_y,
        s=params["raster_markersize"],
        color=params["raster_color"],
    )
    plt.title(r"$\textbf{Rasters}$")

    plt.subplot(3, 1, 2)
    plt.plot(
        t_raster,
        smooth_raster,
        lw=0.7,
        color=params["raw_color"],
        label="smooth raster",
    )
    plt.plot(
        t_y,
        yhat,
        lw=0.7,
        color=params["rec_color"],
        label="rec",
    )
    plt.title(r"$\textbf{Raw}$")

    plt.subplot(3, 1, 3)

    x_onsets = t[x > 0]
    plt.vlines(
        x_onsets,
        ymin=0,
        ymax=torch.max(xhat),
        color=params["onset_color"],
        lw=0.7,
        label="Onset",
    )
    if torch.sum(xhat):
        plt.stem(
            t[xhat > 0],
            xhat[xhat > 0],
            linefmt="y--",
            markerfmt="y.",
            basefmt=" ",
            label="Code",
        )
    plt.legend(
        loc="upper right",
        ncol=1,
        borderpad=0.1,
        labelspacing=0.2,
        handletextpad=0.4,
        columnspacing=0.2,
    )
    plt.ylim(0)

    xtic = [0, 500, 1000, 1500, 2000, 2500, 3000]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.title(r"$\textbf{Code}$")
    plt.xlabel("Time [ms]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    # plt.show()
    # exit()
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def get_xxhat(net, dataset, params, neuron_ctr):
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

    return x, xhat.clone().detach(), torch.squeeze(yhat.clone().detach(), dim=1)


if __name__ == "__main__":
    main()
