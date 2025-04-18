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
        default="../results/xxx",
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
        default="Cyan",
    )
    parser.add_argument(
        "--smoothing-tau",
        type=int,
        help="smoothing tau",
        default=20,
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
    params["device"] = device

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
    dataset = datasetloader.DUNLdatasetwithRasterNoRate(params["data_path"][0])
    test_dataset = datasetloader.DUNLdatasetwithRasterNoRate(
        params["test_data_path"][0]
    )

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

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    # initial model before training
    net_init = model.DUNL1D(params, kernel_init)
    net_init.to(device)
    net_init.eval()

    # go over data -------------------------------------------------------#

    num_neurons = dataset.raster.shape[1]

    test_x, test_xhat, test_yhat = get_xxhat(net, test_dataset, params)

    plot_whisker_group_code_historgram(
        test_x,
        test_xhat,
        params,
        os.path.join(
            out_path,
            "group_code_test_histrogram.svg",
        ),
    )

    plot_whisker_group_code_historgram(
        test_x,
        test_xhat,
        params,
        os.path.join(
            out_path,
            "group_code_test_histrogram.png",
        ),
    )


def plot_whisker_group_code_historgram(
    x,
    xhat,
    params,
    plot_filename,
):

    x = x.cpu()
    xhat = xhat.cpu()

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

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
        }
    )

    row = 2
    col = 5
    fig, axn = plt.subplots(row, col, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    a = torch.max(xhat).item()

    for ctr in range(10):
        plt.subplot(row, col, ctr + 1)

        code = x[:, 0].detach().cpu().numpy()
        code_hat_tmp = xhat[:, ctr].detach().cpu().numpy()

        code_hat = code_hat_tmp[np.where(code_hat_tmp > 0)]
        hist, bin_edges = np.histogram(
            code_hat.flatten(),
            bins=20,
            range=(0, a),
            density=True,
        )
        neuron_id = ctr + 1
        plt.hist(code_hat.flatten(), bins=bin_edges, density=True, color="gray")
        plt.title(f"Neuron {neuron_id}", fontsize=8)

        if ctr > 4:
            plt.xlabel("Codes", labelpad=0)
        if ctr == 0 or ctr == 5:
            plt.ylabel("Histogram", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        plot_filename,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def get_xxhat(net, dataset, params):

    y = dataset.y
    x = dataset.x
    a = dataset.a

    y = y.unsqueeze(dim=2).to(params["device"])
    x = x.unsqueeze(dim=2).to(params["device"])
    a = a.unsqueeze(dim=2).to(params["device"])

    if params["code_supp"]:
        x_code_supp = x
    else:
        x_code_supp = None

    print(a.shape, x.shape, y.shape)

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
