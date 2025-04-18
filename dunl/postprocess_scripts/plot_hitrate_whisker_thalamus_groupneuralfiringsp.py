"""
Copyright (c) 2025 Bahareh Tolooshams

plot KS fit data

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

import utils


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
        "--psth-color",
        type=str,
        help="psth color",
        default="Black",
    )
    parser.add_argument(
        "--rec-color",
        type=str,
        help="rec color",
        default="Blue",
    )
    parser.add_argument(
        "--smoothing-tau-list",
        type=list,
        help="smoothing tau",
        default=[1, 10, 20, 30],
    )
    parser.add_argument(
        "--tol-list",
        type=list,
        help="tol list for hit rate",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--threshold-list",
        type=list,
        help="threshold list",
        default=list(np.linspace(0.025, 0.3, 20)),
    )
    parser.add_argument(
        "--init-model-code-sparse-regularization-list",
        type=list,
        help="list of lam to infer the init model",
        default=[0.03],
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

    # create folders -------------------------------------------------------#
    out_path = os.path.join(params["res_path"], "figures", "missfalse")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # create datasets -------------------------------------------------------#

    for init_reg in params["init_model_code_sparse_regularization_list"]:
        x_train, xhat_train, _, y_train = get_data(
            params["data_path"][0], postprocess_path, init_reg
        )
        x_test, xhat_test, _, y_test = get_data(
            params["test_data_path"][0], postprocess_path, init_reg
        )

        for smoothing_tau in params["smoothing_tau_list"]:
            for tol in params["tol_list"]:
                plot_miss_false(
                    y_train,
                    x_train,
                    xhat_train,
                    None,
                    tol,
                    smoothing_tau,
                    params,
                    out_path,
                    name="train_reg{}".format(init_reg),
                )
                plot_miss_false(
                    y_test,
                    x_test,
                    xhat_test,
                    None,
                    tol,
                    smoothing_tau,
                    params,
                    out_path,
                    name="test_reg{}".format(init_reg),
                )


def plot_miss_false(
    y, x, xhat, xhat_init, event_tol, smoothing_tau, params, out_path, name=""
):
    y_smooth = utils.smooth_raster(y.clone(), smoothing_tau)

    axes_fontsize = 10
    legend_fontsize = 6
    tick_fontsize = 10
    title_fontsize = 10
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
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 3))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    hit_rate_raw_list = list()
    false_rate_raw_list = list()

    for th in params["threshold_list"]:
        event_matrix_from_raw = utils.predict_event(y_smooth, th)
        event_matrix_from_raw = event_matrix_from_raw[
            :, :, : -params["kernel_length"] + 1
        ]

        hit_rate_raw, false_rate_raw = utils.compute_hit_rate(
            x, event_matrix_from_raw, tol=event_tol
        )
        hit_rate_raw = np.minimum(hit_rate_raw, 1)

        hit_rate_raw_list.append(hit_rate_raw)
        false_rate_raw_list.append(false_rate_raw)

    plt.plot(
        1 - np.array(hit_rate_raw_list),
        false_rate_raw_list,
        color="black",
        label="Smoothed raster",
    )

    #### for xhat
    hit_rate_xhat, false_rate_xhat = utils.compute_hit_rate(x, xhat, tol=event_tol)
    hit_rate_xhat = np.minimum(hit_rate_xhat, 1)

    plt.plot(
        1 - np.array(hit_rate_xhat),
        false_rate_xhat,
        ".",
        color="Red",
        label="Code",
    )

    plt.xlabel("Missed Events")
    plt.ylabel("False Events")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(
            out_path,
            "{}_missfalse_tol{}_smoothtau{}.svg".format(name, event_tol, smoothing_tau),
        ),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def get_data(data_path, postprocess_path, init_reg):
    datafile_name = data_path.split("/")[-1].split(".pt")[0]

    y = torch.load(os.path.join(postprocess_path, "y_{}.pt".format(datafile_name)))
    x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
    xhat = torch.load(
        os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
    )

    num_neurons = y.shape[1]
    y = torch.reshape(y, (-1, 1, y.shape[-1]))
    # for grouping look into all neurons
    xhat_groupnorm = xhat.norm(dim=1)
    xhat = torch.repeat_interleave(xhat_groupnorm, num_neurons, dim=0)
    # repeat x for how many neurons are they into the 0 (trial) dim
    x = torch.repeat_interleave(x, num_neurons, dim=0)
    return x, xhat, None, y


if __name__ == "__main__":
    main()
