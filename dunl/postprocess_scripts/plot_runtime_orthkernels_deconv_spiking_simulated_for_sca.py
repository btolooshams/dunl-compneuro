"""
Copyright (c) 2025 Bahareh Tolooshams

plot rec data kernel

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

sys.path.append("../dunl/")

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


def do_inference(
    dataloader_list,
    net,
    params,
    code_sparse_regularization=None,
    device="cpu",
):
    if "code_group_neural_firings_regularization" not in params:
        params["code_group_neural_firings_regularization"] = 0

    for dataloader in dataloader_list:
        datafile_name = dataloader.dataset.data_path.split("/")[-1].split(".pt")[0]

        y_list = list()
        yhat_list = list()

        for idx, (y, x, a, label) in tqdm(
            enumerate(dataloader), disable=params["tqdm_prints_inside_disable"]
        ):
            if params["code_group_neural_firings_regularization"]:
                # we want to group the firing rate of the neurons, so the model will take (b, group, 1, time)
                y_in = y.unsqueeze(dim=2)
                a_in = a.unsqueeze(dim=2)

                # give x as is, the mdoel will take care of the repeat to group neurons.
                x_in = x
            else:
                # put neuron dim into the trial (batch)
                y_in = torch.reshape(y, (int(y.shape[0] * y.shape[1]), 1, y.shape[2]))
                a_in = torch.reshape(a, (int(a.shape[0] * a.shape[1]), 1, a.shape[2]))
                # repeat x for how many neurons are they into the 0 (trial) dim
                x_in = torch.repeat_interleave(x, a.shape[1], dim=0)

            # send data to device (cpu or gpu)
            y_in = y_in.to(device)
            x_in = x_in.to(device)
            a_in = a_in.to(device)

            if params["code_supp"]:
                x_code_supp = x_in
            else:
                x_code_supp = None

            # forward encoder
            xhat_out, a_est = net.encode(y_in, a_in, x_code_supp)
            # forward decoder
            hxmu = net.decode(xhat_out, a_est)

            if params["model_distribution"] == "binomial":
                yhat = torch.sigmoid(hxmu)
            else:
                raise NotImplementedError("model distribution is not implemented")

            # print(y, yhat)
            y_list.append(y)
            yhat_list.append(torch.squeeze(yhat, dim=2))

        # (trials, neurons, time)
        y_list = torch.cat(y_list, dim=0)
        yhat_list = torch.cat(yhat_list, dim=0)

    return y_list, yhat_list


def main():
    print("Runtime.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")

    res_folder = "../results"
    filename_list = os.listdir(res_folder)
    res_path_list = [f"{res_folder}/{x}" for x in filename_list if "runtimespeed" in x]

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

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "runtime",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_trials_list_global = [100, 250, 500, 1000, 2000]

    num_steps_list_global = [250, 500, 1000]

    runtime_list_list = list()
    num_trials_list_list = list()
    ################################
    for num_steps in num_steps_list_global:
        runtime_dict = dict()
        num_trials_dict = dict()
        for t in num_trials_list_global:
            runtime_dict[f"num_trials{t}"] = list()
            num_trials_dict[f"num_trials{t}"] = list()

        for res_path in res_path_list:
            num_trials = int(res_path.split("_")[1].split("trials")[0])

            num_steps_cur = int(res_path.split("_")[5].split("steps")[0])

            if num_trials not in num_trials_list_global:
                continue

            if num_steps_cur != num_steps:
                continue

            time_path = os.path.join(
                res_path,
                f"train_time.pt",
            )
            train_time = torch.load(time_path, map_location=device)

            runtime_dict[f"num_trials{num_trials}"].append(train_time)
            num_trials_dict[f"num_trials{num_trials}"].append(num_trials)

        runtime_list = list()
        num_trials_list = list()
        for num_trials_key in runtime_dict.keys():
            runtime_list.append(runtime_dict[f"{num_trials_key}"])
            num_trials_list.append(num_trials_dict[f"{num_trials_key}"])

        runtime_list = np.stack(runtime_list).T
        num_trials_list = np.stack(num_trials_list).T

        runtime_list = runtime_list / 60

        runtime_list_list.append(runtime_list)
        num_trials_list_list.append(num_trials_list)

    ################################

    outname = f"runtime_with_sca.png"
    plot_runtime_wih_sca(
        runtime_list_list,
        num_trials_list_list,
        num_steps_list_global,
        out_path,
        outname,
    )
    outname = f"runtime_with_sca.svg"
    plot_runtime_wih_sca(
        runtime_list_list,
        num_trials_list_list,
        num_steps_list_global,
        out_path,
        outname,
    )


def plot_runtime_wih_sca(
    k_list_list, num_trials_list_list, label_list, out_path, outname
):
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

    sca_trials = [1, 2, 3, 4, 5, 6, 7, 8]
    sca_runtime = [
        20.50605357438326,
        142.663810569793,
        431.2707537673414,
        1000.2254679370672,
        2091.628797421232,
        3594.698073338717,
        5831.628581888974,
        8967.743003480136,
    ]
    sca_runtime = [x / 60 for x in sca_runtime]  # so now in minutes

    row = 1
    col = 1
    fig, ax = plt.subplots(row, col, sharex=True, sharey=True, figsize=(4, 3))

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplot(row, col, 1)

    color_list = plt.cm.jet(np.linspace(0, 1, len(num_trials_list_list) + 2))
    for ctr in range(len(num_trials_list_list)):
        num_trials_list = num_trials_list_list[ctr]
        k_list = k_list_list[ctr]
        ax.errorbar(
            np.mean(num_trials_list, axis=0),
            np.mean(k_list, axis=0),
            yerr=np.std(k_list, axis=0),
            color=color_list[ctr + 1],
            label=f"DUNL {label_list[ctr]} steps",
        )

    plt.plot(sca_trials, sca_runtime, color="gray", label=f"SCA")
    plt.xscale("log")
    plt.yscale("log")

    plt.grid("on")
    plt.legend()
    xtic = [int(x) for x in np.mean(num_trials_list, axis=0)]
    xtic = [100, 500, 2000]
    xtic.append(1)
    xtic.append(2)
    xtic.append(4)
    xtic.append(8)
    plt.xticks(xtic, xtic)
    plt.ylabel("Training Runtime [minutes]", labelpad=0)
    plt.xlabel("Number of Trials", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
