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
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    epoch_type = "best_val"

    res_folder = "../results"
    filename_list = os.listdir(res_folder)
    filename_list = [f"{x}" for x in filename_list if "500trials" in x]
    filename_list = [f"{x}" for x in filename_list if "5kernel" in x]
    filename_list = [f"{x}" for x in filename_list if "forunrolling" in x]
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

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "r2_unrolling",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ################################

    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

    # unrolling_list = [5, 10, 20, 50]
    unrolling_list = [2, 5, 10, 20]

    r2_score_dict = dict()
    unrolling_dict = dict()
    kernels_dict = dict()
    for u in unrolling_list:
        r2_score_dict[f"unrolling{u}"] = list()
        unrolling_dict[f"unrolling{u}"] = list()
        kernels_dict[f"unrolling{u}"] = list()

    for res_path in res_path_list:
        if "40unrolling" in res_path:
            continue

        num_trials = int(res_path.split("_")[1].split("trials")[0])

        num_kernels = int(res_path.split("_")[2].split("kernel")[0])

        num_unrolled = int(res_path.split("_")[4].split("unrolling")[0])
        print("num_unrolled", num_unrolled)

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

        if 1:
            y_list, yhat_list = do_inference(
                dataloader_list, net, params, device=device
            )

            y_list = y_list.detach().cpu()
            yhat_list = yhat_list.detach().cpu()

            y_list = y_list.reshape(-1, y_list.shape[-1]).numpy()
            yhat_list = yhat_list.reshape(-1, yhat_list.shape[-1]).numpy()

            # compute r2 score
            r2_score = utils.compute_r2_score(y_list, yhat_list)
            print("r2_score", r2_score)
            np.save(f"{res_path}/r2_score.npy", r2_score)
        else:
            r2_score = np.load(f"{res_path}/r2_score.npy")


        r2_score_dict[f"unrolling{num_unrolled}"].append(r2_score.item())
        unrolling_dict[f"unrolling{num_unrolled}"].append(num_unrolled)
        kernels_dict[f"unrolling{num_unrolled}"].append(kernels)

    ################################
    num_unrolled_list_global = list()
    r2_score_list_global = list()
    for u in unrolling_list:
        r2_score_list_global.append(r2_score_dict[f"unrolling{u}"])
        num_unrolled_list_global.append(unrolling_dict[f"unrolling{u}"])

    num_unrolled_list_global = np.stack(num_unrolled_list_global)
    r2_score_list_global = np.stack(r2_score_list_global)

    print(r2_score_list_global)
    print(num_unrolled_list_global)

    outname = f"unrollingeffect_r2_vs_unrolling.png"
    plot_r2_with_errbar(
        r2_score_list_global, num_unrolled_list_global, out_path, outname
    )
    outname = f"unrollingeffect_r2_vs_unrolling.svg"
    plot_r2_with_errbar(
        r2_score_list_global, num_unrolled_list_global, out_path, outname
    )

    ################################
    num_unrolled_list_global = list()
    ker_err_list_global = list()
    for num_unrolled in unrolling_list:

        ker_err_list = list()
        num_unrolled_list = list()

        min_corr_list = list()
        for cur in range(len(kernels_dict[f"unrolling{num_unrolled}"])):
            cur_kernel = kernels_dict[f"unrolling{num_unrolled}"][cur]

            kTk = np.zeros((cur_kernel.shape[0], cur_kernel.shape[0]))
            for kernel_index in range(cur_kernel.shape[0]):
                for kernel_index_true in range(kernels_true.shape[0]):
                    corrr = np.correlate(
                        cur_kernel[kernel_index],
                        kernels_true[kernel_index_true],
                        mode="full",
                    )
                    kTk[kernel_index, kernel_index_true] = np.max(np.abs(corrr))

            kTk = np.max(kTk, axis=0)  # taking the best corresponding kernel
            # rm the last colum of zero
            ker_err = kTk
            ker_err = np.mean(ker_err[:-1])
            print(num_unrolled, ker_err)

            ker_err_list.append(ker_err)
            num_unrolled_list.append(num_unrolled)

        num_unrolled_list_global.append(np.array(num_unrolled_list))
        ker_err_list_global.append(np.array(ker_err_list))

    num_unrolled_list_global = np.stack(num_unrolled_list_global).T
    ker_err_list_global = np.stack(ker_err_list_global).T
    outname = f"unrolled_kernelsim_with_errbar.png"
    plot_kernelerr_wih_errbar(
        ker_err_list_global, num_unrolled_list_global, out_path, outname
    )
    outname = f"unrolled_kernelsim_with_errbar.svg"
    plot_kernelerr_wih_errbar(
        ker_err_list_global, num_unrolled_list_global, out_path, outname
    )


def plot_kernelerr_wih_errbar(k_list, num_unrolled_list, out_path, outname):
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

    sorted_indices = np.argsort(np.mean(num_unrolled_list, axis=0))
    ax.errorbar(
        np.mean(num_unrolled_list, axis=0)[sorted_indices],
        np.mean(k_list, axis=0)[sorted_indices],
        yerr=np.std(k_list, axis=0)[sorted_indices],
        color="black",
    )

    xtic = np.mean(num_unrolled_list, axis=0)[sorted_indices]
    xtic = [int(x) for x in xtic]
    plt.xticks(xtic, xtic)
    plt.ylabel("Mean Correlation(True, Est)", labelpad=0)
    plt.xlabel("Number of Unrolling", labelpad=0)
    plt.ylim([0, 1])

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_r2_with_errbar(r2_score_list, num_unrolling_list, out_path, outname):
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
    ax.errorbar(
        np.mean(num_unrolling_list, axis=1),
        np.mean(r2_score_list, axis=1),
        yerr=np.std(r2_score_list, axis=1),
        color="black",
    )
    xtics = [int(x) for x in np.mean(num_unrolling_list, axis=1)]
    plt.xticks(xtics, xtics)
    plt.ylabel("R2", labelpad=0)
    plt.xlabel("Number of Unrolling", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_r2_all(r2_score_list, num_kernels_list, num_trials, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"
    lw_true = 2

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

    for ctr in range(len(r2_score_list)):
        plt.plot(
            num_kernels_list[ctr],
            r2_score_list[ctr],
            color="black",
            label=num_trials[ctr],
            linewidth=lw_true,
        )

    plt.legend()
    plt.xticks([3, 4, 5, 6, 7], [3, 4, 5, 6, 7])
    plt.ylabel("R2", labelpad=0)
    plt.xlabel("Number of Kernels", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


if __name__ == "__main__":
    main()
