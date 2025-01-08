"""
Copyright (c) 2025 Bahareh Tolooshams

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
    filename_list = [f"{x}" for x in filename_list if "20unrolling" in x]
    filename_list = [f"{x}" for x in filename_list if "noisy" not in x]
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

    out_path = os.path.join(
        "../",
        "figures",
        "orthkernels",
        "r2",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    csv_path = os.path.join("../", "figures", "orthkernels", "csv")
    csv_filename_list = os.listdir(csv_path)
    csv_filename_list = [f"{csv_path}/{x}" for x in csv_filename_list if ".csv" in x]

    kernels_list = [3, 4, 5, 6, 7]
    # kernels_list = [4,5,6]

    ################################
    loss_dict = dict()
    for k in kernels_list:
        loss_dict[f"kernel{k}"] = dict()

    for csv_filename in csv_filename_list:
        if "100trials" not in csv_filename:
            continue

        with open(csv_filename, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            epoch_list = list()
            loss_list = list()
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    ctr += 1
                    epoch_list.append(float(row[1]))
                    loss_list.append(float(row[2]))

            print(csv_filename)
            num_kernels = csv_filename.split("/")[-1].split("_")[2].split("kernel")[0]
            run_type = (
                csv_filename.split("/")[-1].split("-tag-loss_")[-1].split(".csv")[0]
            )
            print(num_kernels, run_type)

            loss_dict[f"kernel{num_kernels}"][f"{run_type}"] = loss_list
            loss_dict[f"kernel{num_kernels}"][f"{run_type}_epoch"] = epoch_list

    plot_loss(loss_dict, out_path)
    ################################

    ################################
    loss_dict_end = dict()
    for k in kernels_list:
        loss_dict_end[f"kernel{k}"] = dict()

    for csv_filename in csv_filename_list:
        if "100trials" not in csv_filename:
            continue

        with open(csv_filename, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            epoch_list = list()
            loss_list = list()
            ctr = 0
            for row in csv_reader:
                if ctr == 0:
                    ctr += 1
                    continue
                else:
                    ctr += 1
                    epoch_list.append(float(row[1]))
                    loss_list.append(float(row[2]))

            print(csv_filename)
            num_kernels = csv_filename.split("/")[-1].split("_")[2].split("kernel")[0]
            run_type = (
                csv_filename.split("/")[-1].split("-tag-loss_")[-1].split(".csv")[0]
            )
            print(num_kernels, run_type)

            if run_type not in loss_dict_end[f"kernel{num_kernels}"].keys():
                loss_dict_end[f"kernel{num_kernels}"][f"{run_type}"] = list()
            loss_dict_end[f"kernel{num_kernels}"][f"{run_type}"].append(loss_list[-1])

    plot_loss_vs_kernels(loss_dict_end, out_path)
    ################################

    kernels_true = np.load(os.path.join(data_folder, "kernels.npy"))

    r2_score_dict = dict()
    kernels_dict = dict()
    for k in kernels_list:
        r2_score_dict[f"kernel{k}"] = dict()
        kernels_dict[f"kernel{k}"] = dict()
        for t in [20, 50, 100, 250, 500, 1000]:
            r2_score_dict[f"kernel{k}"][f"trials{t}"] = list()
            kernels_dict[f"kernel{k}"][f"trials{t}"] = list()

    for res_path in res_path_list:
        num_trials = int(res_path.split("_")[1].split("trials")[0])

        num_kernels = int(res_path.split("_")[2].split("kernel")[0])

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

        if num_kernels not in kernels_list:
            continue

        if 1:
            y_list, yhat_list = do_inference(
                dataloader_list, net, params, device=device
            )

            y_list = y_list.detach().cpu()
            yhat_list = yhat_list.detach().cpu()

            y_list = y_list.reshape(-1, y_list.shape[-1]).numpy()
            yhat_list = yhat_list.reshape(-1, yhat_list.shape[-1]).numpy()

            print("num_trials", num_trials, "num_kernels", num_kernels)

            # compute r2 score
            r2_score = utils.compute_r2_score(y_list, yhat_list)
            print("r2_score", r2_score)
            np.save(f"{res_path}/r2_score.npy", r2_score)
        else:
            r2_score = np.load(f"{res_path}/r2_score.npy")
            pass

        r2_score_dict[f"kernel{num_kernels}"][f"trials{num_trials}"].append(
            r2_score.item()
        )

        kernels_dict[f"kernel{num_kernels}"][f"trials{num_trials}"].append(kernels)

    ################################
    num_kernels_list_global = list()
    max_corr_list_global = list()
    for k in kernels_list:
        max_corr_list = list()
        num_kernels_list = list()
        for t in [20, 50, 100, 250, 500, 1000]:
            max_corr = 0.0
            for cur in range(len(kernels_dict[f"kernel{k}"][f"trials{t}"])):
                cur_kernel = kernels_dict[f"kernel{k}"][f"trials{t}"][cur]

                kTk = np.abs(np.dot(cur_kernel, cur_kernel.T))
                kTk = kTk - np.eye(k)

                kTk = np.max(kTk, axis=0)

                max_corr = np.max(kTk)

                max_corr_list.append(max_corr)
                num_kernels_list.append(k)

        num_kernels_list_global.append(np.array(num_kernels_list))
        max_corr_list_global.append(np.array(max_corr_list))

    num_kernels_list_global = np.stack(num_kernels_list_global).T
    max_corr_list_global = np.stack(max_corr_list_global).T

    outname = f"maxcorrbetweentwokernels_vs_numkernels_together_witherrbar.png"
    plot_maxcorrkernel_wih_errbar(
        max_corr_list_global, num_kernels_list_global, out_path, outname
    )
    outname = f"maxcorrbetweentwokernels_vs_numkernels_together_witherrbar.svg"
    plot_maxcorrkernel_wih_errbar(
        max_corr_list_global, num_kernels_list_global, out_path, outname
    )

    ################################
    num_kernels_list_global = list()
    r2_score_list_global = list()
    num_trials_list = list()

    for k in kernels_list:
        num_kernels_list = list()
        r2_score_list = list()
        for t in [20, 50, 100, 250, 500, 1000]:
            for run in range(len(r2_score_dict[f"kernel{k}"][f"trials{t}"])):
                num_kernels_list.append(k)

                r2 = r2_score_dict[f"kernel{k}"][f"trials{t}"][run]
                r2_score_list.append(r2)

        num_kernels_list_global.append(np.array(num_kernels_list))
        r2_score_list_global.append(np.array(r2_score_list))
        num_trials_list.append(t)

    outname = f"r2_vs_numkernels.png"
    plot_r2_all(
        r2_score_list_global,
        num_kernels_list_global,
        num_trials_list,
        out_path,
        outname,
    )

    num_kernels_list_global = np.stack(num_kernels_list_global).T
    r2_score_list_global = np.stack(r2_score_list_global).T

    outname = f"r2_vs_numkernels_together_witherrbar.png"
    plot_r2_with_errbar(
        r2_score_list_global, num_kernels_list_global, out_path, outname
    )
    outname = f"r2_vs_numkernels_together_witherrbar.svg"
    plot_r2_with_errbar(
        r2_score_list_global, num_kernels_list_global, out_path, outname
    )

    ################################
    num_kernels_list_global = list()
    consistency_list_global = list()
    for k in kernels_list:
        consistency_list = list()
        num_kernels_list = list()
        for t in [20, 50, 100, 250, 500, 1000]:
            min_corr_list = list()
            for cur in range(len(kernels_dict[f"kernel{k}"][f"trials{t}"])):
                cur_kernel = kernels_dict[f"kernel{k}"][f"trials{t}"][cur]

                min_corr = 0.0
                for other in range(len(kernels_dict[f"kernel{k}"][f"trials{t}"])):
                    if cur == other:
                        continue

                    other_kernel = kernels_dict[f"kernel{k}"][f"trials{t}"][other]

                    kTk = np.zeros((cur_kernel.shape[0], cur_kernel.shape[0]))
                    for kernel_index in range(cur_kernel.shape[0]):
                        for kernel_index_other in range(other_kernel.shape[0]):
                            corrr = np.correlate(
                                cur_kernel[kernel_index],
                                other_kernel[kernel_index_other],
                                mode="full",
                            )

                            kTk[kernel_index, kernel_index_other] = np.max(
                                np.abs(corrr)
                            )

                    kTk = np.max(kTk, axis=0)
                    min_corr += np.min(kTk)

                min_corr_list.append(
                    min_corr / len(kernels_dict[f"kernel{k}"][f"trials{t}"])
                )

            min_corr = np.mean(min_corr_list)

            print(t, k, min_corr)
            consistency_list.append(min_corr)
            num_kernels_list.append(k)

        num_kernels_list_global.append(np.array(num_kernels_list))
        consistency_list_global.append(np.array(consistency_list))

    num_kernels_list_global = np.stack(num_kernels_list_global).T
    consistency_list_global = np.stack(consistency_list_global).T

    outname = f"consistencykernel_woth_errbar.png"
    plot_consistencykernel_wih_errbar(
        consistency_list_global, num_kernels_list_global, out_path, outname
    )
    outname = f"consistencykernel_woth_errbar.svg"
    plot_consistencykernel_wih_errbar(
        consistency_list_global, num_kernels_list_global, out_path, outname
    )


def plot_loss(loss_dict, out_path):
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
    for rec_type in ["val", "train", "train_ae"]:
        row = 1
        col = 1
        fig, ax = plt.subplots(row, col, sharex=True, sharey=True, figsize=(4, 3))

        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.subplot(row, col, 1)

        color_list = ["green", "red", "blue", "orange", "gray"]
        ctr = 0
        r = 1
        for kernel_num in [3, 4, 5, 6, 7]:
            loss_list = loss_dict[f"kernel{kernel_num}"][f"{rec_type}"]
            epoch_list = loss_dict[f"kernel{kernel_num}"][f"{rec_type}_epoch"]

            loss_list = np.array(loss_list)
            epoch_list = np.array(epoch_list)

            if kernel_num == 5:
                label = f"{kernel_num} kernels (gt)"
            else:
                label = f"{kernel_num} kernels"

            plt.plot(
                epoch_list[::r],
                loss_list[::r],
                color=color_list[ctr],
                linewidth=lw_true,
                label=label,
            )

            ctr += 1

        plt.legend()

        plt.ylabel("Loss", labelpad=0)
        plt.xlabel("Epochs", labelpad=0)

        fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

        plt.savefig(
            os.path.join(out_path, f"loss_{rec_type}.png"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.savefig(
            os.path.join(out_path, f"loss_{rec_type}.svg"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close()


def plot_loss_vs_kernels(loss_dict_end, out_path):
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
    for rec_type in ["val", "train", "train_ae"]:
        row = 1
        col = 1
        fig, ax = plt.subplots(row, col, sharex=True, sharey=True, figsize=(4, 3))

        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.subplot(row, col, 1)

        ctr = 0
        r = 1
        loss_list_all = list()
        kernel_num_all = list()
        for kernel_num in [3, 4, 5, 6, 7]:
            loss_list = loss_dict_end[f"kernel{kernel_num}"][f"{rec_type}"]
            print(kernel_num, len(loss_list))
            loss_list_all.append(loss_list)
            kernel_num_all.append(kernel_num)

        loss_list_all = np.stack(loss_list_all).T

        ax.errorbar(
            kernel_num_all,
            np.mean(loss_list_all, axis=0),
            yerr=np.std(loss_list_all, axis=0),
            color="black",
        )

        plt.ylabel("Loss", labelpad=0)
        plt.xlabel("Number of Kernels", labelpad=0)

        fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

        plt.savefig(
            os.path.join(out_path, f"loss_vs_kernels_{rec_type}.png"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.savefig(
            os.path.join(out_path, f"loss_vs_kernels_{rec_type}.svg"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close()


def plot_maxcorrkernel(k_list, num_kernels_list, out_path, outname):
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

    plt.plot(num_kernels_list, k_list, color="black", linewidth=lw_true)
    plt.plot(num_kernels_list, k_list, ".", markersize=markersize, color="black")

    plt.xticks([3, 4, 5, 6, 7], [3, 4, 5, 6, 7])
    plt.ylabel("Max Corr between Two Kernels", labelpad=0)
    plt.xlabel("Number of Kernels", labelpad=0)
    plt.ylim([0, 1])

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_maxcorrkernel_wih_errbar(k_list, num_kernels_list, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

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

    ax.errorbar(
        np.mean(num_kernels_list, axis=0),
        np.mean(k_list, axis=0),
        yerr=np.std(k_list, axis=0),
        color="black",
    )

    plt.xticks([3, 4, 5, 6, 7], [3, 4, 5, 6, 7])
    plt.ylabel("Max Corr between Two Kernels", labelpad=0)
    plt.xlabel("Number of Kernels", labelpad=0)
    plt.ylim([0, 1])

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_consistencykernel_wih_errbar(k_list, num_kernels_list, out_path, outname):
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
        np.mean(num_kernels_list, axis=0),
        np.mean(k_list, axis=0),
        yerr=np.std(k_list, axis=0),
        color="black",
    )

    plt.xticks([3, 4, 5, 6, 7], [3, 4, 5, 6, 7])
    plt.ylabel("Consistency of Kernels Across Runs", labelpad=0)
    plt.xlabel("Number of Kernels", labelpad=0)
    # plt.ylim([0, 1])

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    plt.savefig(
        os.path.join(out_path, outname),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()


def plot_r2(r2_score_list, num_kernels_list, out_path, outname):
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

    plt.plot(num_kernels_list, r2_score_list, color="black", linewidth=lw_true)
    plt.plot(num_kernels_list, r2_score_list, ".", markersize=markersize, color="black")

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


def plot_r2_with_errbar(r2_score_list, num_kernels_list, out_path, outname):
    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

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
        np.mean(num_kernels_list, axis=0),
        np.mean(r2_score_list, axis=0),
        yerr=np.std(r2_score_list, axis=0),
        color="black",
    )
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
