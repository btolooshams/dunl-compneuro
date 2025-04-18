"""
Copyright (c) 2020 Bahareh Tolooshams

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
        # default="nmf",
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
    x_corr_from_all_datasets_sur = list()
    y_corr_from_all_datasets_sur = list()

    x_corr_from_all_datasets_exp = list()
    y_corr_from_all_datasets_exp = list()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        # (neuron, time, trials)
        y = np.load(
            os.path.join(postprocess_path, "y_for_pcanmf_{}.npy".format(datafile_name))
        )
        label = np.load(
            os.path.join(
                postprocess_path, "label_for_pcanmf_{}.npy".format(datafile_name)
            )
        )

        y = np.transpose(y, (0, 2, 1))

        print("label", label.shape)
        print("y", y.shape)

        num_neurons = y.shape[0]

        x_corr_sur = torch.zeros(num_neurons, params["num_comp"])
        x_corr_exp = torch.zeros(num_neurons, params["num_comp"])

        y_corr_sur = torch.zeros(num_neurons)
        y_corr_exp = torch.zeros(num_neurons)

        for neuron_ctr in range(num_neurons):
            label_neuron = label  # label is shared across neurons
            y_neuron = y[neuron_ctr]

            sur_indices = label_neuron < 0
            exp_indices = label_neuron > 0

            y_neuron_sur = y_neuron[sur_indices]
            y_neuron_exp = y_neuron[exp_indices]

            label_sur = label[sur_indices]
            label_exp = label[exp_indices]

            print("y_neuron", y_neuron.shape)
            print("label_neuron", label_neuron.shape)

            print("y_neuron_sur", y_neuron_sur.shape)
            print("y_neuron_exp", y_neuron_exp.shape)

            if params["method"] == "pca":
                y_neuron_ready_sur = scaler_transform.transform(y_neuron_sur)
                y_neuron_ready_exp = scaler_transform.transform(y_neuron_exp)
            elif params["method"] == "nmf":
                y_neuron_ready_sur = y_neuron_sur - np.min(y_neuron_sur)
                y_neuron_ready_exp = y_neuron_exp - np.min(y_neuron_exp)

            x_neuron_sur = method_transform.transform(y_neuron_ready_sur)
            x_neuron_exp = method_transform.transform(y_neuron_ready_exp)

            print("x_neuron_sur", x_neuron_sur.shape)
            print("x_neuron_exp", x_neuron_exp.shape)

            for code_ctr in range(params["num_comp"]):
                x_corr_curr_sur, _ = sp.stats.spearmanr(
                    x_neuron_sur[:, code_ctr], abs(label_sur)
                )

                x_corr_curr_exp, _ = sp.stats.spearmanr(
                    x_neuron_exp[:, code_ctr], abs(label_exp)
                )

                x_corr_sur[neuron_ctr, code_ctr] = x_corr_curr_sur
                x_corr_exp[neuron_ctr, code_ctr] = x_corr_curr_exp

        x_corr_from_all_datasets_sur.append(x_corr_sur)
        x_corr_from_all_datasets_exp.append(x_corr_exp)
        y_corr_from_all_datasets_sur.append(y_corr_sur)
        y_corr_from_all_datasets_exp.append(y_corr_exp)

    x_corr_from_all_datasets_sur = np.concatenate(x_corr_from_all_datasets_sur, axis=0)
    y_corr_from_all_datasets_sur = np.concatenate(y_corr_from_all_datasets_sur, axis=0)

    x_corr_from_all_datasets_exp = np.concatenate(x_corr_from_all_datasets_exp, axis=0)
    y_corr_from_all_datasets_exp = np.concatenate(y_corr_from_all_datasets_exp, axis=0)

    dunl_res = torch.load(
        "../data/dopamine-spiking-eshel-uchida/eshel_spearman_for_comparison.pt"
    )
    dunl_corr_sur = dunl_res["code_sur_corr"]
    dunl_corr_exp = dunl_res["code_exp_corr"]
    print(dunl_corr_exp.shape)

    print(x_corr_from_all_datasets_exp.shape, dunl_corr_exp.shape)
    print(x_corr_from_all_datasets_sur.shape, dunl_corr_sur.shape)

    for code_ctr in range(params["num_comp"]):
        out_path_name = os.path.join(
            out_path,
            "dunl_vs_{}_spearman_correlation_{}.svg".format(params["method"], code_ctr),
        )

        plot_spearman_correlation(
            dunl_corr_exp,
            torch.tensor(x_corr_from_all_datasets_exp[:, code_ctr]),
            dunl_corr_sur,
            torch.tensor(x_corr_from_all_datasets_sur[:, code_ctr]),
            params,
            out_path_name=out_path_name,
            color_ctr=code_ctr,
        )

        out_path_name = os.path.join(
            out_path,
            "dunl_vs_{}_spearman_correlation_{}_neg.svg".format(
                params["method"], code_ctr
            ),
        )

        plot_spearman_correlation(
            dunl_corr_exp,
            -torch.tensor(x_corr_from_all_datasets_exp[:, code_ctr]),
            dunl_corr_sur,
            -torch.tensor(x_corr_from_all_datasets_sur[:, code_ctr]),
            params,
            out_path_name=out_path_name,
            color_ctr=code_ctr,
        )

    print(f"plotting of kernels is done. plots are saved at {out_path}")


def plot_spearman_correlation(
    code_exp, m_exp, code_sur, m_sur, params, out_path_name, color_ctr
):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    markersize = 4
    fontfamily = "sans-serif"

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
            "font.family": fontfamily,
        }
    )

    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=params["figsize"])

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")

    for code_ctr in range(1, code_sur.shape[1] - 1):
        plt.subplot(1, 2, code_ctr)
        plt.scatter(
            code_sur[:, code_ctr],
            m_sur,
            marker=".",
            color=params["color_list"][color_ctr],
            s=markersize,
        )
        plt.scatter(
            code_exp[:, code_ctr],
            m_exp,
            marker=".",
            color=params["color_list"][color_ctr],
            s=markersize,
        )
        plt.plot([-0.5, 1], [-0.5, 1], "--", color="gray", linewidth=1)

        if code_ctr == 1:
            if params["method"] == "nmf":
                method = "NMF"
            elif params["method"] == "pca":
                method = "PCA"
            axn[code_ctr - 1].set_ylabel(f"Spearman({method}, Reward)")
            axn[code_ctr].set_xlabel("Spearman(Code, Reward)")

        if code_ctr == 1:
            plt.title(r"$\textbf{Reward\ I}$")
        elif code_ctr == 2:
            plt.title(r"$\textbf{Reward\ II}$")

        plt.scatter(
            torch.mean(code_sur[:, code_ctr]),
            torch.mean(m_sur),
            marker="x",
            color="green",
            lw=1.5,
            s=markersize * 6,
        )
        print(
            "sur",
            "y",
            torch.mean(m_sur),
            "code",
            torch.mean(code_sur[:, code_ctr]),
            code_ctr,
        )
        plt.scatter(
            torch.mean(code_exp[:, code_ctr]),
            torch.mean(m_exp),
            marker="x",
            color="yellow",
            lw=1.5,
            s=markersize * 6,
        )
        print(
            "exp",
            "y",
            torch.mean(m_exp),
            "code",
            torch.mean(code_exp[:, code_ctr]),
            code_ctr,
        )
        xtic = np.array([0, 0.5, 1])
        xtic = [x for x in xtic]
        plt.xticks(xtic, xtic)
        plt.yticks(xtic, xtic)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        out_path_name,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close()


if __name__ == "__main__":
    main()
