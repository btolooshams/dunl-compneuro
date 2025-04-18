"""
Copyright (c) 2025 Bahareh Tolooshams

plot raw data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
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
        "--regret-dur",
        type=int,
        help="regret duration after onset in samples",
        default=60,
    )
    parser.add_argument(
        "--surprise-dur",
        type=int,
        help="surprise duration after onset in samples",
        default=60,
    )
    parser.add_argument(
        "--expected-dur",
        type=int,
        help="expected duration after onset in samples",
        default=105,
    )
    parser.add_argument(
        "--delay-bwd",
        type=int,
        help="delay bwd from onset",
        default=15,
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=15,
    )
    parser.add_argument(
        "--reward-delay",
        type=int,
        help="reward delay from the cue onset",
        default=45,
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.0, 0.3, 0.5, 1.2, 2.5, 5.0, 8.0, 11.0],
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "darkred",
            "peru",
            "yellowgreen",
            "mediumaquamarine",
            "cornflowerblue",
            "mediumblue",
            "navy",
        ],
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

    # create datasets -------------------------------------------------------#
    dataset_list = list()
    dataloader_list = list()
    for data_path_cur in data_path_list:
        print(data_path_cur)
        dataset = datasetloader.DUNLdataset(data_path_cur)
        dataset_list.append(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )
        dataloader_list.append(dataloader)

    # create folders -------------------------------------------------------#
    out_path = os.path.join(
        params["res_path"],
        "figures",
        "raw",
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

    # go over data -------------------------------------------------------#
    for dataloader in dataloader_list:
        y_regret = list()
        y_surprise = list()
        y_expected = list()

        y_surprise_rew_amount = list()
        y_expected_rew_amount = list()

        for idx, (y, x, a, label) in tqdm(
            enumerate(dataloader), disable=params["tqdm_prints_inside_disable"]
        ):
            # send data to device (cpu or gpu)
            y = y.to(device)
            x = x.to(device)
            a = a.to(device)
            label = label.to(device)

            for i in range(y.shape[0]):
                yi = y[i]
                xi = x[i]
                ai = a[i]
                labeli = label[i]

                if labeli == 0:
                    # regret
                    regret_onset = np.where(xi[0] > 0)[-1][0]
                    cue_regret_dot = 0
                    y_regret_curr = yi[
                        :,
                        regret_onset
                        - params["delay_bwd"] : regret_onset
                        + params["regret_dur"],
                    ]
                    y_regret.append(y_regret_curr)
                else:
                    # reward presence
                    cue_flag = torch.sum(torch.abs(xi[1]), dim=-1).item()
                    if cue_flag:
                        # expected trial
                        cue_onset = np.where(xi[1] > 0)[-1][0]
                        cue_expected_dot = 0
                        reward_expected_dot = params["reward_delay"]
                        y_expected_curr = yi[
                            :,
                            cue_onset
                            - params["delay_bwd"] : cue_onset
                            + params["expected_dur"],
                        ]
                        y_expected.append(y_expected_curr)
                        y_expected_rew_amount.append(labeli)
                    else:
                        # surprise trial
                        reward_onset = np.where(xi[2] > 0)[-1][0]
                        reward_surprise_dot = 0
                        y_surprise_curr = yi[
                            :,
                            reward_onset
                            - params["delay_bwd"] : reward_onset
                            + params["surprise_dur"],
                        ]
                        y_surprise.append(y_surprise_curr)
                        y_surprise_rew_amount.append(labeli)

        y_regret = torch.stack(y_regret, dim=0)
        y_expected = torch.stack(y_expected, dim=0)
        y_surprise = torch.stack(y_surprise, dim=0)
        y_expected_rew_amount = torch.stack(y_expected_rew_amount, dim=0)
        y_surprise_rew_amount = torch.stack(y_surprise_rew_amount, dim=0)

        t_regret = np.linspace(
            -params["delay_bwd"], params["regret_dur"], y_regret.shape[-1]
        )
        t_expected = np.linspace(
            -params["delay_bwd"], params["expected_dur"], y_expected.shape[-1]
        )
        t_surprise = np.linspace(
            -params["delay_bwd"],
            params["surprise_dur"],
            y_surprise.shape[-1],
        )

        # plot -------------------------------------------------------#

        num_neurons = y_regret.shape[1]
        for neuron_ctr in range(num_neurons):
            print(f"{dataloader.dataset.data_path} neuron {neuron_ctr}")
            y_regret_neuron = y_regret[:, neuron_ctr]
            y_surprise_neuron = y_surprise[:, neuron_ctr]
            y_expected_neuron = y_expected[:, neuron_ctr]

            fig, axn = plt.subplots(3, 1, sharex=True, sharey=True)

            for ax in axn.flat:
                ax.tick_params(axis="x", direction="out")
                ax.tick_params(axis="y", direction="out")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

            dot_loc = torch.max(torch.mean(y_expected_neuron, dim=0))
            dot_loc = torch.maximum(
                dot_loc, torch.max(torch.mean(y_surprise_neuron, dim=0))
            )
            dot_loc = torch.maximum(dot_loc, torch.max(torch.mean(y_regret, dim=0)))

            plt.subplot(3, 1, 1)
            plt.title(r"$\textbf{Regret\ Trials}$")
            plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
            ctr = -1
            plt.plot(
                t_regret,
                torch.mean(y_regret_neuron, dim=0),
                color="black",
                lw=0.7,
            )
            plt.plot(
                cue_regret_dot,
                dot_loc * 1.035,
                ".",
                markersize=10,
                color="Orange",
            )

            plt.subplot(3, 1, 2)
            plt.title(r"$\textbf{Surprise\ Trials}$")
            plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
            ctr = -1
            for reward in params["reward_amount_list"][1:]:
                ctr += 1
                indices = np.where(y_surprise_rew_amount == reward)
                plt.plot(
                    t_surprise,
                    torch.mean(y_surprise_neuron[indices], dim=0),
                    color=params["color_list"][ctr],
                    lw=0.7,
                    label="{}".format(reward),
                )
                plt.legend(
                    loc="upper right",
                    ncol=4,
                    borderpad=0.1,
                    labelspacing=0.2,
                    handletextpad=0.4,
                    columnspacing=0.2,
                )
            plt.plot(
                reward_surprise_dot,
                dot_loc * 1.035,
                ".",
                markersize=10,
                color="Blue",
            )

            plt.subplot(3, 1, 3)
            plt.title(r"$\textbf{Expected\ Trials}$")
            plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
            plt.axvline(
                x=params["reward_delay"], linestyle="--", linewidth=0.5, color="black"
            )
            ctr = -1
            for reward in params["reward_amount_list"][1:]:
                ctr += 1
                indices = np.where(y_expected_rew_amount == reward)
                plt.plot(
                    t_expected,
                    torch.mean(y_expected_neuron[indices], dim=0),
                    color=params["color_list"][ctr],
                    lw=0.7,
                )

            plt.plot(
                cue_expected_dot,
                dot_loc * 1.035,
                ".",
                markersize=10,
                color="Orange",
            )
            plt.plot(
                reward_expected_dot,
                dot_loc * 1.035,
                ".",
                markersize=10,
                color="Blue",
            )
            xtic = np.array([0, 0.5, 1, 1.5, 2]) * params["reward_delay"]
            plt.xticks(xtic, xtic / params["sampling_rate"])
            plt.xlabel("Time [s]", labelpad=0)

            fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
            datafile_name = dataloader.dataset.data_path.split("/")[-1].split(".pt")[0]
            plt.savefig(
                os.path.join(
                    out_path, "raw_{}_neuron{}.svg".format(datafile_name, neuron_ctr)
                ),
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close()

    print(f"plotting of raw is done. plots are saved at {out_path}")


if __name__ == "__main__":
    main()
