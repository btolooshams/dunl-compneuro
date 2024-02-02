"""
Copyright (c) 2020 Bahareh Tolooshams

plot raw data

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

import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/simulated_dopaminespiking_40neurons_50trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_22_02_49",
        # default="../results/simulated_dopaminespiking_40neurons_100trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_44_03",
        # default="../results/simulated_dopaminespiking_40neurons_200trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_18_24",
        # default="../results/simulated_dopaminespiking_40neurons_300trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_19_37",
        # default="../results/simulated_dopaminespiking_40neurons_400trials_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_22_21_48_04",
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
        "--surprise-dur",
        type=int,
        help="surprise duration after onset in samples",
        default=600,
    )
    parser.add_argument(
        "--expected-dur",
        type=int,
        help="expected duration after onset in samples",
        default=2100,
    )
    parser.add_argument(
        "--delay-bwd",
        type=int,
        help="delay bwd from onset",
        default=975,  # on original resolution of 1 ms
    )
    parser.add_argument(
        "--reward-delay",
        type=int,
        help="reward delay from the cue onset",
        default=1500,  # on original resolution of 1 ms
    )
    parser.add_argument(
        "--smoothing-tau",
        type=int,
        help="smoothing tau for plot psth",
        default=20,  # on original resolution of 1 ms
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0],
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
        dataset = datasetloader.DUNLdatasetwithRaster(data_path_cur)

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
        y_surprise = list()
        y_expected = list()

        y_surprise_rew_amount = list()
        y_expected_rew_amount = list()

        for idx, (y, x, a, label_int, raster) in tqdm(
            enumerate(dataloader), disable=params["tqdm_prints_inside_disable"]
        ):
            label = label_int.clone()
            tmp_ctr = 0
            for reward in params["reward_amount_list"]:
                tmp_ctr += 1
                label[label == tmp_ctr] = reward

            # send data to device (cpu or gpu)
            raster_in = torch.reshape(
                raster, (int(raster.shape[0] * raster.shape[1]), 1, raster.shape[2])
            )

            raster_in = raster_in.to(device)
            x = x.to(device)
            label = label.to(device)

            raster_smooth_out = utils.smooth_raster(raster_in, params["smoothing_tau"])
            # move the neuron axis back
            raster_smooth = (
                torch.reshape(
                    raster_smooth_out,
                    (raster.shape[0], raster.shape[1], raster.shape[2]),
                )
                .detach()
                .clone()
            )

            for i in range(y.shape[0]):
                yi = raster_smooth[i]
                xi = x[i]
                labeli = label[i]

                # reward presence
                cue_flag = torch.sum(torch.abs(xi[0]), dim=-1).item()
                if cue_flag:
                    # expected trial
                    cue_onset = (
                        np.where(xi[0] > 0)[-1][0]
                    ) * dataset.time_bin_resolution
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
                    reward_onset = (
                        np.where(xi[1] > 0)[-1][0]
                    ) * dataset.time_bin_resolution
                    reward_surprise_dot = 0
                    y_surprise_curr = yi[
                        :,
                        reward_onset
                        - params["delay_bwd"] : reward_onset
                        + params["surprise_dur"],
                    ]
                    y_surprise.append(y_surprise_curr)
                    y_surprise_rew_amount.append(labeli)

        y_expected = torch.stack(y_expected, dim=0)
        y_surprise = torch.stack(y_surprise, dim=0)
        y_expected_rew_amount = torch.stack(y_expected_rew_amount, dim=0)
        y_surprise_rew_amount = torch.stack(y_surprise_rew_amount, dim=0)

        t_expected = np.linspace(
            -params["delay_bwd"], params["expected_dur"], y_expected.shape[-1]
        )
        t_surprise = np.linspace(
            -params["delay_bwd"],
            params["surprise_dur"],
            y_surprise.shape[-1],
        )

        # plot -------------------------------------------------------#

        num_neurons = y_surprise.shape[1]
        for neuron_ctr in range(num_neurons):
            print(f"{dataloader.dataset.data_path} neuron {neuron_ctr}")
            y_surprise_neuron = y_surprise[:, neuron_ctr]
            y_expected_neuron = y_expected[:, neuron_ctr]

            fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)

            for ax in axn.flat:
                ax.tick_params(axis="x", direction="out")
                ax.tick_params(axis="y", direction="out")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

            dot_loc = torch.max(torch.mean(y_expected_neuron, dim=0))
            dot_loc = torch.maximum(
                dot_loc, torch.max(torch.mean(y_surprise_neuron, dim=0))
            )

            plt.subplot(2, 1, 1)
            plt.title(r"$\textbf{Surprise\ Trials}$")
            plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
            ctr = -1
            for reward in params["reward_amount_list"]:
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

            plt.subplot(2, 1, 2)
            plt.title(r"$\textbf{Expected\ Trials}$")
            plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
            plt.axvline(
                x=params["reward_delay"], linestyle="--", linewidth=0.5, color="black"
            )
            ctr = -1
            for reward in params["reward_amount_list"]:
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
            xtic = [int(x) for x in xtic]
            plt.xticks(xtic, xtic)
            plt.xlabel("Time [ms]", labelpad=0)

            fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
            datafile_name = dataloader.dataset.data_path.split("/")[-1].split(".pt")[0]
            plt.savefig(
                os.path.join(
                    out_path,
                    "raw_{}_neuron{}_smoothingtau{}.svg".format(
                        datafile_name, neuron_ctr, params["smoothing_tau"]
                    ),
                ),
                bbox_inches="tight",
                pad_inches=0.02,
            )

            plt.close()

    print(f"plotting of raw is done. plots are saved at {out_path}")


if __name__ == "__main__":
    main()
