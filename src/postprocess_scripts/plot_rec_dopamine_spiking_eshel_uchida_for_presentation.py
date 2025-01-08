"""
Copyright (c) 2025 Bahareh Tolooshams

plot rec data

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

import sys

sys.path.append("../src/")

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_2023_07_14_12_37_30",
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
        default=24,
    )
    parser.add_argument(
        "--expected-dur",
        type=int,
        help="expected duration after onset in samples",
        default=84,
    )
    parser.add_argument(
        "--delay-bwd",
        type=int,
        help="delay bwd from onset",
        default=39,  # this is after the bining
    )
    parser.add_argument(
        "--reward-delay",
        type=int,
        help="reward delay from the cue onset",
        default=60,  # this is after the bining
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

    # set time bin resolution -----------------------------------------------#
    data_dict = torch.load(data_path_list[0])
    params["time_bin_resolution"] = data_dict["time_bin_resolution"]

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
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "figures",
        "rec_separate",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.to(device)
    net.eval()

    # plot configuration -------------------------------------------------------#

    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 15
    lw = 2

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

        yhat_surprise = list()
        yhat_expected = list()

        y_surprise_rew_amount = list()
        y_expected_rew_amount = list()

        for idx, (y, x, a, label_int) in tqdm(
            enumerate(dataloader), disable=params["tqdm_prints_inside_disable"]
        ):
            label = label_int.clone()
            tmp_ctr = 0
            for reward in params["reward_amount_list"]:
                tmp_ctr += 1
                label[label == tmp_ctr] = reward

            # put neuron dim into the trial (batch)
            y_in = torch.reshape(y, (int(y.shape[0] * y.shape[1]), 1, y.shape[2]))
            a_in = torch.reshape(a, (int(a.shape[0] * a.shape[1]), 1, a.shape[2]))
            # repeat x for how many neurons are they into the 0 (trial) dim
            x_in = torch.repeat_interleave(x, a.shape[1], dim=0)

            # send data to device (cpu or gpu)
            y_in = y_in.to(device)
            x_in = x_in.to(device)
            a_in = a_in.to(device)

            label = label.to(device)

            if params["code_supp"]:
                x_code_supp = x_in
            else:
                x_code_supp = None

            # forward encoder
            xhat, a_est = net.encode(y_in, a_in, x_code_supp)
            # forward decoder
            hxmu_out = net.decode(xhat, a_est)

            if params["model_distribution"] == "binomial":
                yhat_out = torch.sigmoid(hxmu_out)
            else:
                raise NotImplementedError("model distribution is not implemented")

            # move the neuron axis back
            yhat = (
                torch.reshape(yhat_out, (y.shape[0], y.shape[1], y.shape[2]))
                .detach()
                .clone()
            )

            for i in range(y.shape[0]):
                yi = y[i]
                yihat = yhat[i]
                xi = x[i]
                ai = a[i]
                labeli = label[i]

                # reward presence
                cue_flag = torch.sum(torch.abs(xi[0]), dim=-1).item()
                if cue_flag:
                    # expected trial
                    cue_onset = np.where(xi[0] > 0)[-1][0]
                    cue_expected_dot = 0
                    reward_expected_dot = params["reward_delay"]
                    y_expected_curr = yi[
                        :,
                        cue_onset
                        - params["delay_bwd"] : cue_onset
                        + params["expected_dur"],
                    ]
                    y_expected.append(y_expected_curr)
                    yhat_expected_curr = yihat[
                        :,
                        cue_onset
                        - params["delay_bwd"] : cue_onset
                        + params["expected_dur"],
                    ]
                    yhat_expected.append(yhat_expected_curr)
                    y_expected_rew_amount.append(labeli)
                else:
                    # surprise trial
                    reward_onset = np.where(xi[1] > 0)[-1][0]
                    reward_surprise_dot = 0
                    y_surprise_curr = yi[
                        :,
                        reward_onset
                        - params["delay_bwd"] : reward_onset
                        + params["surprise_dur"],
                    ]
                    y_surprise.append(y_surprise_curr)
                    yhat_surprise_curr = yihat[
                        :,
                        reward_onset
                        - params["delay_bwd"] : reward_onset
                        + params["surprise_dur"],
                    ]
                    yhat_surprise.append(yhat_surprise_curr)
                    y_surprise_rew_amount.append(labeli)

        y_expected = torch.stack(y_expected, dim=0)
        y_surprise = torch.stack(y_surprise, dim=0)
        yhat_expected = torch.stack(yhat_expected, dim=0)
        yhat_surprise = torch.stack(yhat_surprise, dim=0)
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

            yhat_surprise_neuron = yhat_surprise[:, neuron_ctr]
            yhat_expected_neuron = yhat_expected[:, neuron_ctr]

            dot_loc = torch.max(torch.mean(y_expected_neuron, dim=0))
            dot_loc = torch.maximum(
                dot_loc, torch.max(torch.mean(y_surprise_neuron, dim=0))
            )

            ctr = -1
            for reward in params["reward_amount_list"]:
                ctr += 1
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 2))

                ax.tick_params(axis="x", direction="out")
                ax.tick_params(axis="y", direction="out")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

                plt.subplot(1, 1, 1)
                # plt.title(r"$\textbf{Expected\ Trials}$")
                plt.axvline(x=0, linestyle="--", linewidth=0.5, color="black")
                plt.axvline(
                    x=params["reward_delay"],
                    linestyle="--",
                    linewidth=0.5,
                    color="black",
                )
                indices = np.where(y_expected_rew_amount == reward)
                plt.plot(
                    t_expected,
                    torch.mean(yhat_expected_neuron[indices], dim=0),
                    # "--",
                    # color=params["color_list"][ctr],
                    color="black",
                    lw=lw,
                )

                plt.xlim(t_expected[30], t_expected[-1])
                plt.yticks([])

                xtic = np.array([0, 0.5, 1, 1.4]) * params["reward_delay"]
                xtic_figure = [int(x * params["time_bin_resolution"]) for x in xtic]
                plt.xticks(xtic, xtic_figure)
                plt.xlabel("Time (ms)", labelpad=0)

                fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
                datafile_name = dataloader.dataset.data_path.split("/")[-1].split(
                    ".pt"
                )[0]
                plt.savefig(
                    os.path.join(
                        out_path,
                        "interview_rec_{}_neuron{}_reward{}.svg".format(
                            datafile_name, neuron_ctr, str(reward).replace(".", "p")
                        ),
                    ),
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                plt.close()

    print(f"plotting of rec is done. plots are saved at {out_path}")


if __name__ == "__main__":
    main()
