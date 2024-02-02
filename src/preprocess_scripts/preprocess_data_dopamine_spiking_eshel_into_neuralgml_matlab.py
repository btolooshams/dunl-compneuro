"""
Copyright (c) 2020 Bahareh Tolooshams

preprocess data for lfads

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import scipy

import sys

sys.path.append("../src/")

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-folder",
        type=str,
        help="data path",
        default="../data/dopamine-spiking-eshel-uchida",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/dopamine-spiking-eshel-uchida/neuralgml_matlab",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=800,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of workers for dataloader",
        default=4,
    )
    parser.add_argument(
        "--reward-dur",
        type=int,
        help="duration after onset in samples",
        default=600,
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0],
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    data_folder = params["data_folder"]
    filename_list = os.listdir(data_folder)
    data_path_list = [
        f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
    ]

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

    out_path = params["out_path"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # go over data -------------------------------------------------------#
    y_surprise = list()
    y_expected = list()

    y_surprise_rew_amount = list()
    y_expected_rew_amount = list()

    neuron_surprise = list()
    neuron_expected = list()

    neuron_ctr = -1
    for dataloader in dataloader_list:
        neuron_ctr += 1

        for idx, (y, x, a, label_int, raster) in tqdm(
            enumerate(dataloader), disable=True
        ):
            label = label_int.clone()
            tmp_ctr = 0
            for reward in params["reward_amount_list"]:
                tmp_ctr += 1
                label[label == tmp_ctr] = reward

            # send data to device (cpu or gpu)

            for i in range(y.shape[0]):
                yi = raster[i]
                xi = x[i]
                labeli = label[i]

                # reward presence
                cue_flag = torch.sum(torch.abs(xi[0]), dim=-1).item()

                reward_onset = (
                    np.where(xi[1] > 0)[-1][0]
                ) * dataset.time_bin_resolution

                if cue_flag:
                    # expected trial
                    y_expected_curr = yi[
                        :,
                        reward_onset : reward_onset + params["reward_dur"],
                    ]
                    y_expected.append(y_expected_curr)
                    y_expected_rew_amount.append(labeli)
                    neuron_expected.append(torch.tensor(neuron_ctr))
                else:
                    # surprise trial
                    y_surprise_curr = yi[
                        :,
                        reward_onset : reward_onset + params["reward_dur"],
                    ]
                    y_surprise.append(y_surprise_curr)
                    y_surprise_rew_amount.append(labeli)
                    neuron_surprise.append(torch.tensor(neuron_ctr))

    # stack after all data from all datasets
    y_expected = torch.stack(y_expected, dim=0)
    y_surprise = torch.stack(y_surprise, dim=0)
    y_expected_rew_amount = torch.stack(y_expected_rew_amount, dim=0)
    y_surprise_rew_amount = torch.stack(y_surprise_rew_amount, dim=0)
    neuron_expected = torch.stack(neuron_expected, dim=0)
    neuron_surprise = torch.stack(neuron_surprise, dim=0)

    y = torch.cat([y_expected, y_surprise], dim=0)
    # (reward amount is label) surprise has negative
    label = torch.cat([y_expected_rew_amount, -y_surprise_rew_amount], dim=0)
    neuron = torch.cat([neuron_expected, neuron_surprise], dim=0)

    # (trial, time_window)
    y = torch.squeeze(y, dim=1)
    y = np.array(y.detach().cpu().numpy(), dtype=int)
    label = label.detach().cpu().numpy()
    neuron = neuron.detach().cpu().numpy()

    print("saving data!")
    scipy.io.savemat(
        os.path.join(params["out_path"], f"dopamine_eshel_reward_onset.mat"),
        {"y": y, "label": label, "neuron": neuron},
    )


if __name__ == "__main__":
    main()
