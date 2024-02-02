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
import h5py

import sys

sys.path.append("../src/")

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-folder",
        type=str,
        help="data path",
        default="../data/dopamine-spiking-eshel-uchida/train",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/dopamine-spiking-eshel-uchida/lfads",
    )
    parser.add_argument(
        "--data-fname-stem",
        type=str,
        help="data_fname_stem",
        default="dopamine_spiking_eshel_uchida_for_lfads_traintest_separated_0p5train",
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
    parser.add_argument(
        "--train-percentage",
        type=float,
        help="train percentage",
        default=0.8125,
    )
    parser.add_argument(
        "--save-only-fraction-train",
        type=float,
        help="same only this fraction of train",
        default=0.5,
    )
    parser.add_argument(
        "--nreplications",
        type=int,
        help="nreplications",
        default=1,
    )
    parser.add_argument(
        "--compression",
        type=bool,
        help="compression",
        default=None,
    )
    parser.add_argument(
        "--use-json",
        type=bool,
        help="use json",
        default=False,
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
            shuffle=True,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )
        dataloader_list.append(dataloader)

    out_path = params["out_path"]
    data_fname_stem = params["data_fname_stem"]
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
                label[label_int == tmp_ctr] = reward

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

    # (trial, time_window, 1)
    # f This is from LFADS
    # ExTxD, E - # examples, T - # time steps, D - # dimensions in data.
    y = torch.unsqueeze(torch.squeeze(y, dim=1), dim=-1)
    y = np.array(y.detach().cpu().numpy(), dtype=int)
    label = label.detach().cpu().numpy()
    neuron = neuron.detach().cpu().numpy()

    shuffled_indices = np.arange(0, y.shape[0], 1)
    np.random.shuffle(shuffled_indices)

    num_train = int(params["train_percentage"] * len(shuffled_indices))
    num_val = len(shuffled_indices) - num_train

    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train:]

    print("total number of trials:", y.shape[0])
    print("train:", num_train)
    print("val", num_val)

    num_train_modified = int(num_train * params["save_only_fraction_train"])
    train_indices_modified = train_indices[:num_train_modified]

    print("train modified:", num_train_modified)

    data = {
        "train_data": y[train_indices_modified],
        "valid_data": y[val_indices],
        "train_label": label[train_indices_modified],
        "valid_label": label[val_indices],
        "train_neuron": neuron[train_indices_modified],
        "valid_neuron": neuron[val_indices],
        "train_percentage": params["train_percentage"],
        "save_only_fraction_train": params["save_only_fraction_train"],
        "nreplications": params["nreplications"],
    }

    N = 0
    dataset_dict = {}
    dataset_name = "dataset_N" + str(N)
    dataset_dict[dataset_name] = data

    full_name_stem = os.path.join(out_path, data_fname_stem)
    for s, data_dict in dataset_dict.items():
        data_fname = full_name_stem + "_" + s

        dir_name = os.path.dirname(data_fname)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if params["use_json"]:
            the_file = open(data_fname, "wb")
            json.dump(data_dict, the_file)
            the_file.close()
        else:
            try:
                with h5py.File(data_fname, "w") as hf:
                    for k, v in data_dict.items():
                        clean_k = k.replace("/", "_")
                        if clean_k is not k:
                            print(
                                "Warning: saving variable with name: ",
                                k,
                                " as ",
                                clean_k,
                            )
                        else:
                            print("Saving variable with name: ", clean_k)
                        hf.create_dataset(
                            clean_k, data=v, compression=params["compression"]
                        )
            except IOError:
                print("Cannot open %s for writing.", data_fname)
                raise


if __name__ == "__main__":
    main()
