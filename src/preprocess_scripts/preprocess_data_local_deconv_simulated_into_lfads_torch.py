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
        "--data-path-list",
        type=list,
        help="data path list",
        default=[
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_10trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_25trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_50trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_100trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_200trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_400trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
            "../data/local-deconv-calscenario-simulated/simulated_1neurons_800trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt",
        ],
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/local-deconv-calscenario-simulated/lfads",
    )
    parser.add_argument(
        "--data-fname-stem",
        type=str,
        help="data_fname_stem",
        default="local_deconv_calscenario_for_lfads_unstructured",
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
        "--train-percentage",
        type=float,
        help="train percentage",
        default=0.85,
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

    out_path = params["out_path"]
    data_fname_stem = params["data_fname_stem"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # create datasets -------------------------------------------------------#
    for data_path_cur in params["data_path_list"]:
        print(data_path_cur)

        num_trials = int(data_path_cur.split("neurons_")[-1].split("trials")[0])

        dataset = datasetloader.DUNLdatasetwithRaster(data_path_cur)

        y = dataset.raster
        label = dataset.type

        print("y", y.shape)
        print("label", label.shape)

        # (trial, time_window, 1)
        # f This is from LFADS
        # ExTxD, E - # examples, T - # time steps, D - # dimensions in data.
        y = torch.unsqueeze(torch.squeeze(y, dim=1), dim=-1)
        y = np.array(y.detach().cpu().numpy(), dtype=int)
        label = label.detach().cpu().numpy()

        shuffled_indices = np.arange(0, y.shape[0], 1)
        np.random.shuffle(shuffled_indices)

        num_train = int(params["train_percentage"] * len(shuffled_indices))
        num_val = len(shuffled_indices) - num_train

        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:]

        print("total number of trials:", y.shape[0])
        print("train:", num_train)
        print("val", num_val)

        num_train_modified = int(num_train)
        train_indices_modified = train_indices[:num_train_modified]

        print("train modified:", num_train_modified)

        data = {
            "train_encod_data": y[train_indices_modified],
            "train_recon_data": y[train_indices_modified],
            "valid_encod_data": y[val_indices],
            "valid_recon_data": y[val_indices],
            "readout.modules.0.out_features": 1,
        }

        N = 0
        dataset_dict = {}
        dataset_name = "dataset_N" + str(N)
        dataset_dict[dataset_name] = data

        full_name_stem = os.path.join(
            out_path, f"{data_fname_stem}_{num_trials}trials_torch"
        )
        for s, data_dict in dataset_dict.items():
            data_fname = full_name_stem + "_" + s + ".h5"

            dir_name = os.path.dirname(data_fname)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

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
