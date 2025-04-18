"""
Copyright (c) 2025 Bahareh Tolooshams

save data pca nmf

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import configmypy
import os
import argparse

import sys

sys.path.append("../dunl/")

import datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-folder",
        type=str,
        help="config folder",
        default="../config",
    )
    parser.add_argument(
        "--config-filename",
        type=str,
        help="config filename",
        default="./local_speed_simulated_config.yaml",
    )
    parser.add_argument(
        "--num-trials-list",
        type=int,
        help="number of trials list",  # see main
        default=[250, 50, 100, 25, 500, 750, 1000],
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main(params):
    print("Train DUNL on neural data for local speed simulated.")

    params["window_dur"] = 16

    # create dataset and dataloaders ----------------------------------------#

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    data_path_cur = data_path_list[0]
    print(data_path_cur)
    train_dataset = datasetloader.DUNLdataset(data_path_cur)

    y = train_dataset.y
    x = train_dataset.x

    num_trials = y.shape[0]
    num_neurons = y.shape[1]
    num_kernels = x.shape[1]

    yavg = list()
    rew_amount = list()

    # go over all trials
    for i in range(num_trials):

        xi = x[i]
        yi = y[i]

        for kernel_ctr in range(num_kernels):
            onset = np.where(xi[kernel_ctr] > 0)[-1]
            for on in onset:
                y_curr = yi[:, on : on + params["window_dur"]]
                yavg.append(y_curr)

    yavg = torch.stack(yavg, dim=0)

    print("yavg", yavg.shape)

    np.save(
        os.path.join(
            "../data/local-speed-simulated",
            "y_for_pcanmf_numtrials{}.npy".format(num_trials),
        ),
        yavg,
    )


if __name__ == "__main__":
    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                params_init["config_filename"],
                config_name="default",
                config_folder=params_init["config_folder"],
            ),
            configmypy.ArgparseConfig(
                infer_types=True, config_name=None, config_file=None
            ),
            configmypy.YamlConfig(config_folder=params_init["config_folder"]),
        ]
    )
    params = pipe.read_conf()
    params["config_folder"] = params_init["config_folder"]
    params["config_filename"] = params_init["config_filename"]

    for num_trials in params_init["num_trials_list"]:
        print("num_trials", num_trials)
        params["num_trials"] = num_trials

        data_path_name = f"../data/local-speed-simulated/simulated_100neurons_{num_trials}trials_25msbinres_14Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"

        params["data_path"] = [data_path_name]

        main(params)
