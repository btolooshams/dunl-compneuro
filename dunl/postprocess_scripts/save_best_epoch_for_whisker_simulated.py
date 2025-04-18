"""
Copyright (c) 2025 Bahareh Tolooshams

save whisker simulation - model characterization

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse

import sys

sys.path.append("../dunl/")

import datasetloader, lossfunc


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--code-sparse-regularization",
        type=float,
        help="lam reg",
        default=0.03,
    )
    parser.add_argument(
        "--kernel-length-ms", type=int, help="kernel length in ms", default=500
    )
    parser.add_argument(
        "--time-bin-resolution-list",
        type=int,
        help="time bin resolution list",  # see main
        default=[
            5,
            10,
            25,
            50,
        ],  # I assume the original resolution of data is 1 ms, so this would be 25 ms.
    )
    parser.add_argument(
        "--num-trials-list",
        type=int,
        help="number of trials list",  # see main
        # default=[25, 50, 100, 250, 500, 1000],
        default=[1000, 500, 250, 100],
    )
    parser.add_argument(
        "--baseline-mean-list",
        type=float,
        help="baseline mean across neurons",  # see main
        default=[
            -6.2126,  # 2 Hz
            -5.2933,  # 5 Hz
            -4.8203,  # 8 Hz
            -4.4988,  # 11 Hz
            -4.2546,  # 14 Hz
            -4.0574,  # 17 Hz
            # -3.8918, # 20 Hz
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

    baseline_in_Hz = dict()
    baseline_in_Hz["-6.2126"] = 2
    baseline_in_Hz["-5.2933"] = 5
    baseline_in_Hz["-4.8203"] = 8
    baseline_in_Hz["-4.4988"] = 11
    baseline_in_Hz["-4.2546"] = 14
    baseline_in_Hz["-4.0574"] = 17
    baseline_in_Hz["-3.8918"] = 20

    # baseline_in_Hz_list = [2, 5, 8, 11, 14, 17]
    res_path_dict_known = get_result_path(params_init, baseline_in_Hz, code_supp=True)
    res_path_dict_unknown = get_result_path(
        params_init, baseline_in_Hz, code_supp=False
    )

    # trials, bin, baseline
    save_best_epoch(params_init, res_path_dict_known)
    save_best_epoch(params_init, res_path_dict_unknown)


def get_result_path(params, baseline_in_Hz, code_supp):
    res_path_dict = dict()

    if code_supp:
        random_date = "2023_07_25_02_23_25"
    else:
        random_date = "2023_07_24_23_12_31"

    if code_supp:
        event_status = "knownEvent"
    else:
        event_status = "unknownEventKnownNumEvent"

    lam = str(params["code_sparse_regularization"]).replace(".", "p")

    for num_trials in params["num_trials_list"]:
        res_path_dict["num_trials_{}".format(num_trials)] = dict()

        for time_bin_resolution in params["time_bin_resolution_list"]:
            res_path_dict["num_trials_{}".format(num_trials)][
                "time_bin_resolution_{}".format(time_bin_resolution)
            ] = dict()

            kernel_length = int(params["kernel_length_ms"] / time_bin_resolution)

            for baseline_mean in params["baseline_mean_list"]:
                baseline_in_Hz_curr = baseline_in_Hz["{}".format(baseline_mean)]

                res_path = f"../results/simulated_1neurons_{num_trials}trials_{time_bin_resolution}msbinres_{baseline_in_Hz_curr}Hzbaseline_kernellength{kernel_length}_{event_status}_lam{lam}_{random_date}"
                res_path_dict["num_trials_{}".format(num_trials)][
                    "time_bin_resolution_{}".format(time_bin_resolution)
                ]["baseline_mean_{}".format(baseline_mean)] = res_path

    return res_path_dict


def save_best_epoch(params_init, res_path_dict, device="cpu"):
    for num_trials in params_init["num_trials_list"]:
        for time_bin_resolution in params_init["time_bin_resolution_list"]:
            for baseline_mean in params_init["baseline_mean_list"]:
                res_path = res_path_dict["num_trials_{}".format(num_trials)][
                    "time_bin_resolution_{}".format(time_bin_resolution)
                ]["baseline_mean_{}".format(baseline_mean)]

                print(res_path)

                # take parameters from the result path
                params = pickle.load(
                    open(os.path.join(res_path, "params.pickle"), "rb")
                )
                for key in params_init.keys():
                    params[key] = params_init[key]

                if params["data_path"] == "":
                    data_folder = params["data_folder"]
                    filename_list = os.listdir(data_folder)
                    data_path_list = [
                        f"{data_folder}/{x}"
                        for x in filename_list
                        if "trainready.pt" in x
                    ]
                else:
                    data_path_list = params["data_path"]

                data_folder = data_path_list[0].split(data_path_list[0].split("/")[-1])[
                    0
                ]

                # load test folder  ------------------------------------------------#
                data_train_filename = data_path_list[0].split("/")[-1]
                data_test_filename_first = data_train_filename.split(
                    "_{}trials".format(num_trials)
                )[0]
                data_test_filename_second = data_train_filename.split("trials")[-1]
                data_test_filename = (
                    f"{data_test_filename_first}_100trials{data_test_filename_second}"
                )
                test_path = os.path.join(
                    data_folder, "test_{}".format(data_test_filename)
                )
                test_dataset = datasetloader.DUNLdataset(test_path)
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    shuffle=False,
                    batch_size=128,
                    num_workers=4,
                )

                # create folders -------------------------------------------------------#
                model_path = os.path.join(
                    res_path,
                    "model",
                    "model_final.pt",
                )

                # load model ------------------------------------------------------#
                test_loss_best_epoch = 9999999
                best_epoch = 0
                if params["train_num_epochs"] == 100:
                    period = int(np.floor(params["train_num_epochs"] / 10))
                elif params["train_num_epochs"] == 60:
                    period = int(np.floor(params["train_num_epochs"] / 5))
                else:
                    period = 1
                print("period", period)
                for epoch in range(0, params["train_num_epochs"], period):
                    model_path = os.path.join(
                        res_path,
                        "model",
                        "model_epoch{}.pt".format(epoch),
                    )
                    net = torch.load(model_path, map_location=device)
                    net.to(device)
                    net.eval()

                    test_loss = get_rec_err(net, test_loader, params, device)

                    if test_loss <= test_loss_best_epoch:
                        test_loss_best_epoch = test_loss
                        best_epoch = epoch

                    print(
                        epoch,
                        "best epoch",
                        best_epoch,
                        "test_loss_best_epoch",
                        test_loss_best_epoch,
                    )

                print(
                    params["train_num_epochs"],
                    "best epoch",
                    best_epoch,
                    "test_loss_best_epoch",
                    test_loss_best_epoch,
                )

                torch.save(os.path.join(res_path, "best_epoch.pt"), best_epoch)


def get_rec_err(net, dataloader, params, device="cpu"):
    criterion = lossfunc.DUNL1DLoss(params["model_distribution"])

    loss_all = list()
    for idx, (y_load, x_load, a_load, type_load) in tqdm(
        enumerate(dataloader), disable=True
    ):
        # this is to collapse the neurons into the trial dim (as we are sharing kernel among neurons)
        # we do this to make sure that all neurons are present in each batch. So one example in batch correpsond to all neurons from one trial
        y = torch.reshape(
            y_load, (int(y_load.shape[0] * y_load.shape[1]), 1, y_load.shape[2])
        )
        a = torch.reshape(
            a_load, (int(a_load.shape[0] * a_load.shape[1]), 1, a_load.shape[2])
        )
        # repeat x for how many neurons are they into the 0 (trial) dim
        x = torch.repeat_interleave(x_load, a_load.shape[1], dim=0)

        # send data to device (cpu or gpu)
        y = y.to(device)
        x = x.to(device)
        a = a.to(device)

        if params["code_supp"]:
            x_code_supp = x
        else:
            x_code_supp = None

        # forward encoder
        xhat, a_est = net.encode(y, a, x_code_supp)
        # forward decoder
        hxmu = net.decode(xhat, a_est)

        if params["model_distribution"] == "binomial":
            yhat = torch.sigmoid(hxmu)
        else:
            raise NotImplementedError("model distribution is not implemented")

        loss_ae = criterion(y, yhat)

        print("loss_ae", loss_ae.item())

        loss_all.append(loss_ae.item())

    loss_all = np.mean(loss_all)

    return loss_all


if __name__ == "__main__":
    main()
