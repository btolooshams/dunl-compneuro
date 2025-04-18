"""
Copyright (c) 2025 Bahareh Tolooshams

plot recompose data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse

import sys

sys.path.append("../dunl/")

import datasetloader, utils


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path-partial",
        type=str,
        help="res path partial",
        default="../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured",
        # default="../results/6000_3sparse_local_deconv_calscenario_longtrial",
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
        "--color-list",
        type=list,
        help="color decomposition list",
        default=[
            "blue",
            "red",
        ],  #
    )
    parser.add_argument(
        "--swap-kernel",
        type=bool,
        help="bool to swap kernel",
        default=True,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def compute_r2_score(spikes, rate_hat):
    # compute r2 score
    ss_res = np.mean((spikes - rate_hat), axis=1) ** 2
    ss_tot = np.var(spikes)

    r2_fit = 1 - ss_res / ss_tot

    return np.mean(r2_fit)


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    for trials in [25, 50, 100, 200, 400, 800, 1600]:
        # init parameters -------------------------------------------------------#
        print("init parameters.")
        params_init = init_params()

        # this is make sure the inference would be on full eshel data
        if (
            params_init["res_path_partial"]
            == f"../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured"
        ):
            params_init[
                "res_path"
            ] = f"../results/2000_1sparse_local_deconv_calscenario_shorttrial_structured_{trials}trials_25msbinres_kernellength16_kernelnum2_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse1period10_kernelsmooth0.015_knownsuppFalse_2023_10_27_07_45_35"

            params_init["test_data_path"] = [
                "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"
            ]

        elif (
            params_init["res_path_partial"]
            == f"../results/6000_3sparse_local_deconv_calscenario_longtrial"
        ):
            params_init[
                "res_path"
            ] = f"../results/6000_3sparse_local_deconv_calscenario_longtrial_{trials}trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_18"
            params_init["test_data_path"] = [
                "../data/local-deconv-calscenario-longtrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"
            ]

        # take parameters from the result path
        params = pickle.load(
            open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
        )
        for key in params_init.keys():
            params[key] = params_init[key]

        postprocess_path = os.path.join(
            params["res_path"],
            "postprocess",
        )

        data_path_list = params["test_data_path"]

        print("There {} dataset in the folder.".format(len(data_path_list)))

        # set time bin resolution -----------------------------------------------#
        data_dict = torch.load(data_path_list[0])
        params["time_bin_resolution"] = data_dict["time_bin_resolution"]

        # create datasets -------------------------------------------------------#
        dataset = datasetloader.DUNLdatasetwithRasterWithCodeRate(
            params["test_data_path"][0]
        )
        datafile_name = params["test_data_path"][0].split("/")[-1].split(".pt")[0]

        # create folders -------------------------------------------------------#

        codes = dataset.codes
        rate = dataset.rate

        # train_num_trials = len(dataset)

        y = torch.load(
            os.path.join(postprocess_path, "test_y_{}.pt".format(datafile_name))
        )
        y = y[:, 0, :]

        rate_hat = torch.load(
            os.path.join(postprocess_path, "test_ratehat_{}.pt".format(datafile_name))
        )
        rate_hat = rate_hat[:, 0, :]

        r2_score = compute_r2_score(y.numpy(), rate_hat.numpy())
        print(f"DUNL trial {trials}, r2 {r2_score}")


if __name__ == "__main__":
    main()
