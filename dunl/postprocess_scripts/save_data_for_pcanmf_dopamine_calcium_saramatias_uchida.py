"""
Copyright (c) 2025 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import pickle
import argparse


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.0, 0.3, 0.5, 1.2, 2.5, 5.0, 8.0, 11.0],
    )
    parser.add_argument(
        "--window-dur",
        type=int,
        help="window duration to get average activity",
        default=60,  # this is after time bin resolution
    )
    parser.add_argument(
        "--save-only-sur",
        type=bool,
        help="save only surprise trials",
        default=False,
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

    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # load data -------------------------------------------------------#

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        y = torch.load(os.path.join(postprocess_path, "y_{}.pt".format(datafile_name)))
        x = torch.load(os.path.join(postprocess_path, "x_{}.pt".format(datafile_name)))
        label = torch.load(
            os.path.join(postprocess_path, "label_{}.pt".format(datafile_name))
        )

        num_trials = y.shape[0]

        yavg = list()
        rew_amount = list()

        # go over all trials
        for i in range(num_trials):
            yi = y[i]
            xi = x[i]
            labeli = label[i]

            if labeli == 0:
                continue
            else:
                # skip if it's a expected trial
                if params["save_only_sur"]:
                    cue_flag = torch.sum(torch.abs(xi[1]), dim=-1).item()
                    if cue_flag:
                        # expected trial hence, skip
                        continue

                # reward presence
                reward_onset = np.where(xi[2] > 0)[-1][0]

                y_curr = yi[:, reward_onset : reward_onset + params["window_dur"]]
                yavg.append(y_curr)
                rew_amount.append(labeli)

        # (neurons, time, trials)
        yavg = torch.stack(yavg, dim=-1).clone().detach().cpu().numpy()
        rew_amount = np.array(rew_amount)
        if params["save_only_sur"]:
            np.save(
                os.path.join(
                    postprocess_path,
                    "y_for_pcanmf_{}_only_sur.npy".format(datafile_name),
                ),
                yavg,
            )
            np.save(
                os.path.join(
                    postprocess_path,
                    "label_for_pcanmf_{}_only_sur.npy".format(datafile_name),
                ),
                rew_amount,
            )
        else:
            np.save(
                os.path.join(
                    postprocess_path, "y_for_pcanmf_{}.npy".format(datafile_name)
                ),
                yavg,
            )
            np.save(
                os.path.join(
                    postprocess_path, "label_for_pcanmf_{}.npy".format(datafile_name)
                ),
                rew_amount,
            )


if __name__ == "__main__":
    main()
