"""
Copyright (c) 2020 Bahareh Tolooshams

infer and save data

:author: Bahareh Tolooshams
"""

import torch
import os
import pickle
from tqdm import tqdm
import argparse

import sys

sys.path.append("../src/")

import datasetloader, model


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        ####
        default="../results/6000_3sparse_local_deconv_calscenario_longtrial_25trials_lam0.1_lamloss0.1_lamdecay1_code_topkTruesparse3period10_kernelsmooth0.015_knownsuppFalse_2023_11_02_17_48_26",
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

    # this is make sure the inference would be on full eshel data
    if (
        "2000_1sparse_local_deconv_calscenario_shorttrial_structured"
        in params_init["res_path"]
    ):
        params_init["test_data_path"] = [
            "../data/local-deconv-calscenario-shorttrial-structured-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_nov_general_format_processed_kernellength16_kernelnum2_trainready.pt"
        ]

    elif "6000_3sparse_local_deconv_calscenario_longtrial" in params_init["res_path"]:
        params_init["test_data_path"] = [
            "../data/local-deconv-calscenario-longtrial-simulated/test_simulated_1neurons_500trials_25msbinres_8Hzbaseline_long_general_format_processed_kernellength16_kernelnum2_trainready.pt"
        ]

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    data_path_list = params["test_data_path"]

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
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "postprocess",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device)
    net.eval()

    # go over data -------------------------------------------------------#
    compute_and_save(dataloader_list, net, params, out_path, device=device)

    print(f"infered and saved y, x, xhat, label. data is saved at {out_path}")


def compute_and_save(
    dataloader_list,
    net,
    params,
    out_path,
    device="cpu",
):
    for dataloader in dataloader_list:
        datafile_name = dataloader.dataset.data_path.split("/")[-1].split(".pt")[0]

        y_list = list()
        x_list = list()
        xhat_list = list()
        label_list = list()
        yhat_list = list()
        deconv_0_list = list()
        deconv_1_list = list()

        for idx, (y, x, a, label) in tqdm(
            enumerate(dataloader), disable=params["tqdm_prints_inside_disable"]
        ):
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
            xhat_out, a_est = net.encode(y_in, a_in, x_code_supp)

            # forward decoder
            hxmu_out = net.decode(xhat_out, a_est)

            # forward decoder
            xhat_0 = xhat_out.clone()
            xhat_0[:, 1, :] = 0
            xhat_1 = xhat_out.clone()
            xhat_1[:, 0, :] = 0

            hxmu_out_0 = net.decode(xhat_0, a_est)
            hxmu_out_1 = net.decode(xhat_1, a_est)

            if params["model_distribution"] == "binomial":
                yhat_out = torch.sigmoid(hxmu_out)
                deconv_out_0 = torch.sigmoid(hxmu_out_0)
                deconv_out_1 = torch.sigmoid(hxmu_out_1)
            else:
                raise NotImplementedError("model distribution is not implemented")

            # move the neuron axis back
            xhat = (
                torch.reshape(
                    xhat_out,
                    (y.shape[0], y.shape[1], xhat_out.shape[1], xhat_out.shape[2]),
                )
                .detach()
                .clone()
            )

            # move the neuron axis back
            yhat = (
                torch.reshape(yhat_out, (y.shape[0], y.shape[1], y.shape[2]))
                .detach()
                .clone()
            )
            # move the neuron axis back
            deconv_0 = (
                torch.reshape(deconv_out_0, (y.shape[0], y.shape[1], y.shape[2]))
                .detach()
                .clone()
            )
            deconv_1 = (
                torch.reshape(deconv_out_1, (y.shape[0], y.shape[1], y.shape[2]))
                .detach()
                .clone()
            )

            x_list.append(x)
            y_list.append(y)
            xhat_list.append(xhat)
            yhat_list.append(yhat)
            deconv_0_list.append(deconv_0)
            deconv_1_list.append(deconv_1)

        # (trials, 1, time)
        y_list = torch.cat(y_list, dim=0)
        # (trials, kernels, time)
        x_list = torch.cat(x_list, dim=0)
        # (trials, neurons, kernels, time)
        xhat_list = torch.cat(xhat_list, dim=0)
        # (trials, 1, time)
        yhat_list = torch.cat(yhat_list, dim=0)
        deconv_0_list = torch.cat(deconv_0_list, dim=0)
        deconv_1_list = torch.cat(deconv_1_list, dim=0)

        if 1:
            torch.save(
                xhat_list,
                os.path.join(out_path, "test_xhat_{}.pt".format(datafile_name)),
            )
            torch.save(
                y_list, os.path.join(out_path, "test_y_{}.pt".format(datafile_name))
            )
            torch.save(
                x_list, os.path.join(out_path, "test_x_{}.pt".format(datafile_name))
            )
            torch.save(
                yhat_list,
                os.path.join(out_path, "test_ratehat_{}.pt".format(datafile_name)),
            )
            torch.save(
                deconv_0_list,
                os.path.join(out_path, "test_deconv_0_{}.pt".format(datafile_name)),
            )
            torch.save(
                deconv_1_list,
                os.path.join(out_path, "test_deconv_1_{}.pt".format(datafile_name)),
            )


if __name__ == "__main__":
    main()
