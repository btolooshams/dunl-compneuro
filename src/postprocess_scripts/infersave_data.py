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
        default="../results/whisker_05msbinres_lamp03_topk18_smoothkernelp003_2023_07_19_00_03_18",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="batch size",
        default=128,
    )
    parser.add_argument(
        "--train-test-flag",
        type=bool,
        help="True for train, False for test",
        default=True,
    )
    parser.add_argument(
        "--infer-init-model",
        type=bool,
        help="True to infer on init net instead of learned model",
        default=False,
    )
    parser.add_argument(
        "--infer-init-model-code-sparse-regularization-list",
        type=list,
        help="list of lam to infer the init model",
        default=[0, 0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.15],
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

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    # this is make sure the inference would be on full eshel data
    if (
        params_init["res_path"]
        == "../results/dopaminespiking_25msbin_kernellength24_kernelnum3_codefree_kernel111_limiteddata0p1_smoothkernel_0p0005_2023_08_12_22_09_40"
    ):
        params["data_folder"] = "../data/dopamine-spiking-eshel-uchida"

    if params["train_test_flag"]:
        if params["data_path"] == "":
            data_folder = params["data_folder"]
            filename_list = os.listdir(data_folder)
            data_path_list = [
                f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
            ]
        else:
            data_path_list = params["data_path"]
    else:
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
    if params["infer_init_model"]:
        kernel_init = torch.load(params["kernel_initialization"])
        # initial model before training
        net = model.DUNL1D(params, kernel_init)
    else:
        net = torch.load(model_path, map_location=device)
    net.eval()

    # go over data -------------------------------------------------------#
    if params["infer_init_model"]:
        for code_sparse_regularization in params[
            "infer_init_model_code_sparse_regularization_list"
        ]:
            net.code_sparse_regularization = torch.tensor(
                code_sparse_regularization, device=device
            )
            compute_and_save(
                dataloader_list,
                net,
                params,
                out_path,
                code_sparse_regularization,
                device,
            )

    else:
        compute_and_save(dataloader_list, net, params, out_path, device)

    print(f"infered and saved y, x, xhat, label. data is saved at {out_path}")


def compute_and_save(
    dataloader_list,
    net,
    params,
    out_path,
    code_sparse_regularization=None,
    device="cpu",
):
    for dataloader in dataloader_list:
        datafile_name = dataloader.dataset.data_path.split("/")[-1].split(".pt")[0]

        y_list = list()
        x_list = list()
        xhat_list = list()
        label_list = list()

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

            # move the neuron axis back
            xhat = (
                torch.reshape(
                    xhat_out,
                    (y.shape[0], y.shape[1], xhat_out.shape[1], xhat_out.shape[2]),
                )
                .detach()
                .clone()
            )

            x_list.append(x)
            y_list.append(y)
            xhat_list.append(xhat)
            label_list.append(label)

        # (trials, 1, time)
        y_list = torch.cat(y_list, dim=0)
        # (trials, kernels, time)
        x_list = torch.cat(x_list, dim=0)
        # (trials, neurons, kernels, time)
        xhat_list = torch.cat(xhat_list, dim=0)
        # (trials)
        label_list = torch.cat(label_list, dim=0)

        if 1:
            if params["infer_init_model"]:
                torch.save(
                    xhat_list,
                    os.path.join(
                        out_path,
                        "xhat_init_reg{}_{}.pt".format(
                            code_sparse_regularization, datafile_name
                        ),
                    ),
                )
            else:
                torch.save(
                    xhat_list,
                    os.path.join(out_path, "xhat_{}.pt".format(datafile_name)),
                )
                torch.save(
                    y_list, os.path.join(out_path, "y_{}.pt".format(datafile_name))
                )
                torch.save(
                    x_list, os.path.join(out_path, "x_{}.pt".format(datafile_name))
                )
                torch.save(
                    label_list,
                    os.path.join(out_path, "label_{}.pt".format(datafile_name)),
                )


if __name__ == "__main__":
    main()
