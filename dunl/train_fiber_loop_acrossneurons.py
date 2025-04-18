"""
Copyright (c) 2025 Bahareh Tolooshams

train for fiber

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import configmypy
import os
import json
import pickle
import tensorboardX
from datetime import datetime
from tqdm import tqdm
import argparse

import model, lossfunc, boardfunc, datasetloader


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
        default="./dopamine_fiberphotometry_saramatias_uchida_config_1window_1kernel.yaml",
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main(params, random_date):
    print("Train DUNL on neural data (independent kernels for each neuron).")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    if params["code_q_regularization_matrix_path"]:
        params["code_q_regularization_matrix"] = (
            torch.load(params["code_q_regularization_matrix_path"]).float().to(device)
        )

    if params["share_kernels_among_neurons"]:
        raise NotImplementedError(
            "This script is for independent kernels among neurons. Set share_kernels_among_neurons=False!"
        )

    neuron_index = params["neuron_index"]
    number_of_window = params["number_of_window"]
    number_of_unrolling = params["unrolling_num"]

    print("Exp: {}".format(params["exp_name"]))

    # create dataset and dataloaders ----------------------------------------#
    print("create dataset and dataloader.")

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # create folder for results ---------------------------------------------#

    for data_path_cur in data_path_list:
        print(data_path_cur)

        dataset = datasetloader.DUNLdataset(data_path_cur)

        # go over the neurons sequentially -------------------------------------#
        # --------------------------------------------#
        # --------------------------------------------#
        # --------------------------------------------#
        out_path = os.path.join(
            "..",
            "results",
            "z_sara_bahareh_final",
            "{}_numwindow{}_neuron{}_kernellength{}_{}kernels_{}unroll_{}".format(
                params["exp_name"],
                params["number_of_window"],
                neuron_index,
                params["kernel_length"],
                params["kernel_num"],
                params["unrolling_num"],
                random_date,
            ),
        )
        params["out_path"] = out_path
        if not os.path.exists(params["out_path"]):
            os.makedirs(params["out_path"])
        if not os.path.exists(os.path.join(params["out_path"], "model")):
            os.makedirs(os.path.join(params["out_path"], "model"))
        if not os.path.exists(os.path.join(params["out_path"], "figures")):
            os.makedirs(os.path.join(params["out_path"], "figures"))

        # dump params  ---------------------------------------------------------#
        with open(os.path.join(params["out_path"], "params.txt"), "w") as file:
            params_clone = params.copy()
            params_clone["code_q_regularization_matrix"] = str(
                params["code_q_regularization_matrix"]
            )
            file.write(json.dumps(params_clone, sort_keys=True, separators=("\n", ":")))
        with open(os.path.join(params["out_path"], "params.pickle"), "wb") as file:
            pickle.dump(params_clone, file)

        # create board ----------------------------------------------------------#
        print("create board.")
        if params["enable_board"]:
            writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
            writer.add_text("params", str(params))
            writer.flush()

        # --------------------------------------------#
        # --------------------------------------------#
        # --------------------------------------------#

        dataset = datasetloader.DUNLdataset(data_path_cur)

        ########################################################
        ########################################################
        ######## these are hardcoded now! should be moved into how to make the data ready for training.
        print("This script has a hardcoded step for dataset processing.")
        print("these are hardcoded now! should be moved into pre-processing step.")
        dataset.y = torch.nan_to_num(dataset.y)
        dataset.a = torch.nan_to_num(dataset.a)

        dataset.y = dataset.y - torch.min(dataset.y, dim=-1, keepdims=True)[0]
        dataset.y /= torch.max(dataset.y, dim=-1, keepdims=True)[0]
        dataset.y = dataset.y - torch.mean(dataset.y, dim=-1, keepdims=True)

        dataset.y = dataset.y[0]
        dataset.y = (dataset.y).reshape((dataset.y.shape[0], number_of_window, -1))
        dataset.y = dataset.y.permute(1, 0, 2)
        dataset.num_data = dataset.y.shape[0]

        dataset.a = dataset.y[:, :, [0]].clone() * 0.0

        dataset.x = dataset.y.clone() * 0
        dataset.type = dataset.y[:, 0, 0].clone() * 0
        ########################################################
        ########################################################

        num_data = len(dataset)
        num_train = int(np.floor(num_data * params["train_val_split"]))
        if params["train_val_split"] < 1:
            raise NotImplementedError(
                "This script does not use validation dataset, set train_val_split=1 to use this script!"
            )
        else:
            train_dataset = dataset

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=params["train_data_shuffle"],
            batch_size=params["train_batch_size"],
            num_workers=params["train_num_workers"],
        )

        # create model ---------------------------------------------------------#
        if params["kernel_initialization"]:
            kernel_init = torch.load(params["kernel_initialization"])
            if params["kernel_initialization_needs_adjustment_of_time_bin_resolution"]:
                bin_res = int(kernel_init.shape[-1] / params["kernel_length"])
                kernel_init = np.add.reduceat(
                    kernel_init,
                    np.arange(0, kernel_init.shape[-1], bin_res),
                    axis=-1,
                )
        else:
            kernel_init = None

        print("create model.")
        net = model.DUNL1D(params, kernel_init)
        net.to(device)

        if params["kernel_nonneg"]:
            net.nonneg_kernel(params["kernel_nonneg_indicator"])
        if params["kernel_normalize"]:
            net.normalize_kernel()

        # create optimizer and scheduler ---------------------------------------#
        print("create optimizer and scheduler for training.")
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=params["optimizer_lr"],
            eps=params["optimizer_adam_eps"],
            weight_decay=params["optimizer_adam_weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params["optimizer_lr_step"],
            gamma=params["optimizer_lr_decay"],
        )

        # create loss criterion  ------------------------------------------------#
        criterion = lossfunc.DUNL1DLoss(params["model_distribution"])

        # create kernel smoother loss criterion
        if params["kernel_smoother"]:
            criterion_kernel_smoother = lossfunc.Smoothloss()

        # create l1 loss (mean) criterion
        if params["code_l1loss_bp"]:
            criterion_l1_code = lossfunc.l1loss()

        # train  ---------------------------------------------------------------#
        print("start training.")
        for epoch in tqdm(
            range(params["train_num_epochs"]), disable=params["tqdm_prints_disable"]
        ):
            net.train()

            total_train_loss_list = []
            total_train_loss_ae_list = []

            for idx, (y_load, x_load, a_load, type_load) in tqdm(
                enumerate(train_loader), disable=params["tqdm_prints_inside_disable"]
            ):
                # pick one neuron from the tensor (we are sequentially going over the neurons)
                y = y_load[:, [neuron_index], :]

                a = a_load[:, [neuron_index], :]

                # x is already shared among neurons (so no need to chage)
                x = x_load

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
                yhat = net.decode(xhat, a_est)

                # compute loss
                loss_ae = criterion(y, yhat)

                if params["kernel_smoother"]:
                    loss_kernel_smoother = criterion_kernel_smoother(net.get_param("H"))
                else:
                    loss_kernel_smoother = 0.0

                if params["code_l1loss_bp"]:
                    loss_l1_code = criterion_l1_code(xhat)
                else:
                    loss_l1_code = 0.0

                loss = (
                    loss_ae
                    + params["kernel_smoother_penalty_weight"] * loss_kernel_smoother
                    + params["code_l1loss_bp_penalty_weight"] * loss_l1_code
                )

                total_train_loss_list.append(loss.item())
                total_train_loss_ae_list.append(loss_ae.item())

                # backward to update kernels
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # project kernels and normalize
                if params["kernel_nonneg"]:
                    net.nonneg_kernel(params["kernel_nonneg_indicator"])
                if params["kernel_normalize"]:
                    net.normalize_kernel()

            scheduler.step()

            if (epoch + 1) % params["log_info_epoch_period"] == 0:
                total_train_loss = np.mean(total_train_loss_list)
                total_train_loss_ae = np.mean(total_train_loss_ae_list)
                print(
                    "total_train_loss",
                    total_train_loss,
                    "total_train_loss_ae",
                    total_train_loss_ae,
                )
                if params["enable_board"]:
                    writer.add_scalar("loss/train", total_train_loss, epoch)
                    writer.add_scalar("loss/train_ae", total_train_loss_ae, epoch)
                    writer.flush()

                if params["kernel_smoother"]:
                    if params["enable_board"]:
                        writer.add_scalar(
                            "loss/kernel_smoother", loss_kernel_smoother.item(), epoch
                        )
                        writer.flush()

            if (
                params["enable_board"]
                and (epoch + 1) % params["log_fig_epoch_period"] == 0
            ):
                writer = boardfunc.log_kernels(writer, epoch, net)
                writer = boardfunc.log_data_inputrec(
                    writer, epoch, y, yhat, params["model_distribution"]
                )
                if params["code_supp"]:
                    writer = boardfunc.log_data_code(writer, epoch, xhat, x)
                else:
                    if torch.sum(torch.abs(x)) > 0:
                        writer = boardfunc.log_data_code(writer, epoch, xhat, x)
                    else:
                        writer = boardfunc.log_data_code(writer, epoch, xhat)

            if (epoch + 1) % params["log_model_epoch_period"] == 0:
                torch.save(
                    net, os.path.join(out_path, "model", f"model_epoch{epoch}.pt")
                )
        torch.save(net, os.path.join(out_path, "model", "model_final.pt"))


if __name__ == "__main__":

    neuron_index_list = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    ]  # define the list here

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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

    for neuron_index in neuron_index_list:
        print("neuron_index", neuron_index)
        params["neuron_index"] = neuron_index

        main(params, random_date)
