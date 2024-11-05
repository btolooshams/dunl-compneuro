"""
Copyright (c) 2020 Bahareh Tolooshams

train share kernel across neurons

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
import matplotlib.pyplot as plt
import time


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
        default="./local_orthkernels_simulated_config_for_noisy.yaml",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        help="number of trials",
        default=500,
    )
    parser.add_argument(
        "--bin-res",
        type=int,
        help="bin res",
        default=5,
    )
    parser.add_argument(
        "--noise-list",
        type=int,
        help="number of unrolling list",  # see main
        default=[0.8, 0.7, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main(params, random_date):
    print("Train DUNL on neural data for local orthkernels simulated.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    if params["code_q_regularization_matrix_path"]:
        params["code_q_regularization_matrix"] = (
            torch.load(params["code_q_regularization_matrix_path"]).float().to(device)
        )

    if not params["share_kernels_among_neurons"]:
        raise NotImplementedError(
            "This script is for sharing kernels among neurons. Set share_kernels_among_neurons=True!"
        )

    print("Exp: {}".format(params["exp_name"]))

    # create folder for results ---------------------------------------------#
    out_path = os.path.join(
        "..", "results", "{}_{}".format(params["exp_name"], random_date)
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
    if params["enable_board"]:
        writer = tensorboardX.SummaryWriter(os.path.join(params["out_path"]))
        writer.add_text("params", str(params))
        writer.flush()

    # create dataset and dataloaders ----------------------------------------#
    train_dataset_list = []
    train_loader_list = []
    val_dataset_list = []
    val_loader_list = []

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    total_num_val = 0
    for data_path_cur in data_path_list:
        print(data_path_cur)
        dataset = datasetloader.DUNLdataset(data_path_cur)

        num_data = len(dataset)
        num_val = np.floor(num_data * (1 - params["train_val_split"]))
        num_val = int(np.minimum(num_val, params["train_val_max"]))
        if total_num_val + num_val > params["train_val_max"]:
            print(f"reached total {total_num_val}.")
            num_val = np.maximum(params["train_val_max"] - total_num_val, 0)
        total_num_val += num_val
        print(f"num_val is {num_val}")
        num_train = num_data - num_val
        if num_val > 0:
            if params["train_val_split"] < 1:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset,
                    [num_train, num_val],
                )
                val_dataset_list.append(val_dataset)
            else:
                train_dataset = dataset
        else:
            train_dataset = dataset

        train_dataset_list.append(train_dataset)

    for train_dataset in train_dataset_list:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=params["train_data_shuffle"],
            batch_size=params["train_batch_size"],
            num_workers=params["train_num_workers"],
        )
        train_loader_list.append(train_loader)

    if val_dataset_list:
        for val_dataset in val_dataset_list:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=params["train_data_shuffle"],
                batch_size=params["train_batch_size"],
                num_workers=params["train_num_workers"],
            )
            val_loader_list.append(val_loader)

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

    net = model.DUNL1D(params, kernel_init)
    net.to(device)

    if params["kernel_nonneg"]:
        net.nonneg_kernel(params["kernel_nonneg_indicator"])
    if params["kernel_normalize"]:
        net.normalize_kernel()

    # create optimizer and scheduler ---------------------------------------#
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
    best_val_loss = float("inf")

    noise_sampler = torch.distributions.binomial.Binomial(
        total_count=params["bin_res"], probs=params["noise_level"]
    )
    start_time = time.time()
    step = 0
    while step <= params["train_num_steps"]:
        net.train()

        # instead of mixing the datasets, I go over them sequentially
        for train_loader in train_loader_list:
            for idx, (y_load, x_load, a_load, type_load) in tqdm(
                enumerate(train_loader), disable=params["tqdm_prints_inside_disable"]
            ):
                # we want to group the firing rate of the neurons, so the model will take (b, group, 1, time)
                y = y_load.unsqueeze(dim=2)
                a = a_load.unsqueeze(dim=2)

                ## adding noise to y
                noise = noise_sampler.sample(sample_shape=y.shape) / params["bin_res"]
                sign = torch.randint(low=0, high=2, size=noise.shape) * 2 - 1

                # plt.figure()
                # plt.plot(torch.mean(y[0,:,0]+ sign[0,:,0] * noise[0,:,0], dim=0))
                # plt.plot(torch.mean(y[0,:,0], dim=0))
                # plt.savefig("../debig_noise.png")
                # plt.close()

                # exit()

                y = y + sign * noise
                y[y < 0] = 0
                y[y > 1] = 1

                # give x as is, the mdoel will take care of the repeat to group neurons.
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

                # backward to update kernels

                if loss > 1e5:
                    print("encoder has diverged!")
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # project kernels and normalize
                if params["kernel_nonneg"]:
                    net.nonneg_kernel(params["kernel_nonneg_indicator"])
                if params["kernel_normalize"]:
                    net.normalize_kernel()

                step += 1

                scheduler.step()

                if params["enable_board"]:
                    writer.add_scalar("loss/train", loss.item(), step)
                    writer.add_scalar("loss/train_ae", loss_ae.item(), step)
                    writer.flush()

                    if (step) % params["val_period"] == 0:
                        net.eval()
                        if val_loader_list:
                            writer, val_loss = boardfunc.log_loss_noisy(
                                writer,
                                step,
                                net,
                                val_loader_list,
                                criterion,
                                params,
                                noise_sampler,
                                "val",
                                False,
                                device=device,
                            )
                            if val_loss <= best_val_loss:
                                print(f"step {step}: val loss is improved!")
                                best_val_loss = val_loss
                                torch.save(
                                    net,
                                    os.path.join(
                                        out_path, "model", "model_best_val.pt"
                                    ),
                                )
                                best_val_step = step

                        writer.flush()

                    if (step) % params["log_fig_epoch_period"] == 0:
                        writer = boardfunc.log_kernels(writer, step, net)
                        writer = boardfunc.log_data_inputrec_groupneuralfirings(
                            writer, step, y, yhat, params["model_distribution"]
                        )
                        if params["code_supp"]:
                            writer = boardfunc.log_data_code_groupneuralfirings(
                                writer,
                                step,
                                xhat,
                                x,
                                max_neurons=10,
                            )
                        else:
                            if torch.sum(torch.abs(x)) > 0:
                                writer = boardfunc.log_data_code_groupneuralfirings(
                                    writer,
                                    step,
                                    xhat,
                                    x,
                                    max_neurons=10,
                                )
                            else:
                                writer = boardfunc.log_data_code_groupneuralfirings(
                                    writer,
                                    step,
                                    xhat,
                                    max_neurons=10,
                                )

                if step > params["train_num_steps"]:
                    break

                net.train()

            if step > params["train_num_steps"]:
                break

    torch.save(net, os.path.join(out_path, "model", "model_final.pt"))

    end_time = time.time()
    train_time = end_time - start_time
    print("total time is:", train_time)

    torch.save(train_time, os.path.join(out_path, "train_time.pt"))
    torch.save(best_val_step, os.path.join(out_path, "best_val_step.pt"))


if __name__ == "__main__":
    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    a = 200
    seed_list = np.linspace(9201, 9201 + 5 * a - 1, a)
    seed_list = [int(x) for x in seed_list]
    print(seed_list)

    baseline_in_Hz = dict()
    baseline_in_Hz["-6.2126"] = 2
    baseline_in_Hz["-5.2933"] = 5
    baseline_in_Hz["-4.8203"] = 8
    baseline_in_Hz["-4.4988"] = 11
    baseline_in_Hz["-4.2546"] = 14
    baseline_in_Hz["-4.0574"] = 17
    baseline_in_Hz["-3.8918"] = 20
    baseline_in_Hz["0"] = 500

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

    params["bin_res"] = params_init["bin_res"]
    params["num_trials"] = params_init["num_trials"]

    num_trials = params["num_trials"]
    unrolling_num = params["unrolling_num"]
    bin_res = params["bin_res"]
    for noise_level in params_init["noise_list"]:
        num_trials_in_each_file = 10
        num_loops = int(num_trials / num_trials_in_each_file)
        data_path_name_list = list()
        for loop in range(num_loops):
            seed = seed_list[loop]
            data_path_name = f"../data/local-orthkernels-simulated/simulated_1000neurons_{num_trials_in_each_file}trials_{bin_res}msbinres_{500}Hzbaseline_seed{seed}_long_general_format_processed_kernellength80_kernelnum5_trainready.pt"

            data_path_name_list.append(data_path_name)

            print(data_path_name)

        # code_supp = params["code_supp"]
        # code_topk = params["code_topk"]
        # code_topk_sparse = params["code_topk_sparse"]
        # code_topk_period = params["code_topk_period"]
        # lam = params["code_sparse_regularization"]
        # lam_loss = params["code_l1loss_bp_penalty_weight"]
        # lam_decay = params["code_sparse_regularization_decay"]
        # qreg = params["code_q_regularization"]
        # qreg_scale = params["code_q_regularization_scale"]
        # kernel_smoother_penalty_weight = params["kernel_smoother_penalty_weight"]
        # optimizer_lr_step = params["optimizer_lr_step"]
        kernel_num = params["kernel_num"]

        params["noise_level"] = noise_level
        params["data_path"] = data_path_name_list
        params[
            "exp_name"
        ] = f"orth_{num_trials}trials_{kernel_num}kernel_num_{unrolling_num}unrolling_1000epochs_01noise{noise_level}"

        main(params, random_date)
