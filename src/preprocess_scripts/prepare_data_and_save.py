"""
Copyright (c) 2025 Bahareh Tolooshams

preprocess data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import argparse


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        help="data folder",
        default="../data/local-deconv-calscenario-longtrial-simulated",
        # default="../data/local-deconv-calscenario-shorttrial-structured-simulated",
        # default="../data/local-deconv-calscenario-shorttrial-simulated",
        # default="../data/local-deconv-calscenario-simulated",
        # default="../data/local-deconv-simulated",
        # default="../data/whisker-simulated",
        # default="../data/whisker,
        # default="../data/dopamine-spiking-simulated",
        # default="../data/dopamine-spiking-eshel-uchida",
        # default="../data/dopamine-spiking-eshel-uchida/test",
        # default="../data/dopamine-spiking-eshel-uchida/train",
        # default="../data/dopamine-calcium-saramatias-uchida",
    )
    parser.add_argument(
        "--kernel-length",
        type=int,
        help="length of kernel samples in time",
        # default=60, # sara matias
        # default=25, # whisker at 5 ms resolution for 125 ms kernel
        # default=12, # whisker at 10 ms resolution for 120 ms kernel
        # default=24,  # for eshel at 25 ms resolution for 600 ms kernel
        # default=24,  # for simulated spiking at 25 ms resolution for 600 ms kernel
        # default=None,  # set to None for whisker-simulated (will infer from original file, only works for simulated)
        # default=24,  # for local deconv simulated spiking at 25 ms resolution for 600 ms kernel
        # default=16,  # for local deconv simulated spiking at 25 ms resolution for 400 ms kernel
        # default=16,  # for local deconv short trial simulated spiking at 25 ms resolution for 400 ms kernel
        # default=16,  # for local deconv short trial simulated spiking at 25 ms resolution for 400 ms kernel
        default=16,  # for local deconv long trial simulated spiking at 25 ms resolution for 400 ms kernel
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        # default=5, # dopamine-calcium-saramatias-uchida
        # default=1, # whisker
        # default=3, # dopamine-spiking-eshel-uchida
        # default=3  # dopamine-spiking-simulated
        # default=1,  # whisker-simulated
        # default=2,  # local-deconv-simulated
        # default=2,  # local-deconv-calscenario-simulated
        # default=2,  # local-deconv-calscenario-shorttrial-simulated
        # default=2,  # local-deconv-calscenario-shorttrial-structured-simulated
        default=2,  # local-deconv-calscenario-longtrial-structured-simulated
    )

    parser.add_argument("--only_test", type=bool, default=True)

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}"
            for x in filename_list
            if "general_format_processed.npy" in x
        ]
    else:
        data_path_list = [params["data_path"]]

    for data_path in data_path_list:
        if params["only_test"]:
            if "test" not in data_path:
                continue

        print("data {} is being processed!".format(data_path))

        data = torch.load(data_path)

        num_trials = data["y"].shape[0]
        trial_length = data["y"].shape[2]

        if params["kernel_length"] is None:
            x_tmp = np.add.reduceat(
                data["codes"][:, 0],
                np.arange(0, data["codes"].shape[-1], data["time_bin_resolution"]),
                axis=-1,
            )
            kernel_length = trial_length - x_tmp.shape[-1] + 1
            print("kernel_length", kernel_length)
        else:
            kernel_length = params["kernel_length"]

        # create the code ----------------------------------------------------#
        x = torch.zeros(
            num_trials,
            params["kernel_num"],
            trial_length - kernel_length + 1,
        )

        trial_type = torch.zeros(num_trials)

        for trial in range(num_trials):
            trial_type[trial] = data["trial{}".format(trial)]["type"]
            for kernel_index in range(params["kernel_num"]):
                x[
                    trial,
                    kernel_index,
                    data["trial{}".format(trial)][
                        "event{}_onsets".format(kernel_index)
                    ],
                ] = 1

        y = torch.from_numpy(data["y"]).float()
        a = torch.from_numpy(data["a"]).float()

        if "raster" in data:
            raster = torch.from_numpy(data["raster"]).float()
        time_org_resolution = data["time_org_resolution"]
        time_bin_resolution = data["time_bin_resolution"]

        print(
            "There are {} trials, {} neurons, and {} time samples.".format(
                y.shape[0], y.shape[1], y.shape[2]
            )
        )

        if "raster" in data:
            print(f"original resolution is {time_org_resolution} ms")
            print(f"bin resolution is {time_bin_resolution} samples")
            print(
                "raster dim is {}, (trials, neurons, time length in original resolution)".format(
                    list(raster.shape)
                )
            )
        print("y dim is {}, (trials, neurons, time length)".format(list(y.shape)))
        print("type dim is {}, (trials)".format(list(trial_type.shape)))
        print("x dim is {} (trials, num kernels, time length)".format(list(x.shape)))
        print("a dim is {} ((trials, neurons, time length)".format(list(a.shape)))

        res_dict = dict()
        if "raster" in data:
            # raster for spiking data
            res_dict["raster"] = raster
            res_dict["time_org_resolution"] = time_org_resolution
            res_dict["time_bin_resolution"] = time_bin_resolution
        res_dict["y"] = y  # data
        res_dict["x"] = x  # code
        res_dict["a"] = a  # baseline
        res_dict["type"] = trial_type  # trial_type
        if params["only_test"]:
            res_dict["codes"] = data["codes"]
            res_dict["rate"] = data["rate"]

        # save data
        if 1:
            save_path = "{}_kernellength{}_kernelnum{}_trainready.pt".format(
                data_path.split(".npy")[0],
                kernel_length,
                params["kernel_num"],
            )
            torch.save(res_dict, save_path)
            print("processed data is save at {}!".format(save_path))


if __name__ == "__main__":
    main()
