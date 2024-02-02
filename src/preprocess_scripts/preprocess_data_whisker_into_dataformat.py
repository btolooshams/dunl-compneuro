"""
Copyright (c) 2020 Bahareh Tolooshams

preprocess whisker data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy.io as sio
import argparse


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="../data/whisker",
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=1,
    )
    parser.add_argument(
        "--neuron-dict-list",
        type=dict,
        help="neurons to process",
        default={
            1: [2],
            2: [1],
            4: [1],
            5: [1],
            6: [1],
            8: [2],
            10: [2],
            16: [2],
            17: [1, 2],
        },
    )
    parser.add_argument(
        "--stim-num",
        type=int,
        help="number of stimulus",
        default=3,
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        help="trial length",
        default=3000,
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        help="number of neurons",
        default=11,  # this is related to the number of neurons in neuron_dict_list
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        help="number of trials train + test",
        default=50,  # this is 50 calculated from data.
    )
    parser.add_argument(
        "--num-train",
        type=int,
        help="number of train",
        default=30,  # using first 30 for train
    )

    parser.add_argument(
        "--time-bin-resolution",
        type=int,
        help="time bin resolution",
        default=5,  # this is in terms of samples. The original data resolution is 1 ms, hence, 10 sample is 10 ms resolution.
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def main():
    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))
    print("the original resolution of data is 1 ms.")

    baseline_end_onset_org_res = 500
    baseline_end_onset = int(
        baseline_end_onset_org_res / params["time_bin_resolution"]
    )  # 500 ms / time_bin_resolution
    silent_bg_value = 1e-3

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length after binning)

    raster = np.zeros(
        (params["num_trials"], params["num_neurons"], params["trial_length"])
    )
    stim = np.zeros(
        (params["num_trials"], params["num_neurons"], params["trial_length"])
    )
    y = np.zeros(
        (
            params["num_trials"],
            params["num_neurons"],
            int(params["trial_length"] / params["time_bin_resolution"]),
        )
    )
    # (num trials, num neurons, 1)
    baseline = np.zeros((params["num_trials"], params["num_neurons"], 1))

    neuron_ctr = 0
    for dir_num, neuron_num_list in params["neuron_dict_list"].items():
        for neuron_num in neuron_num_list:
            file_name = "{}_{}_{}".format(dir_num, neuron_num, params["stim_num"])
            filepath = "{}/data/Dir{}/Neuron{}".format(
                params["data_path"], dir_num, neuron_num
            )

            trng_data = sio.loadmat(
                "{}/Stim{}/trngdata".format(filepath, params["stim_num"])
            )
            test_data = sio.loadmat(
                "{}/Stim{}/testdata".format(filepath, params["stim_num"])
            )

            # combine all train/test
            t = np.hstack([trng_data["t"], test_data["t"]])
            raw_data = np.hstack([trng_data["y"], test_data["y"]])

            raw_data_curr = np.reshape(
                raw_data, (params["num_trials"], params["trial_length"])
            )
            stim_curr = np.reshape(t, (params["num_trials"], params["trial_length"]))

            # do time binning on the raster
            y_count_curr = np.add.reduceat(
                raw_data_curr,
                np.arange(0, raw_data_curr.shape[-1], params["time_bin_resolution"]),
                axis=-1,
            )

            raster[:, neuron_ctr, :] = raw_data_curr
            stim[:, neuron_ctr, :] = stim_curr
            y[:, neuron_ctr, :] = y_count_curr / params["time_bin_resolution"]

            # use the first 500 samples to estimate the background activity
            bg_rate = np.mean(y[:, neuron_ctr, :baseline_end_onset], axis=-1)
            # replace the bg rate of those that are very small with silent_bg_value
            bg_rate[np.where(bg_rate < silent_bg_value)] = silent_bg_value
            baseline[:, neuron_ctr, 0] = np.log(bg_rate / (1 - bg_rate))

            neuron_ctr += 1

    num_train = params["num_train"]
    num_test = params["num_trials"] - num_train

    data_dict_train = dict()
    data_dict_train["kernel_num"] = params["kernel_num"]
    data_dict_train["stim"] = stim[:num_train]
    data_dict_train["raster"] = raster[:num_train]
    data_dict_train["y"] = y[:num_train]
    data_dict_train["a"] = baseline[:num_train]
    data_dict_train["time_bin_resolution"] = params[
        "time_bin_resolution"
    ]  # samples relative to the original resolution
    data_dict_train["time_org_resolution"] = 1  # ms

    data_dict_test = dict()
    data_dict_test["kernel_num"] = params["kernel_num"]
    data_dict_test["stim"] = stim[num_train:]
    data_dict_test["raster"] = raster[num_train:]
    data_dict_test["y"] = y[num_train:]
    data_dict_test["a"] = baseline[num_train:]
    data_dict_test["time_bin_resolution"] = params[
        "time_bin_resolution"
    ]  # samples relative to the original resolution
    data_dict_test["time_org_resolution"] = 1  # ms

    print(
        "This script assumes binomial distribution on the data when computing the baseline a."
    )
    print(
        "For this dataset, we know the event happens from 500 ms to 2500 ms every 125 ms (total of 16 times)."
    )

    event_onsets = np.floor(
        (np.arange(16) * 125 + baseline_end_onset_org_res)
        / params["time_bin_resolution"]
    )
    event_onsets = [int(x) for x in event_onsets]

    for trial in range(num_train):
        data_dict_train["trial{}".format(trial)] = dict()
        data_dict_train["trial{}".format(trial)][
            "type"
        ] = 0  # there is only one event type

        for kernel_index in range(data_dict_train["kernel_num"]):
            data_dict_train["trial{}".format(trial)][
                "event{}_onsets".format(kernel_index)
            ] = list()

        data_dict_train["trial{}".format(trial)]["event0_onsets"] = event_onsets

    for trial in range(num_test):
        data_dict_test["trial{}".format(trial)] = dict()
        data_dict_test["trial{}".format(trial)][
            "type"
        ] = 0  # there is only one event type

        for kernel_index in range(data_dict_test["kernel_num"]):
            data_dict_test["trial{}".format(trial)][
                "event{}_onsets".format(kernel_index)
            ] = list()

        data_dict_test["trial{}".format(trial)]["event0_onsets"] = event_onsets

    print("this script created a npy dictionary as below.")
    print(
        "key raster: raw data with dim (num_trials, num_neurons, trial_length in org resolution)."
    )
    print(
        "key y: binned data with dim (num_trials, num_neurons, trial_length after time bin)."
    )
    print("key a: baseline activity with dim (num_trials, num_neurons, 1).")
    print("key kernel_num: number of kernels.")
    print(
        "key trial#: this is a dict with keys event#_onsets containing list of indices that event# has happend in that trial."
    )
    print("key trial#: this also has a key of type for trial type.")

    # save data
    if 1:
        train_save_path = (
            "{}/whisker_train_{}msbinres_general_format_processed.npy".format(
                params["data_path"], params["time_bin_resolution"]
            )
        )
        test_save_path = (
            "{}/whisker_test_{}msbinres_general_format_processed.npy".format(
                params["data_path"], params["time_bin_resolution"]
            )
        )

        torch.save(data_dict_train, train_save_path)
        torch.save(data_dict_test, test_save_path)

        print(
            f"general format processed train data for sara matias is saved at {train_save_path}!"
        )
        print(
            f"general format processed test data for sara matias is saved at {test_save_path}!"
        )


if __name__ == "__main__":
    main()
