"""
Copyright (c) 2020 Bahareh Tolooshams

preprocess data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy.io as sio
import os
import argparse


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="../data/dopamine-spiking-eshel-uchida",
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=3,
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        help="trial length",
        default=3100,
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        help="number of neurons",
        default=1,  # each file has one neuron
    )
    parser.add_argument(
        "--time-bin-resolution",
        type=int,
        help="time bin resolution",
        default=25,  # this is in terms of samples. The original data resolution is 1 ms, hence, 25 sample is 25 ms resolution.
    )
    parser.add_argument(
        "--train-percentage",
        type=int,
        help="train percentage",
        default=0.8,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def get_spikes(data):
    # spike timing in ms
    timing = data["responses"]["spike"][0][0]
    spike_timing = np.round(timing)
    spike_timing = spike_timing.astype(int)
    # length
    trial_length = int(np.ceil(timing[-1])[0] + 1)
    # spike vector
    spikes = np.zeros(trial_length)
    spikes[spike_timing] = 1

    return spikes, spike_timing


def get_onset(data):
    # odor timing in ms
    timing = data["events"]["odorOn"][0][0]
    onset_timing = np.round(timing)
    onset_timing = onset_timing.astype(int)
    # length
    trial_length = int(np.ceil(timing[-1])[0] + 1)
    # odor vector
    onset = np.zeros(trial_length)
    onset[onset_timing] = 1

    return onset, onset_timing


def get_trial_type(data):
    # trial type
    return data["events"]["trialType"][0][0]


def main():
    params = init_params()

    data_path = params["data_path"]
    delay_bwd = 1000
    delay_fwd = 2100
    reward_delay = 1500
    num_trial_types = 14
    silent_bg_value = 1e-3
    baseline_end_onset = int(delay_bwd / params["time_bin_resolution"])

    print("data {} is being processed!".format(params["data_path"]))

    filename_list = os.listdir(data_path)
    data_path_list = [f"{data_path}/{x}" for x in filename_list if ".mat" in x]

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length after binning)

    for data_path_curr in data_path_list:
        raster = list()  # (num trials, num neurons, trial length)
        y = list()  # (num trials, num neurons, trial length after binning)
        baseline = list()  # (num trials, num neurons, 1)
        example_type = list()  # (num trials, num neurons, 1)
        x0 = list()
        x1 = list()

        data = sio.loadmat(data_path_curr, squeeze_me=False)

        # get trial types, spikes, and onsets
        trial_type = get_trial_type(data)
        spike_vector, spike_timing = get_spikes(data)
        onset_vector, onset_timing = get_onset(data)

        # create a dictionary for types
        spike_timing_dict = dict()
        reward_timing_dict = dict()
        odor_timing_dict = dict()
        onset_timing_dict = dict()
        for type_num in range(1, num_trial_types + 1):
            spike_timing_dict["T{}".format(type_num)] = []
            reward_timing_dict["T{}".format(type_num)] = []
            odor_timing_dict["T{}".format(type_num)] = []
            onset_timing_dict["T{}".format(type_num)] = []

        for ctr in range(len(onset_timing)):
            onset = onset_timing[ctr].copy()
            t_type = trial_type[ctr].copy()
            spike_t = spike_timing[np.where(spike_timing >= onset - delay_bwd)].copy()
            spike_t = spike_t[np.where(spike_t <= onset + delay_fwd)]

            if t_type == num_trial_types + 1:
                continue

            if len(spike_t) < 1 or np.isnan(t_type):
                continue

            t_type = int(t_type[0])

            spike_timing_dict["T{}".format(t_type)].append(spike_t)
            onset_timing_dict["T{}".format(t_type)].append(onset)
            if t_type < 8:
                reward_timing_dict["T{}".format(t_type)].append(onset)
            else:
                odor_timing_dict["T{}".format(t_type)].append(onset)
                reward_timing_dict["T{}".format(t_type)].append(onset + reward_delay)

        # create a dictionary for spikes
        spike_train_dict = dict()
        onset_train_dict = dict()
        odor_train_dict = dict()
        reward_train_dict = dict()
        for t_type in range(1, num_trial_types + 1):
            num_examples = len(spike_timing_dict["T{}".format(t_type)])
            spike_train_dict["T{}".format(t_type)] = np.zeros(
                (num_examples, delay_bwd + delay_fwd)
            )
            onset_train_dict["T{}".format(t_type)] = np.zeros(
                (num_examples, delay_bwd + delay_fwd)
            )
            odor_train_dict["T{}".format(t_type)] = np.zeros(
                (num_examples, delay_bwd + delay_fwd)
            )
            reward_train_dict["T{}".format(t_type)] = np.zeros(
                (num_examples, delay_bwd + delay_fwd)
            )

            for win_num in range(len(spike_timing_dict["T{}".format(t_type)])):
                onset_time_abs = onset_timing_dict["T{}".format(t_type)][win_num]

                spike_time = (
                    spike_timing_dict["T{}".format(t_type)][win_num]
                    - onset_time_abs
                    + delay_bwd
                    - 1
                )
                spike_train_dict["T{}".format(t_type)][win_num, spike_time] = 1

                onset_time = (
                    onset_timing_dict["T{}".format(t_type)][win_num]
                    - onset_time_abs
                    + delay_bwd
                    - 1
                )
                onset_train_dict["T{}".format(t_type)][win_num, onset_time] = 1

                reward_time = (
                    reward_timing_dict["T{}".format(t_type)][win_num]
                    - onset_time_abs
                    + delay_bwd
                    - 1
                )
                reward_train_dict["T{}".format(t_type)][win_num, reward_time] = 1

                if t_type >= 8:
                    odor_time = (
                        odor_timing_dict["T{}".format(t_type)][win_num]
                        - onset_time_abs
                        + delay_bwd
                        - 1
                    )
                    odor_train_dict["T{}".format(t_type)][win_num, odor_time] = 1

        for t_type in range(1, num_trial_types + 1):
            raw_data_curr = spike_train_dict["T{}".format(t_type)]
            x_odor_curr = odor_train_dict["T{}".format(t_type)]
            x_reward_curr = reward_train_dict["T{}".format(t_type)]

            # do binning
            y_count_curr = np.add.reduceat(
                raw_data_curr,
                np.arange(0, raw_data_curr.shape[-1], params["time_bin_resolution"]),
                axis=-1,
            )
            y_curr = y_count_curr / params["time_bin_resolution"]

            raster.append(np.expand_dims(raw_data_curr, axis=1))
            y.append(np.expand_dims(y_curr, axis=1))

            example_type.append(np.zeros(y_curr.shape[0]) + t_type)

            # use the first samples prior to event to estimate the background activity
            bg_rate = np.mean(y_curr[:, :baseline_end_onset], axis=-1)
            # replace the bg rate of those that are very small with silent_bg_value
            bg_rate[np.where(bg_rate < silent_bg_value)] = silent_bg_value
            baseline.append(
                np.expand_dims(np.log(bg_rate / (1 - bg_rate)), axis=(1, 2))
            )

            x_odor = np.add.reduceat(
                x_odor_curr,
                np.arange(0, x_odor_curr.shape[-1], params["time_bin_resolution"]),
                axis=-1,
            )
            x_reward = np.add.reduceat(
                x_reward_curr,
                np.arange(0, x_reward_curr.shape[-1], params["time_bin_resolution"]),
                axis=-1,
            )

            # make sure that odor and reward are still 1 indicator
            x_odor[x_odor > 1] = 1
            x_reward[x_reward > 1] = 1

            x0.append(x_odor)
            x1.append(x_reward)

        y = np.concatenate(y, axis=0)
        raster = np.concatenate(raster, axis=0)
        baseline = np.concatenate(baseline, axis=0)
        example_type = np.concatenate(example_type, axis=0)

        # for odor cue
        x0 = np.concatenate(x0, axis=0)
        # for reward
        x1 = np.concatenate(x1, axis=0)

        reward_amount = np.zeros(example_type.shape)
        for i in range(len(example_type)):
            example_type_curr = example_type[i]
            if example_type_curr < 8:
                reward_amount[i] = float(example_type_curr)
            else:
                reward_amount[i] = float(example_type_curr) - 7.0

        ##########
        num_data = y.shape[0]
        num_train = int(params["train_percentage"] * num_data)

        y_train = y[:num_train]
        raster_train = raster[:num_train]
        baseline_train = baseline[:num_train]
        x0_train = x0[:num_train]
        x1_train = x1[:num_train]
        reward_amount_train = reward_amount[:num_train]

        y_test = y[num_train:]
        raster_test = raster[num_train:]
        baseline_test = baseline[num_train:]
        x0_test = x0[num_train:]
        x1_test = x1[num_train:]
        reward_amount_test = reward_amount[num_train:]

        write_data(
            y_train,
            raster_train,
            baseline_train,
            x0_train,
            x1_train,
            reward_amount_train,
            params,
            data_path_curr,
            data_type="train",
        )
        write_data(
            y_test,
            raster_test,
            baseline_test,
            x0_test,
            x1_test,
            reward_amount_test,
            params,
            data_path_curr,
            data_type="test",
        )

    print("this script created npy dictionaries as below.")
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


def write_data(
    y, raster, baseline, x0, x1, reward_amount, params, data_path_curr, data_type=None
):
    num_trials = y.shape[0]

    ###### save data
    data_dict = dict()
    data_dict["kernel_num"] = params["kernel_num"]
    data_dict["raster"] = raster
    data_dict["y"] = y
    data_dict["a"] = baseline
    data_dict["time_bin_resolution"] = params[
        "time_bin_resolution"
    ]  # samples relative to the original resolution
    data_dict["time_org_resolution"] = 1  # ms

    for trial in range(num_trials):
        data_dict["trial{}".format(trial)] = dict()
        data_dict["trial{}".format(trial)]["type"] = np.round(
            reward_amount[trial], 2
        )  # trial type has the reward amount

        for kernel_index in range(data_dict["kernel_num"]):
            data_dict["trial{}".format(trial)][
                "event{}_onsets".format(kernel_index)
            ] = list()

        # event 0 is for cue events
        event0_curr_onset = np.where(x0[trial] > 0)[-1]
        if event0_curr_onset.size > 0:
            data_dict["trial{}".format(trial)]["event0_onsets"].append(
                int(event0_curr_onset[0])
            )

        # event 1 onward are for reward events
        event1_curr_onset = np.where(x1[trial] > 0)[-1]
        if event1_curr_onset.size > 0:
            data_dict["trial{}".format(trial)]["event1_onsets"].append(
                int(event1_curr_onset[0])
            )
            if data_dict["kernel_num"] > 2:
                data_dict["trial{}".format(trial)]["event2_onsets"].append(
                    int(event1_curr_onset[0])
                )
            if data_dict["kernel_num"] > 3:
                data_dict["trial{}".format(trial)]["event3_onsets"].append(
                    int(event1_curr_onset[0])
                )

    # save data
    if 1:
        filename = data_path_curr.split(".mat")[0].split("/")[-1]
        save_path = "{}/{}/{}_{}msbinres_general_format_processed.npy".format(
            params["data_path"], data_type, filename, params["time_bin_resolution"]
        )

        torch.save(data_dict, save_path)

        print(
            f"general format processed train data for dopamine eshel is saved at {save_path}!"
        )


if __name__ == "__main__":
    main()
