"""
Copyright (c) 2020 Bahareh Tolooshams

preprocess data

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
        default="../data/dopamine-calcium-saramatias-uchida/VarMag_SM103_20191104.mat",  # 20 neurons, 299 trials
        # default="../data/dopamine-calcium-saramatias-uchida/VarMag_SM99_20191109.mat",  # 30 neurons, 195 trials
        # default="../data/dopamine-calcium-saramatias-uchida/VarMag_SM104_20191103.mat", # 6 neurons, 252 trials
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=5,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    data = sio.loadmat(params["data_path"])

    # cell, states, session from mat file
    cell = data["DFoF_corrraw_fissa"][0]
    states = data["States"][0]
    session = data["Session"]
    cells_time_stamps = np.expand_dims(np.squeeze(data["CellsTimeStamps"]), axis=0)[0]

    # get reward amount and trial types
    reward_amount = np.expand_dims(np.squeeze(session["RewardDelivered"]), axis=0)[0][
        :, 0
    ]
    reward_amount = torch.from_numpy(np.squeeze(reward_amount)).float()

    trial_types = np.int32(
        np.expand_dims(np.squeeze(session["TrialTypes"]), axis=0)[0][0]
    )
    trial_types[trial_types == 4] = 1000
    trial_types[trial_types == 7] = 4
    trial_types[trial_types == 1000] = 7
    trial_types = torch.from_numpy(np.squeeze(trial_types)).float()

    # 1 (cue, no reward)
    # 4 is surprise (no cue, with reward)
    # 7 (cue, with reward)

    time_res = np.squeeze(cells_time_stamps[0][1] - cells_time_stamps[0][0])
    num_trials = len(cell)
    num_neurons = cell[0].shape[-1]

    # trial information --------------------------------------------------#
    reward_time = []
    cue_time_regret = []
    cue_time_expected = []

    trial_org_num_samples = []
    for trial in range(num_trials):
        reward_time.append(states[trial]["Reward"][0][0])
        cue_time_regret.append(states[trial]["Stimulus1Delivery"][0][0])
        cue_time_expected.append(states[trial]["Stimulus7Delivery"][0][0])
        trial_org_num_samples.append(len(cells_time_stamps[trial]))

    # trial time samples -------------------------------------------------#
    # print(np.min(trial_org_num_samples), np.max(trial_org_num_samples))
    trial_length = np.max(trial_org_num_samples)

    trial_time_samples = []
    for trial in range(num_trials):
        trial_length_old = len(cells_time_stamps[trial])
        start_index = cells_time_stamps[trial][-1] + time_res
        end_index = (trial_length - trial_length_old) * time_res + cells_time_stamps[
            trial
        ][-1]

        if trial_length_old == trial_length:
            trial_time_samples_i = np.array(np.squeeze(cells_time_stamps[trial]))
        elif trial_length - trial_length_old == 1:
            zeropad = np.zeros(1) + start_index
            trial_time_samples_i = np.concatenate(
                [np.array(np.squeeze(cells_time_stamps[trial])), zeropad],
            )
        else:
            zeropad = np.linspace(
                start_index, end_index, int(trial_length - trial_length_old)
            )
            zeropad = np.array(np.squeeze(zeropad))
            trial_time_samples_i = np.concatenate(
                [np.array(np.squeeze(cells_time_stamps[trial])), zeropad],
            )
        trial_time_samples.append(trial_time_samples_i)

    reward_time = np.squeeze(np.array(reward_time))
    cue_time_regret = np.squeeze(np.array(cue_time_regret))
    cue_time_expected = np.squeeze(np.array(cue_time_expected))
    # (num trials, trial time samples)
    trial_time_samples = torch.from_numpy(
        np.squeeze(np.stack(trial_time_samples))
    ).float()

    # match time to onset ------------------------------------------------#
    reward_onset = []
    cue_onset_regret = []
    cue_onset_expected = []
    for trial in range(num_trials):
        # reward
        if np.isnan(reward_time[trial]):
            reward_onset.append(reward_time[trial])
        else:
            reward_onset.append(
                np.where(trial_time_samples[trial] <= reward_time[trial])[0][-1]
            )

        # cue regret
        if np.isnan(cue_time_regret[trial]):
            cue_onset_regret.append(cue_time_regret[trial])
        else:
            cue_onset_regret.append(
                np.where(trial_time_samples[trial] <= cue_time_regret[trial])[0][-1]
            )

        # cue expected
        if np.isnan(cue_time_expected[trial]):
            cue_onset_expected.append(cue_time_expected[trial])
        else:
            cue_onset_expected.append(
                np.where(trial_time_samples[trial] <= cue_time_expected[trial])[0][-1]
            )

    reward_onset = np.squeeze(np.array(reward_onset))
    cue_onset_regret = np.squeeze(np.array(cue_onset_regret))
    cue_onset_expected = np.squeeze(np.array(cue_onset_expected))

    baseline_end_onset = []
    for trial in range(num_trials):
        if not np.isnan(reward_onset[trial]):
            reward_onset[trial] = int(reward_onset[trial])

        if not np.isnan(cue_onset_regret[trial]):
            cue_onset_regret[trial] = int(cue_onset_regret[trial])

        if not np.isnan(cue_onset_expected[trial]):
            cue_onset_expected[trial] = int(cue_onset_expected[trial])

        if trial_types[trial] == 1:
            baseline_end_onset.append(cue_onset_regret[trial] - 1)
        elif trial_types[trial] == 4:
            baseline_end_onset.append(reward_onset[trial] - 1)
        elif trial_types[trial] == 7:
            baseline_end_onset.append(cue_onset_expected[trial] - 1)

    baseline_end_onset = torch.from_numpy(np.squeeze(np.array(baseline_end_onset)))

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length)
    y = np.zeros((num_trials, num_neurons, trial_length))
    # (num trials, num neurons, 1)
    baseline = np.zeros((num_trials, num_neurons, 1))

    for neuron in range(num_neurons):
        for trial in range(num_trials):
            trial_length_old = trial_org_num_samples[trial]

            if trial_length_old == trial_length:
                y[trial, neuron] = cell[trial][:, neuron]

            else:
                zeropad = np.zeros(trial_length - trial_length_old)
                y[trial, neuron] = np.concatenate(
                    [cell[trial][:, neuron], zeropad],
                )

            baseline[trial, neuron, 0] = np.mean(
                cell[trial][: int(baseline_end_onset[trial]), neuron]
            )

    data_dict = dict()
    data_dict["kernel_num"] = params["kernel_num"]
    data_dict["y"] = y
    data_dict["a"] = baseline

    for trial in range(num_trials):
        data_dict["trial{}".format(trial)] = dict()
        data_dict["trial{}".format(trial)]["type"] = np.round(
            reward_amount[trial].item(), 2
        )

        for kernel_index in range(data_dict["kernel_num"]):
            data_dict["trial{}".format(trial)][
                "event{}_onsets".format(kernel_index)
            ] = list()

        # event 0 is for cue regret events
        if not np.isnan(states[trial]["Stimulus1Delivery"][0][0]):
            data_dict["trial{}".format(trial)]["event0_onsets"].append(
                int(cue_onset_regret[trial])
            )

        # event 1 is for cue expected events
        if not np.isnan(states[trial]["Stimulus7Delivery"][0][0]):
            data_dict["trial{}".format(trial)]["event1_onsets"].append(
                int(cue_onset_expected[trial])
            )
        # reward events
        if not np.isnan(states[trial]["Reward"][0][0]):
            data_dict["trial{}".format(trial)]["event2_onsets"].append(
                int(reward_onset[trial])
            )
            data_dict["trial{}".format(trial)]["event3_onsets"].append(
                int(reward_onset[trial])
            )
            if data_dict["kernel_num"] == 5:
                data_dict["trial{}".format(trial)]["event4_onsets"].append(
                    int(reward_onset[trial])
                )

    print("this script created a npy dictionary as below.")
    print("key y: raw data with dim (num_trials, num_neurons, trial_length).")
    print("key a: baseline activity with dim (num_trials, num_neurons, 1).")
    print("key kernel_num: number of kernels.")
    print(
        "key trial#: this is a dict with keys event#_onsets containing list of indices that event# has happend in that trial."
    )
    print("key trial#: this also has a key of type for trial type.")

    # save data
    if 1:
        save_path = "{}_general_format_processed.npy".format(
            params["data_path"].split(".mat")[0]
        )
        torch.save(data_dict, save_path)
        print(
            "general format processed data for sara matias is saved at {}!".format(
                save_path
            )
        )


if __name__ == "__main__":
    main()
