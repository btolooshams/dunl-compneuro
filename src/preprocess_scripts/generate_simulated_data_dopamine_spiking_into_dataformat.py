"""
Copyright (c) 2020 Bahareh Tolooshams

generated simulated data similar to dopamine spiking

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import argparse
import matplotlib.pyplot as plt


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/dopamine-spiking-simulated",
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
        default=40,
    )
    parser.add_argument(
        "--time-bin-resolution",
        type=int,
        help="time bin resolution",
        default=25,  # I assume the original resolution of data is 1 ms, so this would be 25 ms.
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        help="number of trials",
        default=14,
    )
    parser.add_argument(
        "--reward-amount-list",
        type=list,
        help="reward amount list",
        default=[0.1, 0.3, 1.2, 2.5, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--code-amplitude-a-range",
        type=list,
        help="a parameters to generate code amplitudes",
        default=[-7, +5],
    )
    parser.add_argument(
        "--value-amplitude-max",
        type=float,
        help="value amplitude max",
        default=20,
    )
    parser.add_argument(
        "--cue-amplitude-mean",
        type=float,
        help="cue amp mean for normal",
        default=15,
    )
    parser.add_argument(
        "--cue-amplitude-std",
        type=float,
        help="cue amp std for normal",
        default=1,
    )
    parser.add_argument(
        "--salience-amplitude-mean",
        type=float,
        help="salience amp mean for normal",
        default=20,
    )
    parser.add_argument(
        "--salience-amplitude-std",
        type=float,
        help="salience amp std for normal",
        default=2,
    )
    parser.add_argument(
        "--baseline-mean",
        type=float,
        help="baseline mean across neurons",
        default=-5.3,
    )
    parser.add_argument(
        "--baseline-std",
        type=float,
        help="baseline std across neurons",
        default=0.3,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print(
        "this script assumes that firing-rate-resolution is the same as time-bin-resolution."
    )
    print("I assume half of trials are surprise and half are expected.")
    params = init_params()

    out_path = params["out_path"]
    silent_bg_value = 1e-3
    baseline_end_onset = int(1000 / params["time_bin_resolution"])

    # (num_kernels, kernel_length)

    kernels = np.load(os.path.join(out_path, "kernels.npy"))
    kernels = torch.unsqueeze(torch.tensor(kernels), dim=1)
    kernels_org_resolution = torch.repeat_interleave(
        kernels, params["time_bin_resolution"], dim=-1
    )
    kernels_org_resolution = F.normalize(kernels_org_resolution, p=2, dim=-1)

    kernel_num = kernels_org_resolution.shape[0]
    kernel_length_org_resolution = kernels_org_resolution.shape[-1]
    code_dur = params["trial_length"] - kernel_length_org_resolution + 1

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length after binning)
    optpes_factor = (
        torch.rand(params["num_neurons"])
        * (params["code_amplitude_a_range"][1] - params["code_amplitude_a_range"][0])
        + params["code_amplitude_a_range"][0]
    )
    cue_across_neurons = (
        params["cue_amplitude_std"] * torch.randn(params["num_neurons"])
        + params["cue_amplitude_mean"]
    )
    salinece_across_neurons = (
        params["salience_amplitude_std"] * torch.randn(params["num_neurons"])
        + params["salience_amplitude_mean"]
    )
    baseline_across_neurons = (
        params["baseline_std"] * torch.randn(params["num_neurons"])
        + params["baseline_mean"]
    )

    raster = torch.zeros(
        (params["num_trials"], params["num_neurons"], params["trial_length"])
    )
    rate = torch.zeros(
        (params["num_trials"], params["num_neurons"], params["trial_length"])
    )
    code = torch.zeros(
        (params["num_trials"], params["num_neurons"], kernel_num, code_dur)
    )
    reward_amount = torch.zeros(params["num_trials"])

    # lets first create the label reward for each trial
    num_rewards = len(params["reward_amount_list"])

    num_trials_for_each_reward = int(np.floor(params["num_trials"] / num_rewards))
    print(f"num_trials_for_each_reward: {num_trials_for_each_reward}")

    for trial_ctr in range(0, params["num_trials"], num_rewards):
        try:
            reward_amount[trial_ctr : trial_ctr + num_rewards] = torch.linspace(
                1, num_rewards, num_rewards
            )
        except:
            num_trials_left = len(reward_amount[trial_ctr : trial_ctr + num_rewards])

            reward_amount[trial_ctr : trial_ctr + num_trials_left] = torch.linspace(
                1, num_rewards, num_rewards
            )[num_trials_left]

    # cue event happens at 1000 ms, then at 2500 (0 to 3100)
    # reward event happens at 1000
    for neuron_ctr in range(params["num_neurons"]):
        value_list_curr_neuron = (
            torch.linspace(
                torch.tensor(1),
                torch.tensor(params["value_amplitude_max"]),
                num_rewards,
            )
            + optpes_factor[neuron_ctr]
        )

        for trial_ctr in range(params["num_trials"]):
            reward_amount_curr = reward_amount[trial_ctr]

            # expected trials
            if trial_ctr < (params["num_trials"] / 2):
                reward_onset = 2500
                # cue
                cue_onset = 1000
                code[trial_ctr, neuron_ctr, 0, cue_onset] = cue_across_neurons[
                    neuron_ctr
                ]
            # surprise trials
            else:
                reward_onset = 1000

            # reward
            code[trial_ctr, neuron_ctr, 1, reward_onset] = salinece_across_neurons[
                neuron_ctr
            ]
            code[trial_ctr, neuron_ctr, 2, reward_onset] = value_list_curr_neuron[
                int(reward_amount_curr - 1)
            ]

        # get Hx
        Hx_org = F.conv_transpose1d(code[:, neuron_ctr], kernels_org_resolution)
        rate_neuron = torch.sigmoid(Hx_org + baseline_across_neurons[neuron_ctr])

        # print(torch.sigmoid(baseline_across_neurons[:10]))

        raster_neuron = torch.bernoulli(rate_neuron)

        rate[:, [neuron_ctr], :] = rate_neuron
        raster[:, [neuron_ctr], :] = raster_neuron

    # do binning
    y_count = np.add.reduceat(
        raster,
        np.arange(0, raster.shape[-1], params["time_bin_resolution"]),
        axis=-1,
    )
    y = y_count / params["time_bin_resolution"]

    # use the first samples prior to event to estimate the background activity
    bg_rate = torch.mean(y[:, :, :baseline_end_onset], dim=-1)
    # replace the bg rate of those that are very small with silent_bg_value
    bg_rate[np.where(bg_rate < silent_bg_value)] = silent_bg_value
    baseline = np.expand_dims(np.log(bg_rate / (1 - bg_rate)), axis=(-1))

    x = np.add.reduceat(
        code[:, 0],
        np.arange(0, code.shape[-1], params["time_bin_resolution"]),
        axis=-1,
    )

    x[torch.abs(x) > 0] = 1

    y = y.detach().clone().cpu().numpy()
    x = x.detach().clone().cpu().numpy()
    rate = rate.detach().clone().cpu().numpy()
    raster = raster.detach().clone().cpu().numpy()
    kernels = kernels.detach().clone().cpu().numpy()

    x0 = x[:, 0]
    x1 = x[:, 1]

    ############# now save the data

    num_trials = y.shape[0]

    ###### save data
    data_dict = dict()
    data_dict["kernel_num"] = kernel_num
    data_dict["raster"] = raster
    data_dict["rate"] = rate
    data_dict["kernels"] = kernels
    data_dict["codes"] = code
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
        save_path = "{}/simulated_{}nuerons_{}trials_{}msbinres_general_format_processed.npy".format(
            params["out_path"],
            params["num_neurons"],
            params["num_trials"],
            params["time_bin_resolution"],
        )

        torch.save(data_dict, save_path)

        print(
            f"general format processed train data for dopamine eshel is saved at {save_path}!"
        )

    print("this script created npy dictionaries as below.")
    print(
        "key raster: raw data with dim (num_trials, num_neurons, trial_length in org resolution)."
    )
    print(
        "key rate: rate function with dim (num_trials, num_neurons, trial_length in org resolution)."
    )
    print("key kernels: kernel with dim (num_kernels, 1, kernel_length).")
    print(
        "key codes: codes with dim (num_trials, num_neurons, num_kernels, code length)."
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


if __name__ == "__main__":
    main()
