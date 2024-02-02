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
import matplotlib as mpl
import matplotlib.pyplot as plt


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/local-deconv-calscenario-shorttrial-structured-simulated",
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        help="trial length",
        default=2000,
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        help="number of neurons",
        default=1,
    )
    parser.add_argument(
        "--train-or-test",
        type=str,
        help="train or test",
        default="train",
        # default="test",
    )
    parser.add_argument(
        "--time-bin-resolution",
        type=int,
        help="time bin resolution",  # see main
        default=25,
    )
    parser.add_argument(
        "--num-trials-list",
        type=list,
        help="number of trials list",  # see main
        # default=[],
        default=[25, 50, 100, 200, 400, 800, 1600],
        # default=[500],
    )
    parser.add_argument(
        "--baseline-mean",
        type=float,
        help="baseline mean across neurons",  # see main
        # -6.2126,  # 2 Hz
        # -5.2933,  # 5 Hz
        default=-4.8203,  # 8 Hz
        # default = -4.4988,  # 11 Hz
        # default = -4.2546,  # 14 Hz
        # -4.0574,  # 17 Hz
    )
    ########################################################################################
    parser.add_argument(
        "--baseline-std",
        type=float,
        help="baseline std across neurons",
        default=0.01,  # I set this to very small value (so that the firing rate does not change much from the reported mean baseline)
    )
    parser.add_argument(
        "--rate-bin-resolution",
        type=int,
        help="rate bin resolution",
        default=5,  # 5 ms. This is the resolution at which we assume the rate is constant.
    )
    parser.add_argument(
        "--code-amplitude-mean",
        type=float,
        help="code amp mean for normal",
        # default=[45, 50],  # Let see what the SNR is given the baseline activity.
        default=[50, 60],  # Let see what the SNR is given the baseline activity.
    )
    parser.add_argument(
        "--code-amplitude-std",
        type=float,
        help="code amp std for normal",
        default=10,
    )
    parser.add_argument(
        "--baseline-end-onset-ms",
        type=float,
        help=" the time of first event. This is to make sure there is enough time to estimate the background activity",
        default=500,  # The first 500 ms of data no events happens
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        help="seed",
        default=119487,
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        help="seed",
        default=5472923,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main(params):
    torch.manual_seed(params["{}_seed".format(params["train_or_test"])])
    np.random.seed(params["{}_seed".format(params["train_or_test"])])

    out_path = params["out_path"]
    silent_bg_value = 1e-3
    baseline_end_onset = int(
        params["baseline_end_onset_ms"] / params["time_bin_resolution"]
    )

    kernels = np.load(os.path.join(out_path, "kernels.npy"))


    kernels = torch.unsqueeze(torch.tensor(kernels), dim=1)
    kernels_org_resolution = torch.repeat_interleave(
        kernels, params["time_bin_resolution"], dim=-1
    )
    kernels_org_resolution = F.normalize(kernels_org_resolution, p=2, dim=-1)
    kernel_num = kernels_org_resolution.shape[0]
    kernel_length_org_resolution = kernels_org_resolution.shape[-1]
    code_dur = params["trial_length"] - kernel_length_org_resolution + 1

    # convert baseline to Hz in 1 ms ----------------------------------------------------#
    baseline_in_Hz = dict()
    baseline_in_Hz["-6.2126"] = 2
    baseline_in_Hz["-5.2933"] = 5
    baseline_in_Hz["-4.8203"] = 8
    baseline_in_Hz["-4.4988"] = 11
    baseline_in_Hz["-4.2546"] = 14
    baseline_in_Hz["-4.0574"] = 17
    baseline_in_Hz["-3.8918"] = 20
    curr_baseline_in_Hz = baseline_in_Hz["{}".format(params["baseline_mean"])]

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length after binning)
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

    # events happens uniformaly at random (but we override them to have min distance between code of code_min_distance)
    code = torch.zeros(
        (params["num_trials"], params["num_neurons"], kernel_num, code_dur)
    )
    index_list = [[700], [1300]]

    for neuron_ctr in range(params["num_neurons"]):
        for trial_ctr in range(params["num_trials"]):
            for kernel_ctr in range(kernel_num):
                index = index_list[kernel_ctr]
                # baseline_end_onset_ms is used to make sure that no event is in the first baseline_end_onset_ms of the trial
                amp = (
                    params["code_amplitude_std"] * torch.randn(1)
                    + params["code_amplitude_mean"][kernel_ctr]
                )
                code[
                    trial_ctr,
                    neuron_ctr,
                    kernel_ctr,
                    index,
                ] = amp

        # get Hx
        Hx_org = F.conv_transpose1d(code[:, neuron_ctr], kernels_org_resolution)
        rate_neuron = torch.sigmoid(Hx_org + baseline_across_neurons[neuron_ctr])

        raster_neuron = torch.bernoulli(rate_neuron)

        rate[:, [neuron_ctr], :] = rate_neuron
        raster[:, [neuron_ctr], :] = raster_neuron

    print("raster", raster.shape)

    spikes = torch.sum(raster, dim=(-1, -2))

    print(
        "curr_baseline_in_Hz",
        curr_baseline_in_Hz,
        torch.mean(spikes),
        torch.std(spikes),
    )

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
    kernels_org_resolution = kernels_org_resolution.detach().clone().cpu().numpy()

    ############# now save the data

    num_trials = y.shape[0]

    ###### save data
    data_dict = dict()
    data_dict["kernel_num"] = kernel_num
    data_dict["raster"] = raster
    data_dict["rate"] = rate
    data_dict[
        "kernels"
    ] = kernels_org_resolution  # be careful (this is the kernel in 1 ms original resolution)
    data_dict["codes"] = code
    data_dict["y"] = y
    data_dict["a"] = baseline
    data_dict["time_bin_resolution"] = params[
        "time_bin_resolution"
    ]  # samples relative to the original resolution
    data_dict["time_org_resolution"] = 1  # ms
    data_dict["rate_bin_resolution"] = params["rate_bin_resolution"]  # ms
    data_dict["baseline_mean"] = params["baseline_mean"]
    data_dict["baseline_std"] = params["baseline_std"]

    data_dict["baseline_in_Hz"] = curr_baseline_in_Hz
    data_dict["code_amplitude_mean"] = params["code_amplitude_mean"]
    data_dict["code_amplitude_std"] = params["code_amplitude_std"]

    for trial in range(num_trials):
        data_dict["trial{}".format(trial)] = dict()
        data_dict["trial{}".format(trial)]["type"] = 0  # There is no event type. So 0

        for kernel_index in range(data_dict["kernel_num"]):
            data_dict["trial{}".format(trial)][
                "event{}_onsets".format(kernel_index)
            ] = list()

        # event 0 is for cue events
        for kernel_ctr in range(data_dict["kernel_num"]):
            event_curr_onset = np.where(x[:, kernel_ctr][trial] > 0)[-1]
            if event_curr_onset.size > 0:
                data_dict["trial{}".format(trial)][
                    "event{}_onsets".format(kernel_ctr)
                ].append(event_curr_onset)

    # save data
    if 1:
        if params["train_or_test"] == "train":
            save_path = "{}/simulated_{}neurons_{}trials_{}msbinres_{}Hzbaseline_nov_general_format_processed.npy".format(
                params["out_path"],
                params["num_neurons"],
                params["num_trials"],
                params["time_bin_resolution"],
                curr_baseline_in_Hz,
            )
        else:
            save_path = "{}/test_simulated_{}neurons_{}trials_{}msbinres_{}Hzbaseline_nov_general_format_processed.npy".format(
                params["out_path"],
                params["num_neurons"],
                params["num_trials"],
                params["time_bin_resolution"],
                curr_baseline_in_Hz,
            )

        torch.save(data_dict, save_path)

        print(
            f"general format processed train data for dopamine eshel is saved at {save_path}!"
        )

    if 0:
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
    params = init_params()

    for num_trials in params["num_trials_list"]:
        print("num_trials", num_trials)
        params["num_trials"] = num_trials

        main(params)

    print("done.")
