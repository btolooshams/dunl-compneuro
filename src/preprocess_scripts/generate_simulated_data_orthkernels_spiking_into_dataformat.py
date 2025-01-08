"""
Copyright (c) 2025 Bahareh Tolooshams

generated simulated data similar to dopamine spiking

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--out-path",
        type=str,
        help="out path",
        default="../data/local-orthkernels-simulated",
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        help="trial length",
        default=6000,
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        help="number of neurons",
        default=1000,
    )
    parser.add_argument(
        "--train-or-test",
        type=str,
        help="train or test",
        # default="train",
        # default="test",
        default="vis",
    )
    parser.add_argument(
        "--time-bin-resolution",
        type=int,
        help="time bin resolution",  # see main
        default=5,
    )
    parser.add_argument(
        "--num-trials-list",
        type=list,
        help="number of trials list",  # see main
        # default=[],
        # default=[25, 50, 100, 250, 500, 750, 1000],
        default=[10],
        # default=[50],
    )
    parser.add_argument(
        "--baseline-mean",
        type=float,
        help="baseline mean across neurons",  # see main
        # -6.2126,  # 2 Hz
        # -5.2933,  # 5 Hz
        # default=-4.8203,  # 8 Hz
        # default = -4.4988,  # 11 Hz
        # default = -4.2546,  # 14 Hz
        # default= -4.0574,  # 17 Hz
        default=0,  # 500 Hz, this is for random spiking when there is no event (kernel)
    )
    ########################################################################################
    parser.add_argument(
        "--baseline-std",
        type=float,
        help="baseline std across neurons",
        default=0.3,
    )
    parser.add_argument(
        "--rate-bin-resolution",
        type=int,
        help="rate bin resolution",
        default=5,  # 5 ms. This is the resolution at which we assume the rate is constant.
    )
    parser.add_argument(
        "--code-supp-list",
        type=list,
        help="number of events in each trial",
        default=[1],
    )
    parser.add_argument(
        "--code-amplitude-mean",
        type=float,
        help="code amp mean for normal",
        default=[
            25,
            25,
            25,
            25,
            25,
        ],  # Let see what the SNR is given the baseline activity.
    )
    parser.add_argument(
        "--code-amplitude-std",
        type=float,
        help="code amp std for normal",
        default=10,
    )
    parser.add_argument(
        "--code-min-distance",
        type=float,
        help="min distance between each code events",
        default=200,  # this is in ms (what should I put here) given the 200
    )
    parser.add_argument(
        "--baseline-end-onset-ms",
        type=float,
        help=" the time of first event. This is to make sure there is enough time to estimate the background activity",
        default=500,  # The first 500 ms of data no events happens
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def main(params):
    if params["train_or_test"] == "train":
        print("train")
        a = 200
        seed_list = np.linspace(9201, 9201 + 5 * a - 1, a)
        seed_list = [int(x) for x in seed_list]
        print(seed_list)
    elif params["train_or_test"] == "test":
        print("test")
        seed_list = [300, 301, 302, 303, 304]
        seed_list = [int(x) for x in seed_list]
        print(seed_list)
    else:
        print("vis")
        seed_list = [300]
        seed_list = [int(x) for x in seed_list]
        print(seed_list)

    out_path = params["out_path"]
    silent_bg_value = 1e-3
    baseline_end_onset = int(
        params["baseline_end_onset_ms"] / params["time_bin_resolution"]
    )

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

    # convert baseline to Hz in 1 ms ----------------------------------------------------#
    baseline_in_Hz = dict()
    baseline_in_Hz["-6.2126"] = 2
    baseline_in_Hz["-5.2933"] = 5
    baseline_in_Hz["-4.8203"] = 8
    baseline_in_Hz["-4.4988"] = 11
    baseline_in_Hz["-4.2546"] = 14
    baseline_in_Hz["-4.0574"] = 17
    baseline_in_Hz["-3.8918"] = 20
    baseline_in_Hz[
        "0"
    ] = 500  # this is for random spiking (with kernel they may get excited or silent)
    curr_baseline_in_Hz = baseline_in_Hz["{}".format(params["baseline_mean"])]

    # neural activity ----------------------------------------------------#
    # (num trials, num neurons, trial length after binning)
    baseline_across_neurons = (
        params["baseline_std"] * torch.randn(params["num_neurons"])
        + params["baseline_mean"]
    )

    seed_ctr = 0
    for seed in seed_list:
        seed_ctr += 1
        print("seed_ctr", seed_ctr)

        torch.manual_seed(seed)
        np.random.seed(seed)

        if params["train_or_test"] == "vis":
            raster = torch.zeros(
                (params["num_trials"], params["num_neurons"], params["trial_length"])
            )
            rate = torch.zeros(
                (params["num_trials"], params["num_neurons"], params["trial_length"])
            )
        y_count = torch.zeros(
            (
                params["num_trials"],
                params["num_neurons"],
                int(params["trial_length"] / params["time_bin_resolution"]),
            )
        )

        # events happens uniformaly at random (but we override them to have min distance between code of code_min_distance)
        code = torch.zeros(
            params["num_trials"], params["num_neurons"], kernel_num, code_dur
        )

        for trial_ctr in range(params["num_trials"]):
            # print("trial_ctr", trial_ctr)
            for kernel_ctr in range(kernel_num):
                # baseline_end_onset_ms is used to make sure that no event is in the first baseline_end_onset_ms of the trial
                code_supp = params["code_supp_list"][
                    np.random.randint(0, len(params["code_supp_list"]))
                ]
                index = np.random.choice(
                    int(
                        (code_dur - params["baseline_end_onset_ms"])
                        / params["code_min_distance"]
                    ),
                    code_supp,
                    replace=False,
                )
                index += int(
                    params["baseline_end_onset_ms"] / params["code_min_distance"]
                )
                for neuron_ctr in range(params["num_neurons"]):
                    # print(index)
                    amp = (
                        params["code_amplitude_std"] * torch.randn(code_supp)
                        + params["code_amplitude_mean"][kernel_ctr]
                    )
                    code[
                        trial_ctr,
                        neuron_ctr,
                        kernel_ctr,
                        index * params["code_min_distance"],
                    ] = amp

            for neuron_ctr in range(params["num_neurons"]):
                # get Hx
                Hx_org = F.conv_transpose1d(
                    code[trial_ctr, neuron_ctr], kernels_org_resolution
                )
                rate_neuron = torch.sigmoid(
                    Hx_org + baseline_across_neurons[neuron_ctr]
                )

                raster_neuron = torch.bernoulli(rate_neuron)

                if params["train_or_test"] == "vis":
                    rate[trial_ctr, [neuron_ctr], :] = rate_neuron
                    raster[trial_ctr, [neuron_ctr], :] = raster_neuron

                # do binning
                y_count[trial_ctr, [neuron_ctr], :] = np.add.reduceat(
                    raster_neuron,
                    np.arange(
                        0, raster_neuron.shape[-1], params["time_bin_resolution"]
                    ),
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
        code = code.detach().clone().cpu().numpy()
        if params["train_or_test"] == "vis":
            rate = rate.detach().clone().cpu().numpy()
            raster = raster.detach().clone().cpu().numpy()
        kernels_org_resolution_to_save = (
            kernels_org_resolution.detach().clone().cpu().numpy()
        )

        # ############# now save the data

        num_trials = y.shape[0]

        ###### save data
        data_dict = dict()
        data_dict["kernel_num"] = kernel_num
        if params["train_or_test"] == "vis":
            data_dict["raster"] = raster
            data_dict["rate"] = rate
        data_dict[
            "kernels"
        ] = kernels_org_resolution_to_save  # be careful (this is the kernel in 1 ms original resolution)
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
        data_dict["code_min_distance"] = params["code_min_distance"]
        data_dict["code_supp_list"] = params["code_supp_list"]
        data_dict["code_amplitude_mean"] = params["code_amplitude_mean"]
        data_dict["code_amplitude_std"] = params["code_amplitude_std"]

        for trial in range(num_trials):
            data_dict["trial{}".format(trial)] = dict()
            data_dict["trial{}".format(trial)][
                "type"
            ] = 0  # There is no event type. So 0

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
                save_path = "{}/simulated_{}neurons_{}trials_{}msbinres_{}Hzbaseline_seed{}_long_general_format_processed.npy".format(
                    params["out_path"],
                    params["num_neurons"],
                    params["num_trials"],
                    params["time_bin_resolution"],
                    curr_baseline_in_Hz,
                    seed,
                )
            elif params["train_or_test"] == "test":
                save_path = "{}/test_simulated_{}neurons_{}trials_{}msbinres_{}Hzbaseline_seed{}_long_general_format_processed.npy".format(
                    params["out_path"],
                    params["num_neurons"],
                    params["num_trials"],
                    params["time_bin_resolution"],
                    curr_baseline_in_Hz,
                    seed,
                )
            elif params["train_or_test"] == "vis":
                save_path = "{}/testvis_simulated_{}neurons_{}trials_{}msbinres_{}Hzbaseline_seed{}_long_general_format_processed.npy".format(
                    params["out_path"],
                    params["num_neurons"],
                    params["num_trials"],
                    params["time_bin_resolution"],
                    curr_baseline_in_Hz,
                    seed,
                )
            torch.save(data_dict, save_path)

            print(
                f"general format processed train data for dopamine eshel is saved at {save_path}!"
            )

        if 1:
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
