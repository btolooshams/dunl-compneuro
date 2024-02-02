"""
Copyright (c) 2020 Tolooshams

create data generator

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os
import scipy.io as sio


class DUNLdataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)

        self.data_path = data_path

        self.y = data["y"]  # recorded data dim(num_trials, num_neurons, trial_length)
        self.x = data[
            "x"
        ]  # event onsets dim(num_trials, num_kernels, trial_length - kernel_length + 1)
        self.a = data["a"]  # baseline dim(num_trials, num_neurons, 1)
        self.type = data["type"]  # trial type dim(num_trials)
        self.num_data = self.y.shape[0]  # number of trials

        print(
            "x is shared among neurons. It is a function of trials, and number of kernels!"
        )

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return (
            self.y[idx],
            self.x[idx],
            self.a[idx],
            self.type[idx],
        )


class DUNLdatasetwithRaster(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)

        self.data_path = data_path

        self.time_org_resolution = data["time_org_resolution"]
        self.time_bin_resolution = data["time_bin_resolution"]

        self.raster = data["raster"]
        self.y = data["y"]  # recorded data dim(num_trials, num_neurons, trial_length)
        self.x = data[
            "x"
        ]  # event onsets dim(num_trials, num_kernels, trial_length - kernel_length + 1)
        self.a = data["a"]  # baseline dim(num_trials, num_neurons, 1)
        self.type = data["type"]  # trial type dim(num_trials)
        self.num_data = self.y.shape[0]  # number of trials

        print(
            "x is shared among neurons. It is a function of trials, and number of kernels!"
        )

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return (
            self.y[idx],
            self.x[idx],
            self.a[idx],
            self.type[idx],
            self.raster[idx],
        )


class DUNLdatasetwithRasterWithCodeRate(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)

        self.data_path = data_path

        self.time_org_resolution = data["time_org_resolution"]
        self.time_bin_resolution = data["time_bin_resolution"]

        self.raster = data["raster"]
        self.y = data["y"]  # recorded data dim(num_trials, num_neurons, trial_length)
        self.x = data[
            "x"
        ]  # event onsets dim(num_trials, num_kernels, trial_length - kernel_length + 1)
        self.a = data["a"]  # baseline dim(num_trials, num_neurons, 1)
        self.type = data["type"]  # trial type dim(num_trials)
        self.num_data = self.y.shape[0]  # number of trials
        self.codes = data["codes"]
        self.rate = data["rate"]

        print(
            "x is shared among neurons. It is a function of trials, and number of kernels!"
        )

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return (
            self.y[idx],
            self.x[idx],
            self.a[idx],
            self.type[idx],
            self.raster[idx],
            self.codes[idx],
            self.rate[idx],
        )
