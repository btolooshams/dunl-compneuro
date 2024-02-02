"""
Copyright (c) 2020 Bahareh Tolooshams

utils

:author: Bahareh Tolooshams
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import linear_model
import hillfit
import scipy.signal as scp
from scipy.stats import linregress


def perform_regression(
    x,
    y,
    mode="linreg",
    y_fit=[0.3, 0.5, 1.2, 2.5, 5, 8, 11],
):
    y_fit = np.array(y_fit).reshape(-1, 1)

    if mode == "linreg":
        model = linear_model.LinearRegression()
        model.fit(y.reshape(-1, 1), x)

        x_fit = model.predict(y_fit)
    elif mode == "hillfit":
        model = hillfit.HillFit(y, x)
        try:
            model.fitting(generate_figure=False, print_r_sqr=False)
            x_fit = model.y_fit
            y_fit = model.x_fit
        except:
            x_fit = None
            y_fit = None
    else:
        raise NotImplementedError("regression is not implemented!")

    return model, y_fit, x_fit


def perform_sorted_regression(
    x,
    y,
    mode="linreg",
    y_fit=[0.3, 0.5, 1.2, 2.5, 5, 8, 11],
):
    y_sorted_indices = np.argsort(y)

    x_sorted = x[y_sorted_indices]
    y_sorted = y[y_sorted_indices]

    model, y_fit, x_fit = perform_regression(x_sorted, y_sorted, mode, y_fit)

    return model, y_fit, x_fit


def normalize_for_give_sorted_curve_x(x):
    x_sameminmax = (x - torch.min(x, dim=0, keepdim=True)[0]) / (
        torch.max(x, dim=0, keepdim=True)[0] - torch.min(x, dim=0, keepdim=True)[0]
    )
    return x_sameminmax


def give_sorted_curve_x_old(x):
    a1 = torch.mean(x[:2], dim=0)
    a2 = torch.mean(x[2:-1], dim=0)
    a3 = torch.mean(x[-1:], dim=0)
    curveture = (a3 - a2) / (a2 - a1)
    sort_ids = np.flip(np.argsort(curveture.clone().detach().cpu().numpy()))
    return x, sort_ids


def give_sorted_curve_x(x):
    slope = list()
    for i in range(x.shape[-1]):
        xi = x[:, i]
        reg = linregress(np.linspace(0, 1, len(xi)), xi)
        slope.append(reg.slope)
    sort_ids = np.flip(np.argsort(slope))
    return x, sort_ids


def compute_signed_distance_from_diag(x, y):
    d_list = np.zeros(len(x))
    for i in range(len(x)):
        a = x[i]
        b = y[i]
        p = np.array([a, b])
        p_diag = np.array([1, 1])
        d = np.cross(p, p_diag) / np.linalg.norm(p)
        d_list[i] = d
    return d_list


def compute_zc(code, reward, reward_amount_list):
    k_indices = np.array(reward_amount_list) + 0.5

    critvals = np.zeros(len(k_indices))

    for k_ctr in range(len(k_indices)):
        ix_above = reward > k_indices[k_ctr]
        ix_below = reward < k_indices[k_ctr]

        critvals[k_ctr] = torch.sum(code[ix_above] > 0) + torch.sum(code[ix_below] < 0)

    critvals = critvals + 0.001 * np.random.randn(len(k_indices))
    mcv = np.argmax(critvals)

    zc = k_indices[mcv]

    return zc


def get_smooth_kernel(t, tau=40):
    return (1 - torch.exp(-t)) * torch.exp(-t / tau)


def smooth_raster(raster, tau=40):
    # smoothing kernel
    t_kernel = torch.linspace(
        0,
        2.5 * tau,
        int((2.5 * tau) + 1),
    )
    smooth_kernel = get_smooth_kernel(t_kernel, tau)
    smooth_kernel /= torch.sum(smooth_kernel)
    smooth_kernel = torch.reshape(smooth_kernel, (1, 1, len(smooth_kernel)))

    raster_smooth_count = np.add.reduceat(
        raster,
        np.arange(0, raster.shape[-1], 1),
        axis=-1,
    )
    raster_smooth = F.conv1d(
        raster_smooth_count,
        torch.flip(smooth_kernel, dims=[-1]),
        padding="same",
    )

    return raster_smooth


def time_rescale(rate, data):
    """
    Application of the time rescaling theorem to data vector
    Inputs
    ======
    rate: 1-D array
        estimated rate for the given trial
    data: 1-D array
        binary data vector for the given trial
    Outputs
    =======
    rescaled_time: 1-D array
        re-ordered and uniform distribution converted rate differences
    """
    assert len(rate) == len(data), "The lengths of rate and data do not match"

    rescaled_time = []

    indices = np.where(data > 0)[0]
    previous_idx = 0
    for idx in indices:
        rescaled_time.append(np.sum(rate[previous_idx:idx]))
        previous_idx = idx
    rescaled_time = 1 - 1 / np.exp(
        rescaled_time
    )  # rescaled time should now follow uniform distribution
    rescaled_time = np.sort(rescaled_time)
    return rescaled_time


def time_rescale_ensemble(rate_set, data_set):
    """
    Application of the time rescaling theorem for a single neuron with multiple trials
    Inputs
    ======
    rate: dictionary
        Key: Trial #
        Value: estimated rate for the given trial
    data: dictionary
        Key: Trial #
        Value: binary data vector for the given trial
    Outputs
    =======
    rescaled_time_set: dictionary
        re-ordered and uniform distribution converted rate differences
        Each key corresponds to a trial #
    rescaled_time_total: 1-D array
        rescaled time entities concatenated across all trials (Used for plotting the time-rescale plot)
    """
    assert len(rate_set) == len(data_set), "The size of the sets do not match"
    numOftrials = len(rate_set)

    rescaled_time_set = {}
    rescaled_time_total = np.empty(0)

    for idx in np.arange(numOftrials):
        temp_set = time_rescale(rate_set[idx], data_set[idx])
        rescaled_time_set[idx] = temp_set
        rescaled_time_total = np.concatenate((rescaled_time_total, temp_set))

    rescaled_time_total = np.sort(rescaled_time_total)

    return rescaled_time_set, rescaled_time_total


def predict_event(y, threshold):
    num_data = y.shape[0]

    event_matrix = y.clone() * 0
    for i in range(num_data):
        yi = y[i, 0, :].clone().detach().cpu().numpy()

        index_peak = scp.find_peaks(yi, height=threshold)[0]
        event_matrix[i, 0, index_peak] = 1

    return event_matrix


def compute_hit_rate(x, xhat, tol=1):
    if tol > 0:
        kernel = torch.ones(1, 1, 2 * tol + 1, device=x.device)
        target = F.conv1d(x, kernel, padding="same")

        x_ind = x.clone()
        x_ind[x_ind > 0] = 1
        target[target > 0] = 1
    else:
        target = x.clone()
        target[target > 0] = 1

        x_ind = target.clone()

    hit = xhat.clone()
    hit[hit > 0] = 1

    hit_false = hit.clone()
    hit_false[target > 0] = 0

    hit_rate = torch.sum(hit * target, dim=(-1, -2)) / torch.sum(x_ind, dim=(-1, -2))
    false_rate = torch.sum(hit_false, dim=(-1, -2)) / torch.sum(hit, dim=(-1, -2))

    hit_rate = torch.nan_to_num(hit_rate, nan=0.0)
    false_rate = torch.nan_to_num(false_rate, nan=1.0)

    hit_rate = torch.mean(hit_rate).item()
    false_rate = torch.mean(false_rate).item()

    return hit_rate, false_rate


def swap_kernel(x, i, j):
    x_clone = x.clone()
    x[i] = x_clone[j]
    x[j] = x_clone[i]
    return x


def swap_code(x, i, j):
    x_clone = x.clone()
    x[:, :, i] = x_clone[:, :, j]
    x[:, :, j] = x_clone[:, :, i]
    return x


def compute_dictionary_error(h, h_est):
    num_conv = h.shape[0]

    err = []
    for conv in range(num_conv):
        corr = np.sum(h[conv] * h_est[conv])
        err.append(np.sqrt(np.maximum(1 - corr**2, 0.0)))

    return err


def compute_dictionary_error_with_cross_correlation(h, h_est):
    num_conv = h.shape[0]
    err = []
    for conv in range(num_conv):
        cross_corr = np.correlate(h[conv], h_est[conv], "full")
        corr = np.max(np.abs(cross_corr))
        err.append(np.sqrt(np.maximum(1 - corr**2, 0.0)))

    return err
