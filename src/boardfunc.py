"""
Copyright (c) 2020 Bahareh Tolooshams

functions to send info to borad during training

:author: Bahareh Tolooshams
"""
import io
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt


def log_loss(
    writer, epoch, net, data_loader_list, criterion, params, loss_name="", device="cpu"
):
    """compute the loss from criterion on data through net and upload to the board through writer"""
    total_loss = []

    with torch.no_grad():
        for data_loader in data_loader_list:
            for idx, (y_load, x_load, a_load, type_load) in tqdm(
                enumerate(data_loader), disable=True
            ):
                y = torch.reshape(
                    y_load, (int(y_load.shape[0] * y_load.shape[1]), 1, y_load.shape[2])
                )
                a = torch.reshape(
                    a_load, (int(a_load.shape[0] * a_load.shape[1]), 1, a_load.shape[2])
                )
                # repeat x for how many neurons are they into the 0 (trial) dim
                x = torch.repeat_interleave(x_load, a_load.shape[1], dim=0)

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
                loss = criterion(y, yhat)

                total_loss.append(loss.item())

    writer.add_scalar("loss/{}".format(loss_name), np.mean(total_loss), epoch)
    writer.flush()

    return writer, np.mean(total_loss)


def log_kernels(writer, epoch, net):
    """plot the kernels on the board through writer"""

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    kernels = (
        torch.squeeze(net.get_param("H"), dim=1).data.clone().detach().cpu().numpy()
    )
    kernel_num = kernels.shape[0]
    a = int(np.ceil(kernel_num / 2))

    fig, axn = plt.subplots(a, 2, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for kernel_index in range(kernel_num):
        plt.subplot(a, 2, kernel_index + 1)
        plt.plot(kernels[kernel_index])
        plt.xlabel("Time")
    writer.add_figure("kernels", fig, epoch)
    writer.flush()
    plt.close()
    return writer


def log_data_inputrec(writer, epoch, y, yhat, model_distribution="gaussian"):
    """plot raw data y and yhat on the board through writer"""

    if model_distribution == "gaussian":
        yhat_after_link = yhat
    elif model_distribution == "binomial":
        yhat_after_link = torch.sigmoid(yhat)
    elif model_distribution == "poisson":
        yhat_after_link = torch.exp(yhat)

    i = 0
    yi = y[i, 0].clone().detach().cpu().numpy()
    yihat_after_link = yhat_after_link[i, 0].clone().detach().cpu().numpy()

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.plot(yi, color="black", label="raw", lw=0.7)
    plt.plot(yihat_after_link, color="blue", label="rec", lw=0.7)
    plt.xlabel("Time")
    plt.legend()
    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)

    writer.add_figure("data/inputrec", plt.gcf(), epoch)
    writer.flush()
    plt.close()
    return writer


def log_data_code(writer, epoch, xhat, x=None):
    """plot code on the board through writer"""

    i = 0
    if x is not None:
        code = x[i].clone().detach().cpu().numpy()
    codehat = xhat[i].clone().detach().cpu().numpy()

    kernel_num = codehat.shape[0]
    a = int(np.ceil(kernel_num / 2))

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(2, a, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for kernel_index in range(kernel_num):
        plt.subplot(2, a, kernel_index + 1)
        if x is not None:
            plt.plot(code[kernel_index], "o", color="black", lw=0.7)
        plt.plot(codehat[kernel_index], ".", color="green", lw=0.7)
        plt.xlabel("Time")
        plt.subplots_adjust(wspace=None, hspace=None)

    writer.add_figure("data/code", fig, epoch)
    writer.flush()
    plt.close()
    return writer
