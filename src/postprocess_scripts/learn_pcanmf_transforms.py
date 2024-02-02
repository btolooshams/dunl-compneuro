"""
Copyright (c) 2020 Bahareh Tolooshams

plot code data

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import scipy as sp
import os
import pickle
import argparse
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default="../results/dopaminecalcium_kernellength60_kernelnum5_code2211n1_kernel00011_qreg_2023_07_13_11_37_31",
    )
    parser.add_argument(
        "--num-comp",
        type=int,
        help="number of components",
        default=2,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        help="max iter for nmf",
        default=1000,
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There {} dataset in the folder.".format(len(data_path_list)))

    # create folders -------------------------------------------------------#
    postprocess_path = os.path.join(
        params["res_path"],
        "postprocess",
    )

    # load data -------------------------------------------------------#
    y_all = list()
    label_all = list()

    for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        # (neuron, time, trials)
        y = np.load(
            os.path.join(postprocess_path, "y_for_pcanmf_{}.npy".format(datafile_name))
        )
        label = np.load(
            os.path.join(
                postprocess_path, "label_for_pcanmf_{}.npy".format(datafile_name)
            )
        )

        y = np.transpose(y, (0, 2, 1))
        y = np.reshape(y, (-1, y.shape[-1]))

        y_all.append(y)
        label_all.append(label)

    y_all = np.concatenate(y_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    # do transform -------------------------------------------------------#
    scaler = StandardScaler()
    y_all_standardized = scaler.fit_transform(y_all)

    print("y_all", y_all.shape)
    print("y_standardized", y_all_standardized.shape)

    pca_transform = PCA(n_components=params["num_comp"])
    y_pca_coeff = pca_transform.fit_transform(y_all_standardized)
    nmf_transform = NMF(n_components=params["num_comp"], max_iter=params["max_iter"])
    y_nmf_coeff = nmf_transform.fit_transform(y_all - np.min(y_all))

    pickle.dump(
        pca_transform,
        open(
            os.path.join(
                postprocess_path, "pca_transform_{}.pkl".format(params["num_comp"])
            ),
            "wb",
        ),
    )
    pickle.dump(
        nmf_transform,
        open(
            os.path.join(
                postprocess_path, "nmf_transform_{}.pkl".format(params["num_comp"])
            ),
            "wb",
        ),
    )
    pickle.dump(
        scaler, open(os.path.join(postprocess_path, "scaler_transform.pkl"), "wb")
    )


if __name__ == "__main__":
    main()
