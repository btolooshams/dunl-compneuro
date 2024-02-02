"""
Copyright (c) 2020 Bahareh Tolooshams

dunl model

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class DUNL1D(torch.nn.Module):
    def __init__(self, params, H=None):
        super(DUNL1D, self).__init__()

        self.unrolling_num = params["unrolling_num"]  # number of encoder unrolling
        self.kernel_num = params["kernel_num"]  # number of kernels (dictionary)
        self.kernel_length = params["kernel_length"]  # the length of kernel
        self.unrolling_mode = params["unrolling_mode"]  # ista or fista

        # related to sparse regularization for code
        code_sparse_regularization = params[
            "code_sparse_regularization"
        ]  # regularization parameter lambda for code
        if ~isinstance(code_sparse_regularization, list):
            code_sparse_regularization = [code_sparse_regularization]
        self.code_sparse_regularization_decay = params[
            "code_sparse_regularization_decay"
        ]  # decay lambda parameters in unrolling
        # True for non-negative code, unless it's a list
        # 1 for pos, -1 for neg, 2 for both
        self.code_nonneg = params["code_nonneg"]
        if isinstance(self.code_nonneg, list):
            code_pos_sided = torch.zeros(1, self.kernel_num, 1)
            code_neg_sided = torch.zeros(1, self.kernel_num, 1)
            for c in range(self.kernel_num):
                if self.code_nonneg[c] == 1:
                    code_pos_sided[0, c, 0] = 1
                    code_neg_sided[0, c, 0] = 0
                elif self.code_nonneg[c] == -1:
                    code_pos_sided[0, c, 0] = 0
                    code_neg_sided[0, c, 0] = 1
                elif self.code_nonneg[c] == 2:
                    code_pos_sided[0, c, 0] = 1
                    code_neg_sided[0, c, 0] = 1

        if "code_q_regularization" in params:
            self.code_q_regularization = params[
                "code_q_regularization"
            ]  # True or False
            if self.code_q_regularization:
                code_q_regularization_matrix = params[
                    "code_q_regularization_matrix"
                ]  # this matrix is being applied to the norm of code from each kernel
                self.register_buffer(
                    "code_q_regularization_matrix", code_q_regularization_matrix
                )
                self.code_q_regularization_period = params[
                    "code_q_regularization_period"
                ]
                self.code_q_regularization_scale = params["code_q_regularization_scale"]
                self.code_q_regularization_norm_type = params[
                    "code_q_regularization_norm_type"
                ]
        else:
            self.code_q_regularization = False

        if "code_topk" in params:
            self.code_topk = params["code_topk"]  # pick only topk non-zero code entries
            if self.code_topk:
                self.code_topk_sparse = params["code_topk_sparse"]  # final sparsity
                self.code_topk_period = params[
                    "code_topk_period"
                ]  # period to apply topk
        else:
            self.code_topk = False

        # related to proximal operator (shrinkage or threshold)
        if "unrolling_prox" in params:
            self.unrolling_prox = params["unrolling_prox"]
        else:
            self.unrolling_prox = "shrinkage"

        if self.unrolling_prox == "shrinkage":
            self.relu = torch.nn.ReLU()
        elif self.punrolling_prox == "threshold":
            self.thres = torch.nn.Threshold(
                threshold=params["unrolling_threshold"], value=0
            )

        if "est_baseline_activity" in params:
            self.est_baseline_activity = params[
                "est_baseline_activity"
            ]  # flag to whether estimate baseline activity
        else:
            self.est_baseline_activity = False

        # this is related to the model distribution
        if "model_distribution" in params:
            self.model_distribution = params["model_distribution"]
        else:
            self.model_distribution = "gaussian"

        if self.model_distribution == "binomial":
            self.sigmoid = torch.nn.Sigmoid()
        elif self.model_distribution == "poisson":
            if "poisson_stability_name" in params:
                self.poisson_stability_name = params["poisson_stability_name"]
            else:
                self.poisson_stability_name = None

            if self.poisson_stability_name == "ELU":
                if "poisson_peak" in params:
                    self.poisson_peak = params["poisson_peak"]
                else:
                    self.poisson_peak = 1
                self.poisson_stability = torch.nn.ELU(alpha=self.poisson_peak)
            elif self.poisson_stability_name == "LeakyReLU":
                self.poisson_stability = torch.nn.LeakyReLU()
            elif self.poisson_stability_name == "ReLU":
                self.poisson_stability = torch.nn.ReLU()
            elif self.poisson_stability_name == "SELU":
                self.poisson_stability = torch.nn.SELU()

        # related to dictionary
        if "kernel_stride" in params:
            self.kernel_stride = params["kernel_stride"]
        else:
            self.kernel_stride = 1

        # related to gradient backrpop computation
        if "backward_gradient_decsent" in params:
            self.backward_gradient_decsent = params["backward_gradient_decsent"]
        else:
            self.backward_gradient_decsent = "bprop"
        if self.backward_gradient_decsent == "truncated_bprop":
            self.backward_truncated_bprop_itr = params["backward_truncated_bprop_itr"]

        # if H is None, initialize from random normal and then normalize each column
        if H is None:
            H = torch.randn((self.kernel_num, 1, self.kernel_length))
            H = torch.nn.functional.normalize(H, p=2, dim=-1)

            # apply smoothing
            if "kernel_init_smoother" in params:
                if params["kernel_init_smoother"]:
                    gaussian_smoother_t = np.arange(-6, 6, 0.1)[:-1]
                    gaussian_smoother = (
                        1
                        / (np.sqrt(2 * np.pi) * params["kernel_init_smoother_sigma"])
                        * np.exp(
                            -(
                                (
                                    gaussian_smoother_t
                                    / params["kernel_init_smoother_sigma"]
                                )
                                ** 2
                            )
                            / 2.0
                        )
                    )
                    gaussian_smoother = torch.FloatTensor(
                        np.array([[gaussian_smoother]])
                    )
                    H = torch.nn.functional.conv1d(H, gaussian_smoother, padding="same")
                    H = torch.nn.functional.normalize(H, p=2, dim=-1)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(H[:,0,:].T)
            # plt.show()
            # exit()

        self.register_parameter("H", torch.nn.Parameter(H))
        self.register_buffer("unrolling_alpha", torch.tensor(params["unrolling_alpha"]))
        self.register_buffer(
            "code_sparse_regularization",
            torch.unsqueeze(torch.tensor(code_sparse_regularization), dim=1),
        )
        if isinstance(self.code_nonneg, list):
            self.register_buffer("code_pos_sided", code_pos_sided)
            self.register_buffer("code_neg_sided", code_neg_sided)

    def get_param(self, name):
        """return the registered parameter with name"""
        return self.state_dict(keep_vars=True)[name]

    def normalize_kernel(self):
        """normalize each kernel to have l2 norm of 1."""
        self.get_param("H").data = torch.nn.functional.normalize(
            self.get_param("H").data, p=2, dim=-1
        )

    def nonneg_kernel(self, nonneg_kernel_indicator=None):
        """project the kernels to non-negative values depending on the nonneg_kernel_indicator list, if None, apply to all"""
        if nonneg_kernel_indicator is None:
            self.get_param("H").data = self.relu(self.get_param("H").data)
        else:
            for ind in range(len(nonneg_kernel_indicator)):
                if nonneg_kernel_indicator[ind]:
                    self.get_param("H").data[ind] = self.relu(
                        self.get_param("H").data[ind]
                    )

    def shrinkage_nonlin(self, x, code_reg, code_supp=None):
        """shrinkage proximal operator"""
        if code_supp is None:
            code_supp_r = 1
        else:
            code_supp_r = code_supp * 1

        if isinstance(self.code_nonneg, list):
            x_pos = self.code_pos_sided * self.relu(x - self.unrolling_alpha * code_reg)
            x_neg = -self.code_neg_sided * self.relu(
                -x - self.unrolling_alpha * code_reg
            )
            x = (x_pos + x_neg) * code_supp_r
        else:
            if self.code_nonneg:
                x = self.relu(x - self.unrolling_alpha * code_reg) * code_supp_r
            else:
                x = (
                    self.relu(torch.abs(x) - self.unrolling_alpha * code_reg)
                    * torch.sign(x)
                    * code_supp_r
                )
        return x

    def hardthresholding_nonlin(self, x, code_supp=None):
        """hard-thresholding proximal operator"""
        if code_supp is None:
            code_supp_r = 1
        else:
            code_supp_r = code_supp * 1

        if isinstance(self.code_nonneg, list):
            x_pos = self.code_pos_sided * self.thres(x)
            x_neg = -self.code_neg_sided * self.thres(-x)
            x = (x_pos + x_neg) * code_supp_r
        else:
            if self.code_nonneg:
                x = self.thres(x) * code_supp_r
            else:
                x = self.thres(torch.abs(x)) * torch.sign(x) * code_supp_r
        return x

    def topk_nonlin(self, x, r):
        """set all but top k code entries to zero"""
        code_topk = int(
            self.code_topk_sparse
            * ((self.unrolling_num - r + 1) / self.unrolling_num + 1)
        )
        values, indices = torch.topk(torch.abs(x), code_topk, dim=-1)
        mask = x * 0
        x = mask.scatter(2, indices, values) * torch.sign(x)
        # topk function (argmax) is not differentiable, so break the gradient for bp
        x = x.clone().detach().requires_grad_(False)
        return x

    def get_Qx(self, x):
        if self.code_q_regularization_norm_type:
            x_norm = torch.norm(
                x, p=self.code_q_regularization_norm_type, dim=-1, keepdim=True
            )
            # (batch, kernels, 1) = (kernel, kenrel) x (batch, kernels, 1)
            Qx = torch.matmul(self.code_q_regularization_matrix, x_norm)
        else:
            device = x.device
            # set code_q_regularization_norm_type to None to come here.
            # This applies the q reg at each time in trial.
            num_batches, kernel_num, time_length = x.shape
            code_q_regularization_matrix_time = torch.zeros(
                time_length * kernel_num, time_length * kernel_num
            ).to(device)
            # make a block for each entry
            for i in range(self.code_q_regularization_matrix.shape[0]):
                for j in range(self.code_q_regularization_matrix.shape[1]):
                    block_ij = (
                        torch.eye(time_length, device=device)
                        * self.code_q_regularization_matrix[i, j]
                    )
                    code_q_regularization_matrix_time[
                        i * time_length : (i + 1) * time_length,
                        j * time_length : (j + 1) * time_length,
                    ] = block_ij

            x_kerneltime = x.reshape(num_batches, kernel_num * time_length, 1)
            # (batch, kernels * time) = (kernel, kenrel * time) x (batch, kernels * time, 1)
            Qx_kerneltime = torch.matmul(
                code_q_regularization_matrix_time, x_kerneltime
            )
            # (batch, kernels, time)
            Qx = Qx_kerneltime.reshape(num_batches, kernel_num, time_length)

        return Qx

    def encode_ista(self, y, a=0, code_supp=None):
        """
        unrolled encoder using ista

        return (code, baseline activity)

        """

        num_batches = y.shape[0]
        device = y.device

        D_enc = torch.nn.functional.conv1d(
            y, self.get_param("H"), stride=self.kernel_stride
        ).shape[-1]

        x = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)

        code_sparse_regularization_r = self.code_sparse_regularization
        for r in range(self.unrolling_num):
            Hxa = (
                torch.nn.functional.conv_transpose1d(
                    x, self.get_param("H"), stride=self.kernel_stride
                )
                + a
            )

            if self.model_distribution == "gaussian":
                res = y - Hxa
            elif self.model_distribution == "binomial":
                res = y - self.sigmoid(Hxa)
            elif self.model_distribution == "poisson":
                res = y - torch.exp(Hxa)
                if self.poisson_stability_name is not None:
                    res = self.poisson_stability(res)

            x = x + self.unrolling_alpha * torch.nn.functional.conv1d(
                res, self.get_param("H"), stride=self.kernel_stride
            )

            if self.unrolling_prox == "shrinkage":
                code_sparse_regularization_r *= self.code_sparse_regularization_decay
                x = self.shrinkage_nonlin(x, code_sparse_regularization_r, code_supp)
            elif self.unrolling_prox == "threshold":
                x = self.hardthresholding_nonlin(x, code_supp)

            if (
                self.code_q_regularization
                and (r + 1) % self.code_q_regularization_period == 0
            ):
                Qx = self.get_Qx(x)
                q_reg = self.code_q_regularization_scale * Qx
                x = self.shrinkage_nonlin(x, q_reg, code_supp)

            if self.code_topk and (r + 1) % self.code_topk_period == 0:
                x = self.topk_nonlin(x, r)

            if (
                self.backward_gradient_decsent == "truncated_bprop"
                and self.unrolling_num - r > self.backward_truncated_bprop_itr
            ):
                x = x.clone().detach().requires_grad_(False)

        if self.code_topk:
            x = self.topk_nonlin(x, self.unrolling_num)

        return x, a

    def encode_fista(self, y, a=0, code_supp=None):
        """
        unrolled encoder using fista

        return (code, baseline activity)

        """

        num_batches = y.shape[0]
        device = y.device

        D_enc = torch.nn.functional.conv1d(
            y, self.get_param("H"), stride=self.kernel_stride
        ).shape[-1]

        x_old = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        x_tmp = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        x_new = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        t_old = (torch.zeros(1, device=device) + 1.0).float()

        code_sparse_regularization_r = self.code_sparse_regularization * 1
        for r in range(self.unrolling_num):
            Hxa = (
                torch.nn.functional.conv_transpose1d(
                    x_tmp, self.get_param("H"), stride=self.kernel_stride
                )
                + a
            )

            if self.model_distribution == "gaussian":
                res = y - Hxa
            elif self.model_distribution == "binomial":
                res = y - self.sigmoid(Hxa)
            elif self.model_distribution == "poisson":
                res = y - torch.exp(Hxa)
                if self.poisson_stability_name is not None:
                    res = self.poisson_stability(res)

            x_new = x_tmp + self.unrolling_alpha * torch.nn.functional.conv1d(
                res, self.get_param("H"), stride=self.kernel_stride
            )

            if self.unrolling_prox == "shrinkage":
                code_sparse_regularization_r *= self.code_sparse_regularization_decay
                x_new = self.shrinkage_nonlin(
                    x_new, code_sparse_regularization_r, code_supp
                )
            elif self.unrolling_prox == "threshold":
                x_new = self.hardthresholding_nonlin(x_new, code_supp)

            if (
                self.code_q_regularization
                and (r + 1) % self.code_q_regularization_period == 0
            ):
                Qx = self.get_Qx(x_new)
                q_reg = self.code_q_regularization_scale * Qx
                x_new = self.shrinkage_nonlin(x_new, q_reg, code_supp)

            if self.code_topk and (r + 1) % self.code_topk_period == 0:
                x_new = self.topk_nonlin(x_new, r)

            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_old.pow(2))) / 2.0
            x_tmp = x_new + (t_old - 1.0) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

            if (
                self.backward_gradient_decsent == "truncated_bprop"
                and self.unrolling_num - r > self.backward_truncated_bprop_itr
            ):
                x_new = x_new.clone().detach().requires_grad_(False)
                x_tmp = x_tmp.clone().detach().requires_grad_(False)
                x_old = x_old.clone().detach().requires_grad_(False)

        if self.code_topk:
            x_new = self.topk_nonlin(x_new, self.unrolling_num)

        return x_new, a

    def encode_ista_est_baseline_activity(self, y, a=0, code_supp=None):
        """
        unrolled encoder using ista and estimate baseline activity

        return (code, estimated baseline activity)

        """

        num_batches = y.shape[0]
        device = y.device

        D_enc = torch.nn.functional.conv1d(
            y, self.get_param("H"), stride=self.kernel_stride
        ).shape[-1]

        a_est = torch.zeros(num_batches, 1, 1, device=device) + a

        x = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)

        code_sparse_regularization_r = self.code_sparse_regularization * 1
        for r in range(self.unrolling_num):
            Hxa = (
                torch.nn.functional.conv_transpose1d(
                    x, self.get_param("H"), stride=self.kernel_stride
                )
                + a_est
            )

            if self.model_distribution == "gaussian":
                res = y - Hxa
            elif self.model_distribution == "binomial":
                res = y - self.sigmoid(Hxa)
            elif self.model_distribution == "poisson":
                res = y - torch.exp(Hxa)
                if self.poisson_stability_name is not None:
                    res = self.poisson_stability(res)

            a_est = a_est + torch.mean(res, dim=-1, keepdims=True)
            x = x + self.unrolling_alpha * torch.nn.functional.conv1d(
                res, self.get_param("H"), stride=self.kernel_stride
            )

            if self.unrolling_prox == "shrinkage":
                code_sparse_regularization_r *= self.code_sparse_regularization_decay
                x = self.shrinkage_nonlin(x, code_sparse_regularization_r, code_supp)
            elif self.unrolling_prox == "threshold":
                x = self.hardthresholding_nonlin(x, code_supp)

            if (
                self.code_q_regularization
                and (r + 1) % self.code_q_regularization_period == 0
            ):
                Qx = self.get_Qx(x)
                q_reg = self.code_q_regularization_scale * Qx
                x = self.shrinkage_nonlin(x, q_reg, code_supp)

            if self.code_topk and (r + 1) % self.code_topk_period == 0:
                x = self.topk_nonlin(x, r)

            if (
                self.backward_gradient_decsent == "truncated_bprop"
                and self.unrolling_num - r > self.backward_truncated_bprop_itr
            ):
                x = x.clone().detach().requires_grad_(False)

        if self.code_topk:
            x = self.topk_nonlin(x, self.unrolling_num)

        return x, a_est

    def encode_fista_est_baseline_activity(self, y, a=0, code_supp=None):
        """
        unrolled encoder using fista and estimate baseline activity

        return (code, estimated baseline activity)

        """

        num_batches = y.shape[0]
        device = y.device

        D_enc = torch.nn.functional.conv1d(
            y, self.get_param("H"), stride=self.kernel_stride
        ).shape[-1]

        a_est_old = torch.zeros(num_batches, 1, 1, device=device) + a
        a_est_tmp = torch.zeros(num_batches, 1, 1, device=device) + a
        a_est_new = torch.zeros(num_batches, 1, 1, device=device) + a

        x_old = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        x_tmp = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        x_new = torch.zeros(num_batches, self.kernel_num, D_enc, device=device)
        t_old = (torch.zeros(1, device=device) + 1.0).float()

        code_sparse_regularization_r = self.code_sparse_regularization * 1
        for r in range(self.unrolling_num):
            Hxa = (
                torch.nn.functional.conv_transpose1d(
                    x_tmp, self.get_param("H"), stride=self.kernel_stride
                )
                + a_est_tmp
            )

            if self.model_distribution == "gaussian":
                res = y - Hxa
            elif self.model_distribution == "binomial":
                res = y - self.sigmoid(Hxa)
            elif self.model_distribution == "poisson":
                res = y - torch.exp(Hxa)
                if self.poisson_stability_name is not None:
                    res = self.poisson_stability(res)

            a_est_new = a_est_tmp + torch.mean(res, dim=-1, keepdims=True)

            x_new = x_tmp + self.unrolling_alpha * torch.nn.functional.conv1d(
                res, self.get_param("H"), stride=self.kernel_stride
            )

            if self.unrolling_prox == "shrinkage":
                code_sparse_regularization_r *= self.code_sparse_regularization_decay
                x_new = self.shrinkage_nonlin(
                    x_new, code_sparse_regularization_r, code_supp
                )
            elif self.unrolling_prox == "threshold":
                x_new = self.hardthresholding_nonlin(x_new, code_supp)

            if (
                self.code_q_regularization
                and (r + 1) % self.code_q_regularization_period == 0
            ):
                Qx = self.get_Qx(x_new)
                q_reg = self.code_q_regularization_scale * Qx
                x_new = self.shrinkage_nonlin(x_new, q_reg, code_supp)

            if self.code_topk and (r + 1) % self.code_topk_period == 0:
                x_new = self.topk_nonlin(x_new, r)

            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_old.pow(2))) / 2.0
            a_est_tmp = a_est_new + (t_old - 1) / t_new * (a_est_new - a_est_old)
            x_tmp = x_new + (t_old - 1.0) / t_new * (x_new - x_old)

            a_est_old = a_est_new
            x_old = x_new
            t_old = t_new

            if (
                self.backward_gradient_decsent == "truncated_bprop"
                and self.unrolling_num - r > self.backward_truncated_bprop_itr
            ):
                x_new = x_new.clone().detach().requires_grad_(False)
                x_tmp = x_tmp.clone().detach().requires_grad_(False)
                x_old = x_old.clone().detach().requires_grad_(False)

        if self.code_topk:
            x_new = self.topk_nonlin(x_new, self.unrolling_num)

        return x_new, a_est_new

    def encode(self, y, a=0, code_supp=None):
        """network forward encoder mapping"""
        if self.unrolling_mode == "ista":
            if self.est_baseline_activity:
                x, a_est = self.encode_ista_est_baseline_activity(
                    y,
                    a,
                    code_supp,
                )
            else:
                x, a_est = self.encode_ista(y, a, code_supp)
        elif self.unrolling_mode == "fista":
            if self.est_baseline_activity:
                x, a_est = self.encode_fista_est_baseline_activity(
                    y,
                    a,
                    code_supp,
                )
            else:
                x, a_est = self.encode_fista(y, a, code_supp)

        return x, a_est

    def decode(self, x, a=0):
        """shallow decoder using the kernels"""
        yhat = (
            torch.nn.functional.conv_transpose1d(
                x, self.get_param("H"), stride=self.kernel_stride
            )
            + a
        )

        return yhat

    def forward(self, y, a=0, code_supp=None):
        """network forward mapping"""
        # encoder
        x, a_est = self.encode(y, a, code_supp)

        # decoder
        yhat = self.decode(x, a_est)

        return yhat
