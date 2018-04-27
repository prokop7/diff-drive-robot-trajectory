# def apply_filter(x_plot, Q, R):
#     x_hat = [0] * len(x_plot)
#     x_hat_minus = [0] * len(x_plot)
#     p_minus = [0] * len(x_plot)
#     p = [0] * len(x_plot)
#     k = [0] * len(x_plot)
#
#     x_hat[0] = 0
#     p[0] = 1
#
#     for i in range(1, len(x_plot)):
#         x_hat_minus[i] = x_hat[i - 1]
#         p_minus[i] = p[i - 1] + Q
#         k[i] = p_minus[i] / (p_minus[i] + R)
#         x_hat[i] = x_hat_minus[i] + k[i] * (x_plot[i] - x_hat_minus[i])
#         p[i] = (1 - k[i]) * p_minus[i]
#
#     return x_hat
import numpy as np
from numpy.linalg import inv


def apply_filter(x_plot, Q, R, sz):
    x_hat = np.zeros(sz)
    x_hat_minus = np.zeros(sz)
    p_minus = np.zeros(sz)
    p = np.zeros(sz)
    k = np.zeros((sz[0], 2))
    x_hat[0] = 0
    p[0] = 1
    C = np.matrix([[1], [1]])

    for i in range(1, len(x_plot)):
        x_hat_minus[i] = x_hat[i - 1]
        p_minus[i] = p[i - 1] + Q
        k[i] = (p_minus[i] * C.transpose()) * inv(C * p_minus[i] * C.transpose() + R)
        x_hat[i] = x_hat_minus[i] + k[i] * (x_plot[i].transpose() - C * x_hat_minus[i])
        p[i] = (np.identity(1) - k[i] * C) * p_minus[i]

    return x_hat
