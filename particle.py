import math
from random import random

import numpy as np
import matplotlib.pyplot as plt
from filterpy.discrete_bayes import normalize
from filterpy.stats import plot_gaussian_pdf, scipy
from numpy.matlib import randn
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats

from numpy.random.mtrand import uniform

magic_coeff = 0.055
wheel_radius = 2.7
wheel_base_half = 7.5
sonar_zero_distance = 13.8
init_x = 0.0
init_y = 0.0
init_angle = 0.0
x_cam_noise = (0.0, 49.0)
y_cam_noise = (0.0, 49.0)
gyro_noise = (0.0, math.radians(16.0))
sonar_normal_noise = (0.0, 4.0)
sonar_invalid_noise = (0.0, 1e+6)


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 1] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 2] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 0] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 0] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 5))
    # angle
    particles[:, 0] = mean[2] + (randn(N) * std[2])
    # lv
    particles[:, 1] = 11 + (randn(N) * 5.5)
    # rv
    particles[:, 2] = 11 + (randn(N) * 5.5)
    # x
    particles[:, 3] = mean[0] + (randn(N) * std[0])
    # y
    particles[:, 4] = mean[1] + (randn(N) * std[1])
    particles[:, 0] %= 2 * np.pi
    return particles


def predict(particles, u, std, dt):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""
    (vl, vr) = u
    R = wheel_base_half * (vl + vr) / (vr - vl)
    wt = (vr - vl) / (wheel_base_half * 2) * dt

    ICCx = particles[:, 3] - R * np.sin(particles[:, 0])
    ICCy = particles[:, 4] + R * np.cos(particles[:, 0])

    particles[:, 3] = np.cos(wt) * (particles[:, 3] - ICCx) - np.sin(wt) * (particles[:, 4] - ICCy) + ICCx
    particles[:, 4] = np.sin(wt) * (particles[:, 3] - ICCx) + np.cos(wt) * (particles[:, 4] - ICCy) + ICCy

    N = len(particles)
    # update heading
    particles[:, 0] += wt
    # particles[:, 0] %= 2 * np.pi

    # TODO
    # move in the (noisy) commanded direction
    particles[:, 1] = vr + (randn(N) * 5)
    particles[:, 2] = vl + (randn(N) * 5)
    # particles[:, 3] += np.cos(particles[:, 0]) * dist
    # particles[:, 4] += np.sin(particles[:, 0]) * dist


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:3] - landmark[0:3], axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 3:5]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


def run_pf1(N, sensor_std_err=1, do_plot=True, plot_particles=False, xlim=(-10, 40), ylim=(0, 140), initial_x=None,
            vl=None, vr=None, t=None, angle=None, dist=None):
    landmarks = np.array([[0, 10, 10], [0.1, 5, 15], [-0.1, 15, 5], [0.3, 10, 15], [-1, 15, 15]])
    NL = len(landmarks)
    iters = len(t)

    plt.figure()

    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        particles = create_uniform_particles((0, 20), (0, 20), (0, 6.28), N)
    weights = np.zeros(N)

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(particles[:, 3], particles[:, 4],
                    alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0., 0., 13.8])

    x = 0
    y = 0
    for i in range(1, iters):
        # TODO
        # robot_pos += (1, 1)
        dt = t[i] - t[i - 1]

        robot_pos = (angle[i - 1], vl[i - 1], vr[i - 1])

        # landmarks = np.array([robot_pos])
        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos) +
              (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05), dt=dt)

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err,
               landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 2],
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[1], robot_pos[2], marker='',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    # xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    # plt.xlim(*xlim)
    # plt.ylim(*ylim)
    # print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()
