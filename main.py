import math
import csv
import numpy as np
from math import sin, cos

from numpy.random.mtrand import seed

import kalman
import matplotlib.pyplot as plt

import particle

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


def print_plot(plots=None, coords=None, bounded=True, title=None):
    if plots is not None:
        (t_plot, x_plot, y_plot) = plots
    else:
        t_plot = []
        x_plot = []
        y_plot = []
        for tuple in coords:
            t_plot.append(tuple[0])
            x_plot.append(tuple[2])
            y_plot.append(tuple[1])

    def print_p(xlabel, t_plot, y_axis, boundary=None):
        plt.ylabel(xlabel)
        plt.xlabel("t")
        plt.plot(t_plot, y_axis)
        if title is not None:
            plt.title(title)
        if boundary is not None:
            plt.axis(boundary)
        plt.show()

    print_p("x(t)", t_plot, x_plot, [1509976324.240, 1509976340.20860, 0, 140] if bounded else None)
    print_p("y(t)", t_plot, y_plot, [1509976324.240, 1509976340.20860, -10, 40] if bounded else None)


def follow_by_wheels():
    coords = []
    with open('log_robot_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x = init_x
        y = init_y
        angle = init_angle
        t_prev = 0
        is_init = False

        for row in spamreader:
            try:
                t = float(row[0])
                if not is_init:
                    t_prev = t
                    vl = float(row[3]) * magic_coeff
                    vr = float(row[4]) * magic_coeff
                    is_init = True

                dt = t - t_prev
                if abs(vr - vl) < 0.0001:
                    x_next = x + vl * dt * cos(angle)
                    y_next = y + vl * dt * sin(angle)
                    angle_next = angle
                else:
                    R = wheel_base_half * (vl + vr) / (vr - vl)
                    wt = (vr - vl) / (wheel_base_half * 2) * dt

                    ICCx = x - R * sin(angle)
                    ICCy = y + R * cos(angle)

                    x_next = cos(wt) * (x - ICCx) - sin(wt) * (y - ICCy) + ICCx
                    y_next = sin(wt) * (x - ICCx) + cos(wt) * (y - ICCy) + ICCy
                    angle_next = angle + wt

                x = x_next
                y = y_next
                angle = angle_next
                vl = float(row[3]) * magic_coeff
                vr = float(row[4]) * magic_coeff
                t_prev = t

                coords.append((t, -y, x))

            except ValueError:
                pass
    print_plot(coords=coords, title="By wheels")


def follow_by_gyro():
    coords = []
    with open('log_robot_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x = init_x
        y = init_y
        # angle = init_angle
        t_prev = 0
        is_init = False
        for row in spamreader:
            try:
                t = float(row[0])
                angle = float(row[2]) * math.pi / 180
                if not is_init:
                    t_prev = t
                    vl = float(row[3]) * magic_coeff
                    vr = float(row[4]) * magic_coeff
                    is_init = True
                # print(t, d, a, vl, vr, sep=', ')

                dt = t - t_prev
                avg_speed = (vr + vl) / 2
                x_next = x + avg_speed * dt * sin(angle)
                y_next = y + avg_speed * dt * cos(angle)

                x = x_next
                y = y_next
                vl = float(row[3]) * magic_coeff
                vr = float(row[4]) * magic_coeff
                t_prev = t

                coords.append((t, x, y))

            except ValueError:
                pass
    print_plot(coords=coords, title="By gyro")


def print_log_camera():
    t_plot = []
    x_plot = []
    y_plot = []
    with open('log_camera_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        k = False
        for row in spamreader:
            if not k:
                k = True
                continue
            t_plot.append(float(row[0]))
            x_plot.append(float(row[1]))
            y_plot.append(float(row[2]))
    print_plot(plots=(t_plot, x_plot, y_plot), title="From camera")


def print_log_camera_kalman():
    t_plot = []
    x_plot = []
    y_plot = []
    with open('log_camera_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        k = False
        for row in spamreader:
            if not k:
                k = True
                continue
            t_plot.append(float(row[0]))
            x_plot.append(float(row[1]))
            y_plot.append(float(row[2]))

    Q = 0.5
    x_plot = kalman.apply_filter(x_plot, Q, x_cam_noise[1])
    y_plot = kalman.apply_filter(y_plot, Q, y_cam_noise[1])
    print_plot(plots=(t_plot, x_plot, y_plot), title="From camera with Kalman Q=" + str(Q))


def follow_by_gyro_kalman():
    coords = []
    v = []
    t = []
    angle = [0]
    Q = 0.1
    with open('log_robot_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x = init_x
        y = init_y
        is_init = False

        for row in spamreader:
            try:
                t.append(float(row[0]))
                if not is_init:
                    t.append(float(row[0]))
                    v.append((float(row[4]) + float(row[3])) * magic_coeff / 2)
                    is_init = True
                angle.append(float(row[2]) * math.pi / 180)
                v.append((float(row[4]) + float(row[3])) * magic_coeff / 2)
            except ValueError:
                pass
    angle = kalman.apply_filter(angle, Q=Q, R=gyro_noise[1])

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x_next = x + v[i - 1] * dt * sin(angle[i])
        y_next = y + v[i - 1] * dt * cos(angle[i])

        x = x_next
        y = y_next

        coords.append((t[i], x, y))

    print_plot(coords=coords, title="By gyro with Kalman, Q=" + str(Q))


def sensor_fusion():
    coords_gyro = []
    coords_wheels = []
    vl = []
    vr = []
    t = []
    angle = [0]
    Q = 0.1

    with open('log_robot_2.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        x = init_x
        y = init_y
        is_init = False

        for row in spamreader:
            try:
                t.append(float(row[0]))
                if not is_init:
                    t.append(float(row[0]))
                    vl.append((float(row[3])) * magic_coeff)
                    vr.append((float(row[4])) * magic_coeff)
                    is_init = True
                angle.append(float(row[2]) * math.pi / 180)
                vl.append((float(row[3])) * magic_coeff)
                vr.append((float(row[4])) * magic_coeff)
            except ValueError:
                pass

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        avg_speed = (vl[i - 1] + vr[i - 1]) / 2
        x_next = x + avg_speed * dt * sin(angle[i])
        y_next = y + avg_speed * dt * cos(angle[i])

        x = x_next
        y = y_next

        coords_gyro.append((t[i], x, y))

    a = init_angle
    x = init_x
    y = init_y
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if abs(vr[i - 1] - vl[i - 1]) < 0.0001:
            x_next = x + vl[i - 1] * dt * cos(a)
            y_next = y + vl[i - 1] * dt * sin(a)
            angle_next = a
        else:
            R = wheel_base_half * (vl[i - 1] + vr[i - 1]) / (vr[i - 1] - vl[i - 1])
            wt = (vr[i - 1] - vl[i - 1]) / (wheel_base_half * 2) * dt

            ICCx = x - R * sin(a)
            ICCy = y + R * cos(a)

            x_next = cos(wt) * (x - ICCx) - sin(wt) * (y - ICCy) + ICCx
            y_next = sin(wt) * (x - ICCx) + cos(wt) * (y - ICCy) + ICCy
            angle_next = a + wt

        x = x_next
        y = y_next
        a = angle_next
        coords_wheels.append((t[i], -y, x))

    x_w = [0]
    x_g = [0]
    for i in range(0, len(coords_gyro)):
        x_w.append(coords_wheels[i][1])
        x_g.append(coords_gyro[i][1])
    x_matrix = np.matrix([x_w, x_g]).transpose()
    Q = 0.5
    R = np.matrix([[gyro_noise[1], 0], [0, 100]]).transpose()

    y_w = [0]
    y_g = [0]
    for i in range(0, len(coords_gyro)):
        y_w.append(coords_wheels[i][2])
        y_g.append(coords_gyro[i][2])
    y_matrix = np.matrix([y_w, y_g]).transpose()

    x_kalman = kalman.apply_filter(x_matrix, Q, R, (len(x_w),)).tolist()
    y_kalman = kalman.apply_filter(y_matrix, Q, R, (len(y_w),)).tolist()
    print_plot(plots=(t, y_kalman, x_kalman), title="Kalman with 2 sensors")


if __name__ == '__main__':
    # follow_by_wheels()
    # follow_by_gyro_kalman()
    # follow_by_gyro()
    # print_log_camera()
    # print_log_camera_kalman()

    # sensor_fusion()
    seed(2)
    particle.run_pf1(N=50000, plot_particles=False)

    # main2()
