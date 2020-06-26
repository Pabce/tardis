"""
File: geodesics.py

Contains the definition of the metric, the Christoffel symbols, the geodesic equation and wrappers for integrating and
plotting one or more curves at the same time. If run directly, it will execute whatever code is below
'if name==__main__'. If called from another file, you may call the functions: 'once', 'sweep' or 'ctc'.
"""

import numpy as np
import matplotlib.pyplot as plt
import pprint
import time

from numba import jit

import rungekutta as rk
import plotting
import tools

np.seterr(over='raise')

# global consts (for now)
R = 70
A = 100
ALPHA = 1/6e6
tanh = np.tanh
# auxiliary constants
R4 = R**4
A2 = A**2

# We are doing this for the x-t plane only
def bubble(t, x):
    traj = (x*x + t*t - A**2)
    return R**4 - traj*traj


def h(t, x):
    return 0.5 + np.tanh(ALPHA * bubble(t, x))*0.5

# Derivatives of the h 'hat' function
def hat_function(t, x):
    traj = (x * x + t * t - A2)
    bubble = R4 - traj*traj

    h = 0.5 + np.tanh(ALPHA * bubble)*0.5

    try:
        h_subprime = 0.5/(np.cosh(ALPHA * bubble))**2 * ALPHA
    except FloatingPointError:
        h_subprime = 0

    ht = h_subprime * -4*t*traj
    hx = h_subprime * -4*x*traj

    return h, ht, hx


# metric (on the x-t plane)
def gxx(t, x): return 1 - h(t, x)*(2*t**2)/(x**2+t**2)
def gtt(t, x): return -gxx(t, x)
def gtx(t, x): return h(t, x)*2*x*t/(x**2+t**2)
def gxt(t, x): return gtx(t, x)
# (these are useless)
def gyy(t, x): return 1
def gzz(t, x): return 1


# connection coeffs (only the non-zero ones for the x-t plane)
def connection_coeffs(t, x):
    h, ht, hx = hat_function(t, x)

    x2, x3, x4 = x*x, x*x*x, x*x*x*x
    t2, t3, t4 = t*t, t*t*t, t*t*t*t
    h2 = h*h
    t2masx2 = t2 + x2
    t2masx2_2 = t2masx2*t2masx2

    denom = t2masx2_2 * (t2masx2 - 4*t2*h + 4*t2*h2)
    if denom == 0:
        print('HOL UP')
    #print(denom, h)

    gamma_000 = t*(4*(t2*x2 + x4)*h2 - t* t2masx2_2 *ht + 2*t2masx2*h*(-x2-t2*x*hx + (t3 + 2*t*x2)*ht)) \
                / denom

    gamma_001 = t2 * (-4*(t2*x + x3)*h2 - t2masx2_2 * hx + 2* t2masx2 * h *(t2*hx + x*(1-t*ht))) \
                / denom

    gamma_011 = t*(4*t2 * t2masx2 * h2 - t2masx2_2 * (2*x*hx + t*ht) +
                2* t2masx2 *h*(-t2 + t2*x*hx + t3*ht)) \
                / denom

    gamma_100 = (-t* t2masx2_2 * (t*hx - 2*x*ht) + 2* t2masx2 * h *(t4*hx + x*(x2 - t3*ht))) \
                / denom

    gamma_101 = t*(-t* t2masx2_2 * ht + 2 * t2masx2 * h *(-x2 + t2*x*hx + t3*ht)) \
                / denom

    gamma_111 = t2*(- t2masx2_2 * hx + 2 * t2masx2 * h *((t2+2*x2)*hx + x*(1 + t*ht))) \
                / denom

    return gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111


@jit(nopython=True)
def connection_coeffs_numba(t, x):
    traj = (x * x + t * t - A2)
    bubble = R4 - traj*traj

    h = 0.5 + np.tanh(ALPHA * bubble)*0.5

    coship = np.cosh(ALPHA * bubble)
    if coship > 10**15:
        h_subprime = 0
    else:
        h_subprime = 0.5/coship**2 * ALPHA

    ht = h_subprime * -4*t*traj
    hx = h_subprime * -4*x*traj

    x2, x3, x4 = x*x, x*x*x, x*x*x*x
    t2, t3, t4 = t*t, t*t*t, t*t*t*t
    h2 = h*h
    t2masx2 = t2 + x2
    t2masx2_2 = t2masx2*t2masx2

    denom = t2masx2_2 * (t2masx2 - 4*t2*h + 4*t2*h2)

    gamma_000 = t*(4*(t2*x2 + x4)*h2 - t* t2masx2_2 *ht + 2*t2masx2*h*(-x2-t2*x*hx + (t3 + 2*t*x2)*ht))

    gamma_001 = t2 * (-4*(t2*x + x3)*h2 - t2masx2_2 * hx + 2* t2masx2 * h *(t2*hx + x*(1-t*ht)))

    gamma_011 = t*(4*t2 * t2masx2 * h2 - t2masx2_2 * (2*x*hx + t*ht) +
                2* t2masx2 *h*(-t2 + t2*x*hx + t3*ht))

    gamma_100 = (-t* t2masx2_2 * (t*hx - 2*x*ht) + 2* t2masx2 * h *(t4*hx + x*(x2 - t3*ht)))

    gamma_101 = t*(-t* t2masx2_2 * ht + 2 * t2masx2 * h *(-x2 + t2*x*hx + t3*ht))

    gamma_111 = t2*(- t2masx2_2 * hx + 2 * t2masx2 * h *((t2+2*x2)*hx + x*(1 + t*ht)))

    return np.array([gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111])/denom


def fun_rk(lam, y):
    u0, u1, t, x = y
    
    gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111 = connection_coeffs(t, x)

    u02 = u0 * u0
    u12 = u1 * u1

    f1 = -(gamma_000*u02 + 2*gamma_001*u1*u0 + gamma_011 * u12)  # + u0*param_correction(lam)

    f2 = -(gamma_100*u02 + 2*gamma_101*u1*u0 + gamma_111 * u12)  # + u1*param_correction(lam)

    f3 = u0

    f4 = u1

    return np.array([f1, f2, f3, f4])


def fun_rk_ctc(lam, y):
    u0, u1, t, x = y
    a0, a1 = tools.get_normalized_perpendicular(u0, u1, t, x, sign="minus")

    f1, f2, f3, f4 = fun_rk(lam, y)

    f1 -= a0/(t**2+x**2)**0.5
    f2 -= a1/(t**2+x**2)**0.5

    return np.array([f1, f2, f3, f4])


# A correction to the geodesic eq. introduced when using a non-affine parameter
def param_correction(lam):
    # alpha = cos(lam) + lam**2

    dalpha = 5*lam
    d2alpha = 0

    return -d2alpha * dalpha**(-2)


def plot_stuff(y, lam, step, err, figure, axis):
    fig, ax = figure, axis
    mpt = plotting.Plotter(fig, ax, A, R, ALPHA)

    # Plot circles defining the bubble boundary
    mpt.plot_bubble()

    # Plot trajectory and related stuff
    mpt.plot_trajectory(y, lam, step, err, colormap="step_size", solid=False, width=3)

    # Time arrows, light cones, etc (uncomment as needed). FOR PLOTTING EQUISPACED ARROWS, TURN EXPERIMENTAL OFF!
    mpt.plot_light_time(15, 6, lightcones=True, time_arrows=True, trajectory=y, param=lam)
    mpt.plot_light_time(14, 6, param_direction=True, tangents=False, trajectory=y, param=lam)

    # Secondary figures, uncomment as needed:
    # plt.figure(4)
    # plt.plot(y[0,:])
    # plt.plot(lam, get_initial_cond(y[0,:], y[2,:], y[3,:])/y[1,:], color='red')
    # plt.plot(lam, np.ones(len(y[0,:])), linestyle='dashed')
    # plt.plot(y[1,:])
    # plt.plot(lam, get_modulus(y[1, :], get_initial_cond(y[1, :], y[2, :], y[3, :], -1), y[2, :], y[3, :]), color='red')  # this is the modulus of the initial tangent vector (?)
    # plt.plot(lam, get_modulus(*y))

    # plt.figure(2)
    # plt.plot(lam, y[3,:], color='orange')
    # plt.plot(lam, y[2,:], color='blue')

    #plt.figure(3)
    #plt.plot(lam, step)

    
# Yow may specify a t and x range or a radius and an angle range. Also, either u1 o three_velocity is required
# Plotter is a plotter object (from plotting.py)
def sweep(t_range, x_range, steps, num, modulus, u1=np.inf, three_velocity=np.inf, radius=None, angle_range=None,
          sign="minus", nature="geodesic", plotter=None):
    # The initial parameter is chosen to be != 0 to avoid strange integration errors due to very small numbers.
    lam0 = 10

    div = steps - 1 if steps != 1 else 1
    if not radius:
        t_step = (t_range[1] - t_range[0])/div
        x_step = (x_range[1] - x_range[0])/div
    else:
        angle_step = (angle_range[1] - angle_range[0])/div

    if plotter:
        fig, ax = plotter.fig, plotter.ax
    else:
        fig, ax = plt.subplots(figsize=(11, 9))

    fun = fun_rk if nature=="geodesic" else fun_rk_ctc

    # Store all the integrations
    solutions = []
    for i in range(steps):
        print(i)
        if not radius:
            t0 = t_range[0] + t_step * i
            x0 = x_range[0] + x_step * i
        else:
            t0 = radius * np.sin(angle_range[0] + angle_step * i)
            x0 = radius * np.cos(angle_range[0] + angle_step * i)

        if u1 != np.inf:
            u0_0, u1_0 = tools.get_initial_cond(t0, x0, modulus, u1=u1, sign=sign)
        else:
            u0_0, u1_0 = tools.get_initial_cond(t0, x0, modulus, three_velocity=three_velocity, sign=sign)
        y0 = [u0_0, u1_0, t0, x0]

        y, true_y, lam, step, err = rk.rk4(fun, y0, x0=lam0, num=num, h_start=0.02, h_max=10 ** 1, h_min=10 ** -10,
                                       h_max_change=1.5, acc=10 ** -9, experimental=True, cutoff=False)

        solutions.append((y, true_y, lam, step, err ))

    # PLOTTING
    if plotter:
        mpt = plotter
    else:
        mpt = plotting.Plotter(fig, ax, A, R, ALPHA)

    # Plot circles defining the bubble boundary
    if not mpt.has_bubble:
        mpt.plot_bubble()

    # Plot trajectory and related stuff #colormap=[0,0,1,1,1]
    for sol in solutions:
        y, true_y, lam, step, err = sol

        mpt.plot_trajectory(true_y, lam, step, err, colormap=[0,1,1,0,1], limits=[0,4], colorbar="once", solid=True, width=2, nature=nature)


def once():
    # INITIAL CONDITIONS
    lam0 = 10
    mod = -1

    start_time = time.time()

    y2, true_y, lam2, step2, err2 = rk.rk4(fun_rk, y0, x0=lam0, num=18000, h_start=0.02, h_max=10**1,
                                   h_min=10 ** -7, h_max_change=1.5, acc=10 ** -9, experimental=False, cutoff=True)
    print('Smallest step taken was ', np.min(step2))


    print("--- %s seconds ---" % (time.time() - start_time))
    pprint.pprint(y2[:, -1])

    fig, ax = plt.subplots(figsize=(11, 9))
    plot_stuff(true_y, lam2, step2, err2, fig, ax)


def ctc():
    # INITIAL CONDITIONS
    lam0 = 10
    mod = -1

    t0, x0, u1_0 = 0, 80, 0
    u0_0, u1_0 = tools.get_initial_cond(t0, x0, -1, u1=u1_0, sign="minus")
    #print(u1_1/u0_1)
    y0 = [u0_0, u1_0, t0, x0]

    y, true_y, lam, step, err = rk.rk4(fun_rk_ctc, y0, x0=lam0, num=4000, h_start=0.02, h_max=10**1,
                                   h_min=10 ** -7, h_max_change=1.5, acc=10 ** -9, experimental=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    # PLOTTING
    mpt = plotting.Plotter(fig, ax, A, R, ALPHA)

    # Plot circles defining the bubble boundary
    mpt.plot_bubble()

    # Plot trajectory and related stuff
    mpt.plot_trajectory(y, lam, step, err, colormap="local_speed",
                        colorbar="plot", solid=True, nature="ctc", width=5)


if __name__ == '__main__':
    #CTCs
    #sweep((0,0), (80,121), 5, 5000, -1, u1=0, nature="ctc")

    fig, ax = plt.subplots(figsize=(11, 9))
    aplotter = plotting.Plotter(fig, ax, A, R, ALPHA)

    aplotter.plot_bubble()

    # Null sweeping
    sweep((-150, -150), (-450, 145), 60, 6000, 0, u1=1, plotter=aplotter, sign='minus')
    # sweep((-150, -150), (-145, 450), 65, 6000, 0, u1=-1, sign='minus', plotter=aplotter)
    #sweep((-120, +170), (297, -154), 57, 6000, 0, u1=1, plotter=aplotter, sign='plus')
    #sweep((+150, +150), (-150, 440), 70, 6000, 0, u1=-1, plotter=aplotter, sign='plus')

    plt.show()
