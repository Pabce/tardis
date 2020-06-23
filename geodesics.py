"""
File: geodesics.py

Contains the definition of the metric, the Christoffel symbols, the geodesic equation and wrappers for integrating and
plotting one or more curves at the same time. If run directly, it will execute whatever code is below
'if name==__main__'. If called from another file, you may call the functions: 'once', 'sweep' or 'ctc'.
"""

import numpy as np
import matplotlib.pyplot as plt
import rungekutta as rk
import pprint
import time

from numba import jit

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


def correction_function(t, x, h, ht, hx):
    arg1 = (x*x-t*t*(2*h-1) + 20)/t*t
    arg2 = arg1 - 40/t*t
    W = 0.5*(np.tanh(arg1) - np.tanh(arg2))

    # To avoid overflow error in the cosh function, we simply set these derivatives to 0 if coordinates are sufficiently
    # far away from the bubble boundary.
    if abs(arg1) > 30 or abs(arg2) > 30 or 1==1:
        W = 0
        Wt = 0
        Wx = 0
    else:
        sech1 = 1/np.cosh(arg1)**2
        sech2 = 1/np.cosh(arg2)**2

        Wx = (sech1 - sech2) * 1/t*t * (x - t*t*hx)
        Wt = -((t ** 3 * ht + x * x + 20) * sech1 + (t ** 3 * ht + x * x - 20) * sech2) / t ** 3
        print('asdf')

    return W, Wt, Wx


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


def connection_coeffs_correc(t, x):
    h, ht, hx = hat_function(t, x)

    x2, x3, x4 = x*x, x*x*x, x*x*x*x
    t2, t3, t4 = t*t, t*t*t, t*t*t*t
    h2 = h*h
    t2masx2 = t2 + x2
    t2masx2_2 = t2masx2*t2masx2

    W, Wt, Wx = correction_function(t,x,h,ht,hx)
    W2 = W*W

    denom = t2masx2_2 * (t2masx2 - 4*t2*h + 4*h2*(t2 + 2*t4*x*W*t + t3*t3*t2masx2*W2))

    # t*(4*(t2*x2 + x4)*h2 - t* t2masx2_2 *ht + 2*t2masx2*h*(-x2-t2*x*hx + (t3 + 2*t*x2)*ht))
    gamma_000 = t*((-t2-x2+2*t2*h)*(2*x2*h + t*t2masx2*ht) + 2*h*(x+t2*t2masx2*W)*t * t2masx2
                * (-t*hx + 2*(x+t2*t2masx2*W) * ht) + 2*h*x2 + 3*t2*t2masx2_2*W + t3*t2masx2_2*Wt) \
                / denom

    gamma_001 = -t2*(4*x*h2*(t2masx2+t2*x*t2masx2*W) + t2masx2*hx + 2*t2masx2*h*(-x-t2*hx + t*(x+t2*t2masx2*W)*ht)) \
                /denom

    gamma_011 = t*((2*t2*h2* (t2masx2 + t2*x*t2masx2*W +t2*t2masx2*Wx) - t2masx2_2 * (x+t2*t2masx2 *W)*hx
                + t*ht) + 2*t2masx2*h*(-t2+ t2*(x+t2*t2masx2*W)*h - t2*t2masx2_2*Wx + t3 *ht)) \
                / denom

    gamma_100 = 2*t2*h * ((x + t2*t2masx2*W) * (2*x2*h + t *t2masx2 *ht) - (-t2masx2 + 2*t2*h)*t*t2masx2
                * (t*hx + 2*(x + t2*t2masx2)*W)*ht + 2*h*x2 + 3*t2*t2masx2_2*W + t3*t2masx2_2*Wt) \
                /denom

    gamma_101 = t*(-4*t2*x*h2 * t2*t2masx2*W - t*t2masx2_2*ht + 2*t2masx2*h*(-x2 + t2*(x + t2*t2masx2*W) + t3*ht)) \
                /denom

    gamma_111 = t2*((-t2*x2*2*t2*h) * (-2*x*h + t2masx2 * hx) + 2*h*(x+t2*x2 + t2)*W*(2*h*(x2 + t2*t2masx2_2 * Wx) + \
                t2masx2* (x + t2*t2masx2*W)*hx + t*ht)) \
                /denom

    return gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111


@jit(nopython=True)
def connection_coeffs_symbolic(t, x):
    gamma_000 = -t * (x ** 2 * (
                2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (t ** 2 + x ** 2) * (
                    4.0 * ALPHA * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                        -A ** 2 + t ** 2 + x ** 2) + 2 * tanh(
                ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (
                                  tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) - (
                                  2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (
                                      t ** 2 + x ** 2) * (4.0 * ALPHA * t ** 2 * (
                                      tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                                      -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
                              ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (-t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2)) / (
                            2 * (t ** 2 + x ** 2) * (t ** 2 * x ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                                                                 -1.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (
                                                                     -A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))
    gamma_001 = t ** 2 * x * ((2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (
                t ** 2 + x ** 2) * (4.0 * ALPHA * t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
        ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (
                                          tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + (
                                          -t ** 2 * (tanh(
                                      ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2) * (
                                          -4.0 * ALPHA * (t ** 2 + x ** 2) * (
                                              tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                      -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
                                      ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) / (
                            2 * (t ** 2 + x ** 2) * (t ** 2 * x ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                                                                 -1.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (
                                                                     -A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))
    gamma_011 = t * (t ** 2 * x ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) * (
                -4.0 * ALPHA * (t ** 2 + x ** 2) * (
                    tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                            -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
            ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0) + (-t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2) * (2.0 * t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + 4 * x ** 2 * (tanh(
        ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) - (t ** 2 + x ** 2) * (4.0 * ALPHA * t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                                                                       -A ** 2 + t ** 2 + x ** 2) + 8 * ALPHA * x ** 2 * (
                                                                                                       tanh(ALPHA * (
                                                                                                                   R ** 4 - (
                                                                                                                       -A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                                                                       -A ** 2 + t ** 2 + x ** 2) + 4.0 * tanh(
        ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 4.0))) / (2 * (t ** 2 + x ** 2) * (
                t ** 2 * x ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                    -1.0 * t ** 2 * (
                        tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))
    gamma_100 = -x * (t ** 2 * (
                2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (t ** 2 + x ** 2) * (
                    4.0 * ALPHA * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                        -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
                ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (
                                  tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + (
                                  2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (
                                      t ** 2 + x ** 2) * (4.0 * ALPHA * t ** 2 * (
                                      tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                                      -A ** 2 + t ** 2 + x ** 2) + 2 * tanh(
                              ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (-t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2)) / (
                            2 * (t ** 2 + x ** 2) * (t ** 2 * x ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                                                                 -1.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (
                                                                     -A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))
    gamma_101 = -t * (t ** 2 * x ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) * (
                -4.0 * ALPHA * (t ** 2 + x ** 2) * (
                    tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                            -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
            ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0) - (
                                  2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) - (
                                      t ** 2 + x ** 2) * (4.0 * ALPHA * t ** 2 * (
                                      tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                                      -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
                              ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) * (-t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2)) / (
                            2 * (t ** 2 + x ** 2) * (t ** 2 * x ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                                                                 -1.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (
                                                                     -A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))
    gamma_111 = -t ** 2 * x * ((tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) * (
                2.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + 4 * x ** 2 * (
                    tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) - (t ** 2 + x ** 2) * (
                            4.0 * ALPHA * t ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                        -A ** 2 + t ** 2 + x ** 2) + 8 * ALPHA * x ** 2 * (
                                        tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                        -A ** 2 + t ** 2 + x ** 2) + 4.0 * tanh(
                        ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 4.0)) - (-t ** 2 * (
                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) + t ** 2 + x ** 2) * (
                                           -4.0 * ALPHA * (t ** 2 + x ** 2) * (
                                               tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) ** 2 - 1) * (
                                                       -A ** 2 + t ** 2 + x ** 2) + 2.0 * tanh(
                                       ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 2.0)) / (
                            2 * (t ** 2 + x ** 2) * (t ** 2 * x ** 2 * (
                                tanh(ALPHA * (R ** 4 - (-A ** 2 + t ** 2 + x ** 2) ** 2)) + 1.0) ** 2 + (
                                                                 -1.0 * t ** 2 * (tanh(ALPHA * (R ** 4 - (
                                                                     -A ** 2 + t ** 2 + x ** 2) ** 2)) + 1) + t ** 2 + x ** 2) ** 2))

    return np.array([gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111])


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

    # cgamma_000, cgamma_001, cgamma_011, cgamma_100, cgamma_101, cgamma_111 = connection_coeffs(t, x)
    gamma_000, gamma_001, gamma_011, gamma_100, gamma_101, gamma_111 = connection_coeffs(t, x)

    #print(cgamma_000 - gamma_000, cgamma_001 - gamma_001, cgamma_011 - gamma_011, cgamma_100 - gamma_100, cgamma_101 - gamma_101, cgamma_111 - gamma_111)

    # print(param_correction(lam), 'ads')
    # print(-(gamma_000*u0**2 + 2*gamma_001*u1*u0 + gamma_011 * u1**2))
    #a0, a1 = tools.get_normalized_perpendicular(u0, u1, t, x, sign="minus")
    #print(u0, u1, a0, a1)
    #print(gtt(t,x)*a0*u0 + gxt(t,x)*a0*u1+ gxt(t,x)*a1*u0+gxx(t,x)*a1*u1)
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

# TODO: change x for lambda to reduce confusion (LAM HERE IS X IN RK4!!!) (HERE X IS THE COORDINATE)
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
    #mpt.plot_light_time(14, 6, param_direction=True, tangents=False, trajectory=y, param=lam)

    # Secondary figures:
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


# TODO: Only one colorbar to appear if you plot different ranges. Option to plot "backwards and forwards".
#  Colorcode according to origin/ending. Ability to plot to a desired color. Ability to plot only certain types.
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
        plt.figure(2)
        #plt.plot(lam, [modulus]*len(y[0,:]), color='red')
        #plt.plot(lam, tools.get_squared_mod(*y)*np.sin(tools.get_squared_mod(*y)*10**15)*1.01 -1)
        #plt.plot(lam, tools.get_squared_mod(*y))
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel(r'Affine parameter $(\lambda)$', fontsize=16)
        plt.ylabel(r'Local speed', fontsize=16)
        plt.plot(lam, tools.get_3speed_on_trajectory(y))



def once():
    # INITIAL CONDITIONS
    lam0 = 10
    mod = -1

    # -100, -50 (u1_0 = -1, plus) nice
    # -150, -20 problematic
    # -150, -200 REALLY Problematic
    # CAREFUL! REVIEW INITIAL CONDITIONS!!! (also for non-null...)
    t0, x0 = 80, 80
    u1_0 = -1
    u0_0, u1_0 = tools.get_initial_cond(t0, x0, 0, u1=-1, sign="plus")
    y0 = [u0_0, u1_0, t0, x0]


    t1, x1, u1_1 = -150, -35, 0
    u0_1, u1_1 = tools.get_initial_cond(t1, x1, -1, three_velocity=0.64, sign="minus")
    #print(u1_1/u0_1)
    y1 = [u0_1, u1_1, t1, x1]

    print(y1)

    start_time = time.time()

    y2, true_y, lam2, step2, err2 = rk.rk4(fun_rk, y1, x0=lam0, num=18000, h_start=0.02, h_max=10**1,
                                   h_min=10 ** -7, h_max_change=1.5, acc=10 ** -9, experimental=False, cutoff=True)
    print('Smallest step taken was ', np.min(step2))

    # tend = 300.0; dt0 = 10e-10; atol = 1e-11; rtol = 1e-11
    # lamout, yout, info = pyode(fun_pyode, None, y0, lam0, tend, dt0, atol, rtol, method='bs', nsteps=10000)

    print("--- %s seconds ---" % (time.time() - start_time))
    pprint.pprint(y2[:, -1])

    fig, ax = plt.subplots(figsize=(11, 9))
    plot_stuff(true_y, lam2, step2, err2, fig, ax)
    # mpt = plotting.Plotter(fig, ax, A, R, ALPHA)
    # mpt.plot_light_time(15, True, True, radius=100)

    # plt.figure(10)
    # plt.plot(lam2, y2[0,:])

    # plt.figure(9)
    # plt.plot(y2[0,:])



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
    #ctc()
    #once()
    #sweep((-150, -150), (-160, 170), 4, 1003, -1, three_velocity=0.999)
    #sweep((0, 0), (80, 115), 5, 6000, -1, u1=0)

    #CTCs
    #sweep((0,0), (100.35,121), 1, 5000, -1, u1=0, nature="ctc")

    fig, ax = plt.subplots(figsize=(11, 9))
    aplotter = plotting.Plotter(fig, ax, A, R, ALPHA)

    aplotter.plot_bubble()

    # Lighcone bullshit
    # s = 5
    # aplotter.plot_light_time(4, s, lightcones=True, time_arrows=True, radius=12)
    # aplotter.plot_light_time(9, s, lightcones=True, time_arrows=True, radius=30)
    # aplotter.plot_light_time(11, s, lightcones=True, time_arrows=True, radius=50)
    # aplotter.plot_light_time(13, s, lightcones=True, time_arrows=True, radius=70)
    # aplotter.plot_light_time(15, s, lightcones=True, time_arrows=True, radius=90)
    # aplotter.plot_light_time(17, s, lightcones=True, time_arrows=True, radius=110)
    # aplotter.plot_light_time(19, s, lightcones=True, time_arrows=True, radius=122.3)
    # aplotter.plot_light_time(21, s, lightcones=True, time_arrows=True, radius=140)
    # aplotter.plot_light_time(23, s, lightcones=True, time_arrows=True, radius=160)
    # aplotter.plot_light_time(25, s, lightcones=True, time_arrows=True, radius=180)
    # aplotter.plot_light_time(27, s, lightcones=True, time_arrows=True, radius=200)

    # for i in range(200):
    #     aplotter.plot_lightcone(0, -110.12 - i*0.2, 0.1)
    #     aplotter.plot_time_arrow(0, -110.12 - i * 0.2, 0.1)

    # aplotter.plot_light_time(1397, 0.15, lightcones=True, time_arrows=True, radius=121)
    # aplotter.plot_light_time(1397, 0.15, lightcones=True, time_arrows=True, radius=122.066)
    # aplotter.plot_light_time(1397, 0.15, lightcones=True, time_arrows=True, radius=123)

    # Null sweeping
    sweep((-150, -150), (-450, 145), 60, 6000, 0, u1=1, plotter=aplotter, sign='minus')
    # sweep((-150, -150), (-145, 450), 65, 6000, 0, u1=-1, sign='minus', plotter=aplotter)
    #sweep((-120, +170), (297, -154), 57, 6000, 0, u1=1, plotter=aplotter, sign='plus')
    #sweep((+150, +150), (-150, 440), 70, 6000, 0, u1=-1, plotter=aplotter, sign='plus')

    #sweep(0, 0, 200, 2500, 0, u1=1, radius=123, angle_range=(np.pi/2, np.pi))

    plt.show()
