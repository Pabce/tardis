"""
File: rungekutta.py

Custom implementation of the classic rk4 algorithm. Contains the 'rk4' integration function, which takes as arguments:
a function vector, initial conditions, number of integration steps, constraints on the step size, truncation error
tolerance and the 'experimental' and 'cutoff' parameters. Returns a solution numpy matrix plus information about the
integration process.
"""
import numpy as np

LIMT = (-152, 152)
LIMX = (-152, 152)

np.seterr(over='raise')
# TODO: IMPORTANT FOR LIGHTCONES! introduce 'limits' in the method, e.g. to stop when certain coordinate is reached, when the vector differs from the modulus, etc.
# TODO: Maybe modify tolerances? Absolute error in coords (t,x) BUT modulus vs null vector in u0, u1 tangent vectors. Or a relative comparison. !!!!


def rk4(f, y0, x0, num, h_start, h_max, h_min, h_max_change, acc, experimental=False, cutoff=False):
    variable_num = len(y0)
    leno = num
    y_arr = np.zeros((variable_num, leno))

    # Here the true tangent vectors will be stored, if experimental mode is on
    true_y_arr = np.zeros((variable_num, leno))
    # We will only check for cutoff if the curve has first entered the valid range
    check_cutoff = False

    x_arr = np.zeros(leno)
    h_arr = np.zeros(leno)
    err_arr = np.zeros(leno)

    x = x0
    y = y0
    x_arr[0] = x0
    y_arr[:, 0] = y0
    true_y_arr[:, 0] = y0
    h = h_start
    h_arr[0] = h_start

    i = 1
    min_warning = 0
    corrections = 0
    while i < num:
        y1, fxy = rk4_step(f, y, x, h)
        y2, fxy2 = rk4_step(f, y, x, h/2, fxy=fxy)
        y3, fxy3 = rk4_step(f, y2, x + h/2, h/2)
        corrected = False
        # EXPERIMENTAL: Reduce tangent vectors to avoid explosion of numerical noise and truncation error.
        # For plotting trajectories this works great, often extending the geodesic past a "conflictive" point with very
        # little error. For some reason, using relative errors doesn't work...
        if experimental:
            if np.abs(y1[0]) > 5 or np.abs(y1[1]) > 5:
                y1[0] /= 5
                y1[1] /= 5
                y3[0] /= 5
                y3[1] /= 5

                corrected = True

        #rel = max(y3[0], y3[1], y1[0], y1[1])
        truncation_err = np.max(np.abs(y3-y1))

        if truncation_err == 0:
            truncation_err = 10**-15

        if truncation_err <= acc:
            y = y3
            x += h

            x_arr[i] = x
            y_arr[:, i] = y

            # True tangent vectors!
            true_y_arr[:, i] = y3
            if corrected:
                corrections += 1

            true_y_arr[0, i] *= 5**corrections
            true_y_arr[1, i] *= 5**corrections


            h_arr[i] = h
            err_arr[i] = truncation_err

            i += 1
            min_warning = 0

        # if (np.abs(true_y_arr[:, i-1] - y_arr[:, i-1]) > 10e-10).any():
        #     print("ASDJALSKJDÑLKJ")
        #     print(true_y_arr[:, i-1], y_arr[:, i-1])

        # else:
        #     print('------------')
        #     print(y1)
        #     print(y3)
        #     print(i, np.abs(y3-y1))

        h_new = h * (acc/truncation_err)**0.2


        # Check and enforce limits on new step
        if h_new/h > h_max_change:
            h_new = h*h_max_change
        elif h_new/h < 1/h_max_change:
            h_new = h/h_max_change

        if h_new > h_max:
            h_new = h_max
        if h_new < h_min:
            if min_warning == 1:
                print('h = h_min did not make the cut')
                print('Failed at step {}'.format(i))
                return y_arr[:, 0:i], true_y_arr[:, 0:i], x_arr[0:i], h_arr[0:i], err_arr[0:i]
            h_new = h_min
            min_warning = 1

        h = h_new

        # If cutoff is enabled, stop integration if curve has exited limits
        if y[2] < LIMT[0] or y[2] > LIMT[1]: #or y[3] < LIMX[0] or y[3] > LIMX[1]:
            if cutoff and check_cutoff:
                return y_arr[:, 0:i], true_y_arr[:, 0:i], x_arr[0:i], h_arr[0:i], err_arr[0:i]
        else:
            check_cutoff = True



    return y_arr, true_y_arr, x_arr, h_arr, err_arr


def rk4_step(f, y, x, h, fxy=None):
    fxy = f(x,y) if fxy is None else fxy

    k1 = h * fxy
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + 1 * h, y + 1 * k3)

    return y + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4, fxy

