import numpy as np
from geodesics import gxx, gtx, gtt, gxt, h

# The modulus of 4-velocity is 0 if the path is null, -1 if it is timelike.
# (We have simply solved the modulus equation for u0).
# For null geodesics, the value we choose for u1 is irrelevant. The value of 3-velocity (as seen by external obervers)
# can be given for timelike geodesics instead of u1.
def get_initial_cond(t, x, modulus, u1="N", three_velocity="N", sign="default"):
    s = -1
    if sign == "plus":
        s = +1

    if three_velocity != "N":
        if modulus == 0:
            print("WARNING: 3-velocity is not a valid initial condition for null trajectories!")
        v = three_velocity
        u0 = -s*(-(gtt(t, x) + 2*gtx(t, x)*v + gxx(t,x)*v**2))**-0.5
        u1 = u0*v
    elif u1 != "N":
        a = gtt(t, x)
        b = 2 * gtx(t, x) * u1
        c = gxx(t, x) * u1 ** 2 - modulus

        u0 = (-b + s * (b*b - 4*a*c)**0.5)/(2*a)

    return u0, u1


def get_squared_mod(u0, u1, t, x):
    return gtt(t,x)*u0**2 + 2*gtx(t,x)*u0*u1 + gxx(t,x)*u1**2


def get_normalized_perpendicular(v0, v1, t, x, sign="minus"):
    # We fix
    if abs(v1) > abs(v0):
        a0 = 1 if sign=="plus" else -1
        # Solving the equation for a1 yields
        a1 = -a0* (gtt(t, x)*v0 + gtx(t,x)*v1)/(gxt(t, x)*v0 + gxx(t, x)*v1)
    else:
        a1 = 1 if sign=="plus" else -1
        # Solving the equation for a0 yields
        a0 = -a1* (gxt(t, x)*v0 + gxx(t, x)*v1)/(gtt(t, x)*v0 + gtx(t,x)*v1)

    # We normalize
    norm2 = get_squared_mod(a0, a1, t, x)

    a0 /= (abs(norm2))**0.5
    a1 /= (abs(norm2))**0.5

    # Chosing the correct sign for consistency
    ang = np.arctan2(t, x)
    if 0 < ang < np.pi/2:
        s0, s1 = -1, -1
    elif np.pi/2 <= ang < np.pi:
        s0, s1 = -1, 1
    elif -np.pi/2 <= ang <= 0:
        s0, s1 = 1, -1
    else:
        s0, s1 = 1, 1

    return abs(a0)*s0, abs(a1)*s1


def in_bubble(t, x):
    if h(t, x) > 0.5:
        return True
    return False


def find_local_lightcone(t, x):
    # We normalize these in an euclidean sense, as they all have mod 0. This is useful only for visual purposes.
    v1 = 1
    v0, v1 = get_initial_cond(t, x, 0, u1=v1)
    normv = (v0**2+v1**2)**0.5
    v0, v1 = v0/normv, v1/normv

    w1 = 1
    w0, w1 = get_initial_cond(t, x, 0, u1=w1, sign="plus")
    normw = (w0 ** 2 + w1 ** 2) ** 0.5
    w0, w1 = w0 / normw, w1 / normw

    return (v0, v1), (w0, w1), (-v0,-v1), (-w0,-w1)


def find_local_rest_velocity(t, x):
    # Get the local light cone
    l1, l2, l3, l4 = find_local_lightcone(t, x)

    # Find the 4 bisecting vectors
    b1 = l1[0] + l2[0], l1[1] + l2[1]
    b2 = l2[0] + l3[0], l2[1] + l3[1]
    b3 = l3[0] + l4[0], l3[1] + l4[1]
    b4 = l4[0] + l1[0], l4[1] + l1[1]
    bisec = [b1, b2, b3, b4]

    # Those with sq mod < 0 will be timelike, i.e, our local rest velocities for traveling to the future/past once we
    # normalize them
    tlike = []
    for b in bisec:
        mod = get_squared_mod(b[0], b[1], t, x)

        if mod < 0:
            tlike.append((b[0]/(-mod)**0.5, b[1]/(-mod)**0.5))

    # In order to provide a consistent time direction within the bubble
    if in_bubble(t, x):
        at = np.arctan2(t, x)
        if at > 3*np.pi/4 or at < -np.pi/4:
            tlike.reverse()

    return tlike


# Input parameters are "y vectors" cointaining coords and tangent vectors for observer and particle.
def get_energy(observer, particle, modulus):
    # E = - p_mu U^mu
    # POSITION OF OBSERVER AND PARTICLE SHOULD BE THE SAME! (Observer position is not used in the calculation)
    u0, u1, t_obs, x_obs = observer

    # If the particle is massive (follows a timelike geodesic):
    # Let m=1. To find p_mu = 1*U_mu we must lower the velocity index multiplying by the metric!
    # It the particle has m=0, the tangent vector is already the momentum!
    p0, p1, t_p, x_p = particle

    p_0 = p0*gtt(t_p, x_p) + p1*gtx(t_p, x_p)
    p_1 = p0*gtx(t_p, x_p) + p1*gxx(t_p, x_p)

    # Energy perceived by observer will be
    e = - p_0*u0 - p_1*u1

    return np.abs(e)


# Given a trajectory (y output of rk4), calculates energy as measured by inerial observer at each point
def get_energy_on_trajectory(trajectory):
    y = trajectory

    num = len(y[0, :])
    v = np.zeros((2, num))
    for i in range(num):
        v[0, i] = find_local_rest_velocity(y[2, i], y[3, i])[1][0]
        v[1, i] = find_local_rest_velocity(y[2, i], y[3, i])[1][1]

    obs = v[0, :], v[1, :], y[2, :], y[3, :]
    par = y[0, :], y[1, :], y[2, :], y[3, :]

    energy = get_energy(obs, par, -1)
    return energy


# Given a trajectory (y output of rk4), calculates the modulus of 3 velocity (speed)
# as measured by a local inertial observer, or as perceived
# by an external observer (not physically significant, but illustrative).
# We may choose to represent the angle with the v=0 axis (local or external rest velocity)
def get_3speed_on_trajectory(trajectory, local=True, angle=False):
    y = trajectory

    if local:
        # We find the (euclidean) angle between U and the local time arrow.
        # The velocity is the tangent of this angle (u1/u0 in local inertial coordinates...)
        num = len(y[0, :])
        speed = np.zeros(num)

        for i in range(num):
            v0, v1 = find_local_rest_velocity(y[2, i], y[3, i])[1]
            u0, u1 = y[0, i], y[1, i]

            cosa = (u0*v0 + u1*v1)/((u0**2 + u1**2)**0.5 * (v0**2 + v1**2)**0.5)
            alpha = np.arccos(np.clip(cosa, -1, 1))

            if angle:
                speed[i] = alpha if alpha < np.pi/2 else alpha - np.pi/2
                speed[i] *= 180/np.pi
            else:
                speed[i] = abs(np.tan(alpha))

    else:
        # This one is easy (we are calculating the velocity as seen by an external obsever)
        speed = np.abs(y[1, :] / y[0, :])
        if angle:
            speed = np.arctan(speed)/np.pi * 180


    return speed


# Returns a cropped array with only the values when the trajectory is inside the bubble
def cut_outside_bubble(trajectory):
    y = trajectory

    leng = len(y[0, :])
    cutoff = leng
    cutoff_val = 0.35

    for i in range(leng):
        if h(y[2, i], y[3, i]) < cutoff_val:
            cutoff = i
            break

    return y[:, 0:cutoff]


# Returns a series of values depending on the start and endpoint of the trajectory
def get_origin(trajectory):
    y = trajectory

    startpoint = y[2, 0], y[3, 0]
    endpoint = y[2, -1], y[3, -1]

    if startpoint[0] < -140:
        start = -1
    elif startpoint[0] > 140:
        start = 1
    else:
        start = -2

    if endpoint[0] < -140:
        end = -1
    elif endpoint[0] > 140:
        end = 1
    else:
        end = 0

    return start, end