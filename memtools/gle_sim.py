import numpy as np
from numba import njit


@njit
def du_dx(x, spl_m, values, start, width):
    """ Evaluates the spline representation of the potential at x, returns the negativ
        gradient of the splines. Evaluates to value at the border beyond the border.  """

    # find index of bin by shifting by the start of
    # the first bin and floor dividing
    idx = int((x - start) // width)
    # set gradient of the free energy beyond the borders
    # to the value at the border
    if idx < 0:
        idx = 0
        x = start
    elif idx > len(values) - 2:
        idx = len(values) - 2
        x = values[0] + (len(values) - 1) * width
    # evaluate the gradient of the spline rep
    output = -(
        3 * spl_m[idx, 0] * (x - values[idx]) ** 2
        + 2 * spl_m[idx, 1] * (x - values[idx])
        + spl_m[idx, 2]
    )
    return output


@njit
def compute_state_update(
    update_y,
    current_x,
    current_v,
    state_y,
    dt,
    mass,
    ks,
    gammas,
    spl_m,
    values,
    start,
    width,
    random_force
):
    """ Compute forces for a single rk4 step. """
    update_x = dt * current_v
    update_v = du_dx(current_x, spl_m, values, start, width)
    for i in range(len(ks)):
        update_v += ks[i] * (state_y[i] - current_x)
        update_y[i] = dt * ks[i] / gammas[i] * (current_x - state_y[i]) + random_force[i]
    update_v *= dt / mass
    return update_x, update_v


@njit
def integrate(
    x0: float,
    v0: float,
    y0: np.ndarray,
    steps: int,
    dt: float,
    mass: float,
    kbt: float,
    gammas: np.ndarray,
    ks: np.ndarray,
    spl_m: np.ndarray,
    values: np.ndarray,
    stride: int=1
) -> tuple:
    """ Integrate a GLE simulation with the Runge Kutta method of 4th order. """
    traj = np.zeros(steps // stride)
    x = x0
    v = v0
    y = y0
    ky1 = np.zeros(len(ks))
    ky2 = np.zeros(len(ks))
    ky3 = np.zeros(len(ks))
    ky4 = np.zeros(len(ks))
    random_force = np.zeros(len(ks))
    width = np.mean(values[1:] - values[:-1])
    start = values[0]
    for step in range(steps):
        for j in range(len(ks)):
            random_force[j] = np.sqrt(2 * dt * kbt / gammas[j]) * np.random.normal()
        kx1, kv1 = compute_state_update(
            ky1, x, v, y, dt, mass, ks, gammas, spl_m, values, start, width, random_force
        )
        kx2, kv2 = compute_state_update(
            ky2, x + kx1 / 2., v + kv1 / 2., y + ky1 / 2., dt, mass, ks, gammas, spl_m, values, start, width, random_force
        )
        kx3, kv3 = compute_state_update(
            ky3, x + kx2 / 2., v + kv2 / 2., y + ky2 / 2., dt, mass, ks, gammas, spl_m, values, start, width, random_force
        )
        kx4, kv4 = compute_state_update(
            ky4, x + kx3, v + kv3, y + ky3, dt, mass, ks, gammas, spl_m, values, start, width, random_force
        )
        x += (kx1 + 2. * kx2 + 2. * kx3 + kx4) / 6.
        v += (kv1 + 2. * kv2 + 2. * kv3 + kv4) / 6.
        for j in range(len(ks)):
            y[j] += (ky1[j] + 2. * ky2[j] + 2. * ky3[j] + ky4[j]) / 6.
        if step % stride == 0:
            traj[(step - 1) // stride] = x
    return traj, v, y
