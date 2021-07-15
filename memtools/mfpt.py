import numpy as np
from numba import njit
from memtools import correlation


def autocorr(x, cutoff=1e-4):
    """ Return autocorrelation of x - mean(x). Cut off the correlation
    after it decays to less than first time. This is only well defined
    for a purely positive time series, like a first-passage time dist. """

    x = np.array(x) - np.mean(x)
    corr = correlation(x, x)
    # if we do not find a hit with argwhere, we get IndexError
    try:
        trunc = np.argwhere(corr < cutoff)[0][0]
    except IndexError:
        tunc = len(x)
    return corr[ : trunc]

def calc_err(corr, N):

    if len(corr) < 1:
        return 0.
    err_sqd = N * corr[0]
    for i in range(1, len(corr)):
        err_sqd += (N - i) * corr[i]
    err_sqd /= N ** 2
    return np.sqrt(err_sqd)

@njit
def _compute_fp_events(crossing_indexes, crossing_times, n_points):
    """ Computes fpt evens from prepared arrays. Expects an array which contains the index of the
    bin the time series crosses into and and the time passes corresponding
    to that crossing event. Returns the fpt distribution starting from the left and the right. """

    fpts_left = [[0.] for __ in range(n_points)]
    fpts_right = [[0.] for __ in range(n_points)]
    # only count the longest passage time found for every transition
    # For every point, we go back to the last occurence of that very same point and write the
    # leftmost starting point of every other point into an array, or -100, if we find no occurence
    found_starting_points = np.zeros(n_points, dtype=np.int64)
    index_to_start = 1
    for i in range(index_to_start, len(crossing_indexes)):
        # occurence not found == -100
        found_starting_points.fill(-100)
        crossing_to = crossing_indexes[i]
        j = i - 1
        crossing_from = crossing_indexes[j]
        # count how often destination is reached from source, not how many
        # crossings of source happend before reaching destination
        while crossing_from != crossing_to:
            # write index of starting point into array
            found_starting_points[crossing_from] = j
            j -= 1
            # keep track of where we are in steping back through the array, stop stepping back when
            # start point == end point or no more data points left in arrray
            if j >= 0:
                crossing_from = crossing_indexes[j]
            else:
                crossing_from = crossing_to
        # now update every starting point that was found:
        for k in range(len(found_starting_points)):
            if found_starting_points[k] >= 0:
                j = found_starting_points[k]
                crossing_from = crossing_indexes[j]
                delta_t = crossing_times[i] - crossing_times[j]
                if crossing_from == 0:
                    fpts_left[crossing_to].append(delta_t)
                if crossing_from == n_points - 1:
                    fpts_right[crossing_to].append(delta_t)
    return fpts_left, fpts_right


def calc_mfpt(traj, start, end, n_points, dt):
    """ Compute first passage events for a given trajectory. Consider all possiple passage
    events over regularly interspaced points over a given intervall. Returns the mean
    of all first passage events and the statistical error
    Arguments:
    start [float]: start of the interval over which to consider fpts
    end [float]: end of the interval over which to consider fpts
    n_points [int]: number of poits over which to consider fpts.
    dt [float]: time step of traj
    Returns:
    mfpt [list of 2 np.ndarray(n_points)] mean first passage times starting from the left 
    and from the right
    mfpt_errs [list of 2 np.ndarray(n_points)] statistical error for mean first passage times
     """
    width_bins = (end - start) / (n_points - 1)
    # map trajectory to index based on which point it is between, or over or under the range
    indexes = ((traj - start) // width_bins).astype(np.int64)
    indexes = np.maximum(indexes, -1)
    indexes = np.minimum(indexes, n_points - 1)
    times = (np.arange(len(traj)) + 1) * dt
    # find the crossing events
    crossings = np.where((indexes[1:] - indexes[:-1]) != 0)[0]
    # and into which bin the transition happens
    crossing_indexes = (indexes[crossings] + indexes[crossings + 1]) // 2 + 1
    # and the corresponding times
    crossing_times = times[crossings]
    fpts_left, fpts_right = _compute_fp_events(crossing_indexes, crossing_times, n_points)
    # pop elements put in to determine type for numba, except start or end, to keep them
    # from being empty
    for i in range(1, n_points):
        fpts_left[i].pop(0)
    for i in range(n_points - 1):
        fpts_right[i].pop(0)
    mfpt = np.array([
        [np.mean(x) for x in fpts_left],
        [np.mean(x) for x in fpts_right],
    ])
    mfpt_err = np.array([
        [calc_err(autocorr(x), len(x)) if len(x) > 1 else 0 for x in fpts_left],
        [calc_err(autocorr(x), len(x)) if len(x) > 1 else 0 for x in fpts_right],
    ])
    return mfpt, mfpt_err
