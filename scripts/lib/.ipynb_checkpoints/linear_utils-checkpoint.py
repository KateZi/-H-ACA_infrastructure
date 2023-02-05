import numpy as np

import scipy.signal as sig
import scipy.optimize as optimize

import lib.utils as utils

import logging

logging.basicConfig(

    level=logging.INFO, format="%(levelname)s: %(asctime)s: %(message)s"
)


def linear_alignment(X, Y, alpha=0.5, bnds=((-0.25,0.25), (0.75, 1.25)), crop=False, method=None):
    """
    Return alignment parameters for X, aligned (to Y) X
    Works for zero knots
    """
    Y_len = len(Y)
    offset = 0
    coeff = 1
    p0 = np.r_[offset, coeff]

    def func(p):
        res = warp_linear(X, p, crop=crop)
        Y2 = res[0]
        lp,rp = res[-1]
        return Y2, [lp,rp]

    def err(p):
        res = func(p)
        Y2 = res[0]
        lp,rp = res[1]

        intersection_idx = (0,min(Y_len, len(Y2)))
        
        cost = alpha * np.linalg.norm(Y[intersection_idx[0]:intersection_idx[1]] - Y2[intersection_idx[0]:intersection_idx[1]]) + (1 - alpha) * np.linalg.norm(p- p0)
        return cost

    r = optimize.minimize(err, x0=p0, bounds=bnds, method=method)

    res = func(r.x)

    return r.fun, r.x, res


def warp_linear(signal, params, crop=False):

    off = params[0]
    slope = params[1]

    signal_shape = list(signal.shape)
    og_shape = signal_shape.copy()

    # if we stretch we end up with a longer signal for now
    # preserves shape, parameters and potential prefered window size
    if slope < 1:
        signal_shape[0] = int(signal_shape[0]/slope)
    
    warped_signal = np.zeros(shape=signal_shape)
    my_range = np.arange(0, signal_shape[0])

    # to crop the actual signal
    ts = []
    # to smooth if we don't want a sharp crop and to preserve the size
    lp, rp = 0, 0

    for t in my_range:
        # piecewise interpolation
        x = t / (og_shape[0] - 1)

        z = off + slope * x

        if z <= 0:
            warped_signal[t] = signal[0]
            lp += 1
        elif z >= 1:
            warped_signal[t] = signal[-1]
            rp += 1
        else:
            _i = z * (og_shape[0] - 1)
            rem = _i % 1
            i = int(_i)
            temp = (1 - rem) * signal[i] + rem * signal[i + 1]
            warped_signal[t] = temp
            ts.append(t)

    if len(ts) != 0:
        ts = np.array(ts)
        warped_signal = warped_signal[ts]
        res_signal = warped_signal
    if not crop:
        end_values = [(ele[min(0,ts[0]-1)],ele[min(len(signal)-1,len(signal)-ts[-1]+1)]) for ele in signal.T]
        end_values = None
        res_signal = utils.pad_mul_dim(res_signal, (lp,rp), 
                                       {'mode':'linear_ramp', 
                                        'end_values': end_values})
    
    return res_signal, [lp, rp] 