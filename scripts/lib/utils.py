from scipy.signal import savgol_filter
from scipy import interpolate
from scipy import signal as sig

import numpy as np

import logging
logging.basicConfig(

    level=logging.INFO, format="%(levelname)s: %(asctime)s: %(message)s"
)


def pad_mul_dim(mul_dim_arr, pad_width, kwargs={'mode':'edge'}):
    dims = mul_dim_arr.shape
    res_arr = np.zeros((dims[0] + pad_width[0] + pad_width[1], dims[1]))
    
    pad_mode = kwargs['mode']
    if pad_mode=='linear_ramp':
        try:
            end_values = kwargs['end_values']
        except:
            end_values = [(m[0], m[-1]) for m in mul_dim_arr.T]
        if end_values is None:
            end_values = [(m[0], m[-1]) for m in mul_dim_arr.T]
        for dim in range(dims[1]):
            res_arr[:,dim] = np.pad(mul_dim_arr[:,dim], pad_width, mode=pad_mode, end_values=end_values[dim])
    else:
        res_arr = np.pad(mul_dim_arr, (pad_width, (0,0)), mode=pad_mode)
    
    return res_arr


def gen_colors(n_colors):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_colors)]
    return colors


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.full([len(arrs), np.max(lens)], np.nan)
    for idx, l in enumerate(arrs):
        arr[idx, :len(l)] = l.flatten()
    
    avg = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return avg, std


def random_with_step(max_idx, size, step):
    res = sorted(np.random.randint(0, max_idx, size))

    to_fix = np.where(np.diff(res) < step)[0]

    while len(to_fix) > 0:
        temp = np.random.randint(max_idx)
        res[to_fix[0]] = temp
        to_fix = np.where(np.diff(sorted(res)) < step)[0]

    return res


# Helper function to return indexes of nans
def nan_helper(y):
    '''
    Helper returning indeces of Nans
    '''
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates all nan values of given array
def interpol(arr):
    '''
    Custom interplation function
    Interpolates only over Nans
    '''

    y = np.transpose(arr)

    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])

    arr = np.transpose(y)

    return arr


def smooth_landmarks(data_mat, confidence=0.5, savgol_window = 5, savgol_order = 1, smooth=False):
    '''
    Use for better results when using the DLC outputs
    Replaces low likelihood points with Nans and interpolates over them

    params:
    data_mat: the landmarks array in the format [[x11, y11, l11, x12, y12, l12, x13, y13, l13], ...]
    where x11,y11,l11 - a joint; x12,y12,l12 - another joint and etc. the first index is time.
    (achieved by pd.DataFrame.to_numpy())
    be careful to pass a copy unless you want and accept changes in the original array

    returns:
    smooth array
    '''

    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])
    
    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan
    # interpolate over low confidence areas
    for i in pose_list:
        i = interpol(i)
        # smooth after interpolation:
        if smooth:
            logging.info('smoothing...')
            i[:,0] = savgol_filter(i[:,0], savgol_window, savgol_order)
            i[:,1] = savgol_filter(i[:,1], savgol_window, savgol_order)

    return pose_list


def my_resample(mul_d_arr, t, o_num, n_num):
    '''
    upsampling method that works how i want it
    '''
    range0 = np.arange(0, t, 1/o_num)
    # range0 = np.linspace(0, t, len(mul_d_arr))
    range1 = np.arange(0, t, 1/n_num)
    
    y = []
    
    if len(mul_d_arr.shape) == 1:
        f = interpolate.interp1d(range0, mul_d_arr, fill_value="extrapolate")
        y.append(f(range1))
    else:
        for dim in range(mul_d_arr.shape[1]):
            f = interpolate.interp1d(range0, mul_d_arr[:, dim], fill_value="extrapolate")
            y.append(f(range1))

    return np.array(list(zip(*y)))
