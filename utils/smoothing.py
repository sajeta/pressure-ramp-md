import numpy as np
import pandas as pd
from scipy.fftpack import rfft, irfft


def smooth(x, window_len=11, window='hanning', w=None):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if w is None:
        if type(window) == str:
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

            if window == 'flat': #moving average
                w = np.ones(window_len,'d')
            elif window == 'kaiser':
                w = np.kaiser(window_len, beta=14)
            else:
                w = eval('np.'+window+'(window_len)')
        else:
            w = window(window_len)

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y


def smooth_conv(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_fft(y, k):
    w = rfft(y)
    # f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
    spectrum = w**2

    cutoff_idx = spectrum < (spectrum.max()/k)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    return irfft(w2)


def moving_avg(x, n):
    # cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[n:] - cumsum[:-n]) / float(n)
    return pd.Series(x).rolling(n, min_periods=1).mean().values


def block_avg(x, n_blocks):
    tmp = np.array_split(x, n_blocks)
    # avg = np.vectorize(lambda X: np.mean(X))(tmp)
    avg = np.array([np.mean(X) for X in tmp])
    return avg


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def moving_average_(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
