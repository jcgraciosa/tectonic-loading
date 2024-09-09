import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.signal import medfilt
from scipy.interpolate import SmoothBivariateSpline


def order_trench(trench_xy, st_lonlat):

    """
    Input:
    trench_xy - numpy array containing two columns - lon (x) and lat (y)
    st_lonlat - initial reference - lon (x) and lat(y)

    Output:
    out_xy - numpy array containing two columns - ordered lon (x) and ordered lat (y)
    """

    trench_x = trench_xy[:, 0]
    trench_y = trench_xy[:, 1]

    st_lon = st_lonlat[0]
    st_lat = st_lonlat[1]

    ordered_x = np.zeros(trench_xy.shape[0])
    ordered_y = np.zeros(trench_xy.shape[0])

    # FIXME: convert this to spherical coordinates - do you need to do this?
    # find nearest to st_latlon
    dist = np.sqrt(np.power(st_lon - trench_x, 2) + np.power(st_lat - trench_y, 2))
    least = np.where(dist == dist.min())[0]
    ordered_x[0], ordered_y[0] = trench_x[least], trench_y[least]

    cp_x = np.delete(trench_x, least)
    cp_y = np.delete(trench_y, least)
    for idx in range(1, trench_x.shape[0]):
        dist = np.sqrt(np.power(ordered_x[idx - 1] - cp_x, 2) + np.power(ordered_y[idx - 1] - cp_y, 2))
        least = np.where(dist == dist.min())[0]
        if least.shape[0] > 1:
            least = least[0]
        ordered_x[idx], ordered_y[idx] = cp_x[least], cp_y[least]
        cp_x = np.delete(cp_x, least)
        cp_y = np.delete(cp_y, least)

    out_xy = np.vstack([ordered_x, ordered_y]).T

    return out_xy
