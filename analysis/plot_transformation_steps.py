import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pickle as pkl
import datetime as dt
import os
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# set font to arial
plt.rcParams['font.family'] = 'Arial'
from scipy.ndimage import generic_filter


def main():
    """
    Entry point for data loading and preprocessing.

    This function primarily exists to call get_data_from_tles(),
    which:
      - Loads altitude histories from interpolated TLE data
      - Aligns space weather drivers
      - Computes normalized decay metrics suitable for modeling
    """

    (
        x_train, y_train,
        x_test, y_test,
        satcat,
        alt_by_obj,
        dadt,
        t_train, t_test,
        dalt_mean, dalt_std,
        norm_factor_test,
        mean_f107, std_f107,
        mean_aph, std_aph,
        dalt_train_raw,
        dalt_train_raw_stats
    ) = get_data_from_tles()



def keep_long_runs(arr, min_length=5):
    """
    Retain only indices that belong to sufficiently long
    consecutive integer runs.

    This is useful for identifying persistent events or
    filtering short-lived detections.

    Parameters
    ----------
    arr : ndarray
        Array of integer indices.
    min_length : int
        Minimum length of consecutive run to keep.

    Returns
    -------
    ndarray
        Filtered array containing only indices in long runs.
    """

    arr = np.sort(np.unique(arr))
    diffs = np.diff(arr)

    # Identify breaks in consecutiveness
    breaks = np.where(diffs != 1)[0]

    # Start/end indices of each run
    starts = np.insert(breaks + 1, 0, 0)
    ends = np.append(breaks, len(arr) - 1)

    result = []
    for start, end in zip(starts, ends):
        run = arr[start:end + 1]
        if len(run) >= min_length:
            result.extend(run)

    return np.array(result)


def get_data_from_tles():
    """
    Load satellite altitude histories and space weather data,
    then compute normalized decay metrics for training and testing.

    This routine:
      - Loads interpolated altitude time series
      - Aligns space weather drivers (F10.7, ap/aph)
      - Computes drag-normalized decay rates
      - Produces normalized features and targets for regression

    Returns
    -------
    Multiple arrays used downstream for modeling and diagnostics.
    """

    norm_alt = True

    # ----------------------------------------------------------
    # Load interpolated altitude histories and satellite catalog
    # ----------------------------------------------------------
    file = 'data/example_objs_interp.pkl'
    with open(file, 'rb') as f:
        t, alt_by_obj, satcat = pkl.load(f)

    # ----------------------------------------------------------
    # Load or cache space weather data
    # ----------------------------------------------------------
    start_date = min(t)
    end_date = max(t)
    sw_cache = (
        'data/sw_data_'
        + start_date.strftime('%d_%m_%Y')
        + '_'
        + end_date.strftime('%d_%m_%Y')
        + '.pkl'
    )

    if os.path.exists(sw_cache):
        with open(sw_cache, 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(t, sw_data, 0, 0)
        with open(sw_cache, 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, t], f)

    print('loaded all data!')

    # ----------------------------------------------------------
    # Define training interval
    # ----------------------------------------------------------
    tdelta_days = 365
    start_date = dt.datetime(2023, 11, 1)
    end_date = start_date + dt.timedelta(days=tdelta_days)

    # Convert datetime to timestamps for indexing
    t_ts = np.array([dt.datetime.timestamp(tt) for tt in t])
    start_idx = np.argmin(np.abs(t_ts - dt.datetime.timestamp(start_date)))
    end_idx = np.argmin(np.abs(t_ts - dt.datetime.timestamp(end_date)))

    idx_obj = np.arange(len(satcat))
    train_days = 365

    # ----------------------------------------------------------
    # Training data extraction
    # ----------------------------------------------------------
    alt_by_obj_train = alt_by_obj[start_idx:start_idx + 24 * train_days, idx_obj]
    f107_train = f107[start_idx:start_idx + 24 * train_days]
    aph_train = aph[start_idx:start_idx + 24 * train_days]

    mean_f107, std_f107 = np.mean(f107_train), np.std(f107_train)
    mean_aph, std_aph = np.mean(aph_train), np.std(aph_train)

    f107_train = (f107_train - mean_f107) / std_f107
    aph_train = (aph_train - mean_aph) / std_aph

    t_train = t[start_idx:start_idx + 24 * train_days - 1]

    # ----------------------------------------------------------
    # Reference density model (exponential atmosphere)
    # ----------------------------------------------------------
    if norm_alt:
        d_ref = np.zeros((len(idx_obj), len(alt_by_obj_train) - 1))
        d_ref_all = np.zeros((len(idx_obj), len(alt_by_obj) - 1))
        for i in range(len(idx_obj)):
            d_ref[i, :] = dens_expo(alt_by_obj_train[:-1, i]) * 1e9
            d_ref_all[i, :] = dens_expo(alt_by_obj[:-1, i]) * 1e9
    else:
        d_ref = np.ones((len(idx_obj), len(alt_by_obj_train) - 1))
        d_ref_all = np.ones((len(idx_obj), len(alt_by_obj) - 1))

    def compute_v_for_alt(alt):
        """Circular orbital velocity for given altitude [km]."""
        r_e = 6378.15
        mu = 398600.4418
        return np.sqrt(mu / (alt + r_e))

    # ----------------------------------------------------------
    # Drag-normalized decay computation
    # ----------------------------------------------------------
    v = compute_v_for_alt(alt_by_obj_train)[:-1, :]
    a = alt_by_obj_train[:-1, :] + 6378.15
    mu = 398600.4418

    norm_factor_train = d_ref.T * np.sqrt(a * mu)
    dadt = np.diff(alt_by_obj_train, axis=0) / 3600
    dalt_train_raw = dadt / norm_factor_train

    # Remove positive (non-decaying) values
    dalt_train_raw_stats = np.where(dalt_train_raw > 0, np.nan, dalt_train_raw)

    dalt_mean = np.nanmean(dalt_train_raw_stats, axis=0)
    dalt_std = np.nanstd(dalt_train_raw_stats, axis=0)

    dalt_train = (dalt_train_raw_stats - dalt_mean) / dalt_std

    # ----------------------------------------------------------
    # Diagnostic plots
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 6))

    xticks = [dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 1), dt.datetime(2025, 1, 1)]
    xticklabels = [f"{d.year}/{d.month}" for d in xticks]

    ax1 = plt.subplot(2, 2, 1)
    plt.plot(t_train, alt_by_obj_train[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel('Altitude [km]')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)

    ax2 = plt.subplot(2, 2, 2)
    plt.plot(t_train[:-1], dadt[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel(r'$\dot{a}$ [km/s]')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    ax3 = plt.subplot(2, 2, 3)
    plt.plot(t_train[:-1], dalt_train_raw[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel(r'$d_s$')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels)

    ax4 = plt.subplot(2, 2, 4)
    plt.plot(t_train[:-1], dalt_train[:, :10], color="#3099ac", alpha=0.3)
    plt.ylabel(r'$d_{sn}$')
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticklabels)

    plt.tight_layout()
    plt.savefig("plots/plot_transformation_steps.png", dpi=300)
    plt.close()


    # ----------------------------------------------------------
    # Test set construction
    # ----------------------------------------------------------
    test_days = tdelta_days - train_days
    alt_by_obj_test = alt_by_obj[start_idx:start_idx + 24 * (train_days + test_days), idx_obj]

    f107_test = (f107[start_idx:start_idx + 24 * (train_days + test_days)] - mean_f107) / std_f107
    aph_test = (aph[start_idx:start_idx + 24 * (train_days + test_days)] - mean_aph) / std_aph

    t_test = t[start_idx:start_idx + 24 * (train_days + test_days) - 1]

    if norm_alt:
        d_ref_test = np.zeros((len(idx_obj), len(alt_by_obj_test) - 1))
        for i in range(len(idx_obj)):
            d_ref_test[i, :] = dens_expo(alt_by_obj_test[:-1, i])
    else:
        d_ref_test = np.ones((len(idx_obj), len(alt_by_obj_test) - 1))

    v_test = compute_v_for_alt(alt_by_obj_test)[:-1, :]
    norm_factor_test = (alt_by_obj_test[:-1, :] + 6378.15) * d_ref_test.T * v_test ** 2

    dalt_test_raw = np.diff(alt_by_obj_test, axis=0) / norm_factor_test
    dalt_test_raw[dalt_test_raw > 0] = np.nan
    dalt_test = (dalt_test_raw - dalt_mean) / dalt_std

    # ----------------------------------------------------------
    # Final feature/target matrices
    # ----------------------------------------------------------
    x_train = np.column_stack((f107_train[:-1], aph_train[:-1]))
    y_train = dalt_train

    x_test = np.column_stack((f107_test[:-1], aph_test[:-1]))
    y_test = dalt_test

    return (
        x_train, y_train,
        x_test, y_test,
        satcat,
        alt_by_obj,
        np.diff(alt_by_obj, axis=0),
        t_train, t_test,
        dalt_mean, dalt_std,
        norm_factor_test,
        mean_f107, std_f107,
        mean_aph, std_aph,
        dalt_train_raw,
        dalt_train_raw_stats
    )


def _vectorized_fill_with_closest_valid(arr):
    """Fill NaNs in 2D array with the closest non-NaN values along each column."""
    arr = arr.copy()
    isnan = np.isnan(arr)
    idx = np.where(~isnan, np.arange(arr.shape[0])[:, None], np.nan)
    filled_idx = np.where(isnan, np.nan, np.arange(arr.shape[0])[:, None])

    # Forward fill
    fwd = np.maximum.accumulate(np.where(np.isnan(idx), -np.inf, idx))
    fwd_vals = arr[fwd.astype(int), np.arange(arr.shape[1])]

    # Backward fill
    rev = np.flip(np.maximum.accumulate(np.where(np.isnan(np.flip(idx, axis=0)), -np.inf, np.flip(idx, axis=0))), axis=0)
    back_vals = arr[rev.astype(int), np.arange(arr.shape[1])]

    # Distances
    dist_fwd = np.abs(np.arange(arr.shape[0])[:, None] - fwd)
    dist_back = np.abs(np.arange(arr.shape[0])[:, None] - rev)

    # Choose closer
    use_fwd = dist_fwd <= dist_back
    filled = np.where(isnan, np.where(use_fwd, fwd_vals, back_vals), arr)

    return filled

def _vectorized_rolling_stat_fixed(arr, window, func, axis=0, k=3.0, ddof=0):
    """Generic rolling function with outlier rejection using scipy generic_filter."""
    def wrapped_func(x):
        x = x.reshape(-1)
        if np.all(np.isnan(x)):
            return np.nan
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return func(x)
        x = x[np.abs(x - mean) <= k * std]
        if len(x) == 0:
            return np.nan
        return func(x)

    size = (window, 1) if axis == 0 else (1, window)
    result = generic_filter(arr, function=wrapped_func, size=size, mode='nearest')
    return _vectorized_fill_with_closest_valid(result) if axis == 0 else _vectorized_fill_with_closest_valid(result.T).T

def rolling_nanmean(arr, window, axis=0, k=3.0):
    return _vectorized_rolling_stat_fixed(arr, window, np.nanmean, axis=axis, k=k)

def rolling_nanstd(arr, window, axis=0, k=3.0, ddof=0):
    return _vectorized_rolling_stat_fixed(arr, window, lambda x: np.nanstd(x, ddof=ddof), axis=axis, k=k, ddof=ddof)


def read_sw_nrlmsise00(swfile):
    '''
    Parse and read the space weather data

    Usage: 
    sw_obs_pre = read_sw_nrlmsise00(swfile)

    Inputs: 
    swfile -> [str] Path of the space weather data
    
    Outputs: 
    sw_obs_pre -> [2d str array] Content of the space weather data

    Examples:
    >>> swfile = 'sw-data/SW-All.csv'
    >>> sw_obs_pre = read_sw(swfile)
    >>> print(sw_obs_pre)
    [['2020' '01' '07' ... '72.4' '68.0' '71.0']
    ['2020' '01' '06' ... '72.4' '68.1' '70.9']
    ...
    ...
    ['1957' '10' '02' ... '253.3' '267.4' '231.7']
    ['1957' '10' '01' ... '269.3' '266.6' '230.9']]
    '''
    sw_df = pd.read_csv(swfile)  
    sw_df.dropna(subset=['C9'],inplace=True)
    # Sort from newest date to past
    sw_df.sort_values(by=['DATE'],ascending=False,inplace=True)
    sw_df.reset_index(drop=True,inplace=True)
    return sw_df

def get_sw_params(t_dt, sw_data, aph_bias, aph_sd):
    aph = np.zeros(len(t_dt))
    f107A = np.zeros(len(t_dt))
    f107 = np.zeros(len(t_dt))
    Ap = np.zeros(len(t_dt))
    for i in range(len(t_dt)):
        # query the model
        f107A[i],f107[i],Ap[i],aph_obs = get_sw(sw_data,t_dt[i].strftime('%Y-%m-%d'),float(t_dt[i].strftime('%H')))
        hour_of_day = t_dt[i].hour
        hour = np.array([0,3,6,9,12,15,18])
        aph[i] = aph_obs[np.argmin(abs(hour_of_day-hour))]

        # define random deviation on ap
        aph_dev = np.random.normal(aph_bias,aph_sd)
        aph[i] = aph[i] + aph_dev

    return f107A, f107, Ap, aph

def get_sw(sw_df,t_ymd,hour):
    '''
    Extract the necessary parameters describing the solar activity and geomagnetic activity from the space weather data.

    Usage: 
    f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)

    Inputs: 
    SW_OBS_PRE -> [2d str array] Content of the space weather data
    t_ymd -> [str array or list] ['year','month','day']
    hour -> []
    
    Outputs: 
    f107A -> [float] 81-day average of F10.7 flux
    f107 -> [float] daily F10.7 flux for previous day
    ap -> [int] daily magnetic index 
    aph -> [float array] 3-hour magnetic index 

    Examples:
    >>> f107A,f107,ap,aph = get_sw(SW_OBS_PRE,t_ymd,hour)
    '''
    ymds = sw_df['DATE']
    j_, = np.where(sw_df['DATE'] == t_ymd)
    j = j_[0]
    f107A,f107,ap = sw_df.iloc[j]['F10.7_OBS_CENTER81'],sw_df.iloc[j]['F10.7_ADJ'],sw_df.iloc[j]['AP_AVG']
    aph_tmp_b0 = sw_df.iloc[j]['AP1':'AP8']   

    return f107A,f107,ap,aph_tmp_b0

def dens_expo(h):
    params = [
        (0, 25, 0, 1.225, 7.249),
        (25, 30, 25, 3.899e-2, 6.349),
        (30, 40, 30, 1.774e-2, 6.682),
        (40, 50, 40, 3.972e-3, 7.554),
        (50, 60, 50, 1.057e-3, 8.382),
        (60, 70, 60, 3.206e-4, 7.714),
        (70, 80, 70, 8.77e-5, 6.549),
        (80, 90, 80, 1.905e-5, 5.799),
        (90, 100, 90, 3.396e-6, 5.382),
        (100, 110, 100, 5.297e-7, 5.877),
        (110, 120, 110, 9.661e-8, 7.263),
        (120, 130, 120, 2.438e-8, 9.473),
        (130, 140, 130, 8.484e-9, 12.636),
        (140, 150, 140, 3.845e-9, 16.149),
        (150, 180, 150, 2.070e-9, 22.523),
        (180, 200, 180, 5.464e-10, 29.74),
        (200, 250, 200, 2.789e-10, 37.105),
        (250, 300, 250, 7.248e-11, 45.546),
        (300, 350, 300, 2.418e-11, 53.628),
        (350, 400, 350, 9.518e-12, 53.298),
        (400, 450, 400, 3.725e-12, 58.515),
        (450, 500, 450, 1.585e-12, 60.828),
        (500, 600, 500, 6.967e-13, 63.822),
        (600, 700, 600, 1.454e-13, 71.835),
        (700, 800, 700, 3.614e-14, 88.667),
        (800, 900, 800, 1.17e-14, 124.64),
        (900, 1000, 900, 5.245e-15, 181.05),
        (1000, float('inf'), 1000, 3.019e-15, 268)
    ]
    
    dens = np.zeros(len(h))
    
    for i, h_ellp in enumerate(h):
        for (h_min, h_max, h_0, rho_0, H) in params:
            if h_min <= h_ellp < h_max:
                dens[i] = rho_0 * math.exp(-(h_ellp - h_0) / H)
                break
    
    return dens

if __name__ == "__main__":
    main()