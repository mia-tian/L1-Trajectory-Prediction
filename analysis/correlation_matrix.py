import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pickle as pkl
import datetime as dt
import os
import math
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Use Arial font for all plots
plt.rcParams['font.family'] = 'Arial'


def main():
    """
    Load satellite altitude histories and space weather drivers,
    compute normalized decay-rate proxies, and visualize correlations
    between space weather inputs and satellite responses.

    This script is primarily exploratory and diagnostic:
      - It visualizes altitude histories
      - Computes normalized decay rates
      - Examines correlation structure between drivers and satellites
    """

    # ----------------------------------------------------------
    # Load processed satellite and space weather data
    # ----------------------------------------------------------
    x_train, y_train, x_test, y_test, satcat, alt_by_obj, dadt, \
    t_train, t_test, dalt_mean, dalt_std, norm_factor_test, \
    mean_f107, std_f107, mean_aph, std_aph = get_data_from_tles()

    n_timesteps = x_train.shape[0]
    n_satellites = y_train.shape[1]
    n_features = x_train.shape[1]

    # ----------------------------------------------------------
    # Plot altitude histories for context
    # ----------------------------------------------------------
    plt.figure(figsize=(5, 4))

    for i in range(len(alt_by_obj[0, :])):
        if i < 10:
            plt.plot(
                t_test,
                alt_by_obj[:-1, i].T,
                label=satcat[i],
                color='tab:orange',
                alpha=0.7
            )
        else:
            plt.plot(
                t_test,
                alt_by_obj[:-1, i].T,
                label=satcat[i],
                color='tab:blue',
                alpha=0.7
            )

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Altitude [km]')
    plt.title('Altitude of satellites')
    plt.ylim([360, 950])
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Combine inputs and outputs for correlation analysis
    # ----------------------------------------------------------
    # X = space weather drivers, Y = satellite decay responses
    test_ip_op = np.column_stack((x_test, y_test))

    labels_x = ['F10.7', 'ap']
    labels_y = [str(i) for i in satcat]
    labels = labels_x + labels_y

    # ----------------------------------------------------------
    # Correlation matrix across drivers and satellites
    # ----------------------------------------------------------
    C = np.corrcoef(test_ip_op.T)

    plt.figure()
    plt.imshow(C, cmap='coolwarm')

    # Annotate each cell numerically
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(
                j, i, round(C[i, j], 2),
                ha='center',
                va='center',
                color='black',
                fontsize=6
            )

    plt.xticks(np.arange(len(labels)), labels, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(labels)), labels, fontsize=10)

    # Draw gridlines to separate cells
    plt.gca().set_xticks(np.arange(-0.5, C.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, C.shape[0], 1), minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=0.25)
    plt.tick_params(which="minor", bottom=False, left=False)

    plt.clim(-1, 1)
    plt.colorbar(label='Correlation coefficient')
    plt.show()


def get_data_from_tles():
    """
    Load interpolated satellite altitude data and corresponding
    space weather drivers, then compute normalized decay-rate proxies.

    Returns:
      - Normalized space weather inputs (x_train, x_test)
      - Normalized decay responses (y_train, y_test)
      - Metadata needed to undo normalization and interpret results
    """

    mu = 398600.4418  # km^3/s^2
    norm_alt = True

    # ----------------------------------------------------------
    # Load interpolated altitude histories
    # ----------------------------------------------------------
    file = 'data/example_objs_interp.pkl'
    with open(file, 'rb') as f:
        t, alt_by_obj, satcat = pkl.load(f)

    # ----------------------------------------------------------
    # Load or generate space weather data
    # ----------------------------------------------------------
    start_date = min(t)
    end_date = max(t)

    sw_file = (
        'data/sw_data_'
        + start_date.strftime('%d_%m_%Y')
        + '_'
        + end_date.strftime('%d_%m_%Y')
        + '.pkl'
    )

    if os.path.exists(sw_file):
        with open(sw_file, 'rb') as f:
            f107A, f107, Ap, aph, t_sw = pkl.load(f)
    else:
        print('loading sw data...')
        sw_data = read_sw_nrlmsise00('data/SW-All.csv')
        f107A, f107, Ap, aph = get_sw_params(t, sw_data, 0, 0)
        with open(sw_file, 'wb') as f:
            pkl.dump([f107A, f107, Ap, aph, t], f)

    print('loaded all data!')

    # ----------------------------------------------------------
    # Define training and testing window
    # ----------------------------------------------------------
    tdelta_days = 365
    start_date = dt.datetime(2023, 11, 1)
    end_date = start_date + dt.timedelta(days=tdelta_days)

    t_ts = np.array([dt.datetime.timestamp(ti) for ti in t])
    start_idx = np.argmin(np.abs(t_ts - dt.datetime.timestamp(start_date)))
    end_idx = np.argmin(np.abs(t_ts - dt.datetime.timestamp(end_date)))

    idx_obj = np.arange(len(satcat))
    train_days = 120

    alt_by_obj_train = alt_by_obj[start_idx:start_idx + 24 * train_days, idx_obj]

    # ----------------------------------------------------------
    # Normalize space weather inputs using training statistics
    # ----------------------------------------------------------
    f107_train = f107[start_idx:start_idx + 24 * train_days]
    mean_f107, std_f107 = np.mean(f107_train), np.std(f107_train)
    f107_train = (f107_train - mean_f107) / std_f107

    aph_train = aph[start_idx:start_idx + 24 * train_days]
    mean_aph, std_aph = np.mean(aph_train), np.std(aph_train)
    aph_train = (aph_train - mean_aph) / std_aph

    t_train = t[start_idx:start_idx + 24 * train_days - 1]

    # ----------------------------------------------------------
    # Reference density for drag normalization
    # ----------------------------------------------------------
    if norm_alt:
        d_ref = np.zeros((len(idx_obj), len(alt_by_obj_train) - 1))
        d_ref_all = np.zeros((len(idx_obj), len(alt_by_obj) - 1))

        for i in range(len(idx_obj)):
            d_ref[i, :] = dens_expo(alt_by_obj_train[:-1, i])
            d_ref_all[i, :] = dens_expo(alt_by_obj[:-1, i])
    else:
        d_ref = np.ones((len(idx_obj), len(alt_by_obj_train) - 1))
        d_ref_all = np.ones((len(idx_obj), len(alt_by_obj) - 1))

    # ----------------------------------------------------------
    # Circular-orbit velocity model
    # ----------------------------------------------------------
    def compute_v_for_alt(alt):
        r_e = 6378.15  # km
        alt = alt + r_e
        return np.sqrt(mu / alt)

    # ----------------------------------------------------------
    # Normalize altitude decay rates
    # ----------------------------------------------------------
    v = compute_v_for_alt(alt_by_obj_train)[:-1, :]
    norm_factor_train = d_ref.T * 1e9 * np.sqrt(mu * (alt_by_obj_train[:-1, :] + 6378.15))

    dalt_train_raw = (np.diff(alt_by_obj_train, axis=0) / 3600) / norm_factor_train
    dalt_mean, dalt_std = np.mean(dalt_train_raw, axis=0), np.std(dalt_train_raw, axis=0)
    dalt_train = (dalt_train_raw - dalt_mean) / dalt_std

    # ----------------------------------------------------------
    # Testing data (same normalization)
    # ----------------------------------------------------------
    test_days = tdelta_days - train_days
    alt_by_obj_test = alt_by_obj[start_idx:start_idx + 24 * (train_days + test_days), idx_obj]

    f107_test = (f107[start_idx:start_idx + 24 * (train_days + test_days)] - mean_f107) / std_f107
    aph_test = (aph[start_idx:start_idx + 24 * (train_days + test_days)] - mean_aph) / std_aph

    t_test = t[start_idx:start_idx + 24 * (train_days + test_days) - 1][:-1]

    if norm_alt:
        d_ref_test = np.zeros((len(idx_obj), len(alt_by_obj_test) - 1))
        for i in range(len(idx_obj)):
            d_ref_test[i, :] = dens_expo(alt_by_obj_test[:-1, i])
    else:
        d_ref_test = np.ones((len(idx_obj), len(alt_by_obj_test) - 1))

    v_test = compute_v_for_alt(alt_by_obj_test)[:-1, :]
    norm_factor_test = d_ref_test.T * 1e9 * np.sqrt(mu * (alt_by_obj_test[:-1, :] + 6378.15))

    dalt_test_raw = (np.diff(alt_by_obj_test, axis=0) / 3600) / norm_factor_test
    dalt_test = (dalt_test_raw - dalt_mean) / dalt_std

    # ----------------------------------------------------------
    # Assemble inputs and outputs
    # ----------------------------------------------------------
    x_train = np.column_stack((f107_train[:-1], aph_train[:-1]))
    y_train = dalt_train

    x_test = np.column_stack((f107_test[:-1], aph_test[:-1]))
    y_test = dalt_test

    return (
        x_train, y_train, x_test, y_test,
        satcat, alt_by_obj, np.diff(alt_by_obj, axis=0),
        t_train, t_test,
        dalt_mean, dalt_std,
        norm_factor_test,
        mean_f107, std_f107,
        mean_aph, std_aph
    )


def read_sw_nrlmsise00(swfile):
    """
    Read and preprocess NRLMSISE-compatible space weather data.
    """
    sw_df = pd.read_csv(swfile)
    sw_df.dropna(subset=['C9'], inplace=True)
    sw_df.sort_values(by=['DATE'], ascending=False, inplace=True)
    sw_df.reset_index(drop=True, inplace=True)
    return sw_df


def get_sw_params(t_dt, sw_data, aph_bias, aph_sd):
    """
    Extract F10.7 and ap time histories aligned to satellite epochs.
    """
    aph = np.zeros(len(t_dt))
    f107A = np.zeros(len(t_dt))
    f107 = np.zeros(len(t_dt))
    Ap = np.zeros(len(t_dt))

    for i in range(len(t_dt)):
        f107A[i], f107[i], Ap[i], aph_obs = get_sw(
            sw_data,
            t_dt[i].strftime('%Y-%m-%d'),
            float(t_dt[i].strftime('%H'))
        )

        hour_of_day = t_dt[i].hour
        hour_bins = np.array([0, 3, 6, 9, 12, 15, 18])
        aph[i] = aph_obs[np.argmin(abs(hour_of_day - hour_bins))]
        aph[i] += np.random.normal(aph_bias, aph_sd)

    return f107A, f107, Ap, aph


def get_sw(sw_df, t_ymd, hour):
    """
    Extract daily and sub-daily space weather parameters.
    """
    j = np.where(sw_df['DATE'] == t_ymd)[0][0]
    f107A = sw_df.iloc[j]['F10.7_OBS_CENTER81']
    f107 = sw_df.iloc[j]['F10.7_ADJ']
    ap = sw_df.iloc[j]['AP_AVG']
    aph = sw_df.iloc[j]['AP1':'AP8']
    return f107A, f107, ap, aph


def dens_expo(h):
    """
    Piecewise exponential atmospheric density model.
    """
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
        for h_min, h_max, h_0, rho_0, H in params:
            if h_min <= h_ellp < h_max:
                dens[i] = rho_0 * math.exp(-(h_ellp - h_0) / H)
                break

    return dens


if __name__ == "__main__":
    main()
