"""
Correlation analysis between Swarm A, B, and C orbital evolution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_swarm_csv(path):
    df = pd.read_csv(path)
    df['Date/Time (UTC)'] = pd.to_datetime(df['Date/Time (UTC)'])
    return df['Date/Time (UTC)'], df['SMA'], df['Inclination']


def resample_hourly(time, values, start_date):
    t_rs = pd.date_range(start=start_date, end=time.iloc[-1], freq='H')
    v_rs = np.interp(t_rs, time, values)
    return t_rs, v_rs


def plot_sma_time_series(data):
    plt.figure(figsize=(3, 5))
    for label, (t, sma, style) in data.items():
        plt.plot(t, sma, **style, label=label)

    plt.xlabel('Date')
    plt.ylabel('Altitude [km]')
    plt.xticks(rotation=45)
    plt.grid(axis='y', color='gray', linewidth=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sma_and_inclination(data):
    plt.figure(figsize=(5, 5))

    # SMA subplot
    plt.subplot(2, 1, 1)
    for label, (t, sma, _, style) in data.items():
        plt.plot(t, sma, **style, label=label)
    plt.ylabel('Altitude [km]')
    plt.xticks([])
    plt.legend()

    # Inclination subplot
    plt.subplot(2, 1, 2)
    for label, (t, _, inc, style) in data.items():
        plt.plot(t, inc, **style, label=label)
    plt.xlabel('Date')
    plt.ylabel('Inclination [deg]')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_scatter_and_density(x, y, title):
    # Statistics
    coeff = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    cov = np.cov(x, y)
    scaled_cov = cov / (np.std(x) * np.std(y))

    print(f'{title}')
    print('Scaled covariance:\n', scaled_cov)
    print('Correlation coefficient:', corr)
    print('Best-fit line:', coeff, '\n')

    # Density plot
    plt.figure(figsize=(7, 4))
    plt.grid(True, color='gray', linewidth=0.25)
    plt.hist2d(x, y, bins=50, cmap='viridis', norm=mpl.colors.LogNorm())

    x_ref = np.linspace(x.min(), x.max(), 100)
    y_ref = coeff[0] * x_ref + coeff[1]
    plt.plot(x_ref, y_ref, color='red', label='Best fit')

    plt.xlim([-0.01, 0.005])
    plt.ylim([-0.01, 0.005])
    plt.xlabel('Rate of change in SMA [km/h]')
    plt.ylabel('Rate of change in SMA [km/h]')
    plt.colorbar(label='Count')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
t_A, sma_A, inc_A = load_swarm_csv('data/Orbit-Data-39452-GP.csv')
t_B, sma_B, inc_B = load_swarm_csv('data/Orbit-Data-39451-GP.csv')
t_C, sma_C, inc_C = load_swarm_csv('data/Orbit-Data-39453-GP.csv')

# -----------------------------------------------------------------------------
# Time series plots
# -----------------------------------------------------------------------------
plot_sma_time_series({
    'Swarm A': (t_A, sma_A, dict(color='tab:orange', linewidth=2)),
    'Swarm B': (t_B, sma_B, dict(color='darkcyan', linewidth=2)),
    'Swarm C': (t_C, sma_C, dict(color='tab:purple', linestyle='--', linewidth=2)),
})

plot_sma_and_inclination({
    'Swarm A': (t_A, sma_A, inc_A, dict(color='tab:orange')),
    'Swarm B': (t_B, sma_B, inc_B, dict(color='tab:blue')),
    'Swarm C': (t_C, sma_C, inc_C, dict(color='tab:green', linestyle='--')),
})

# -----------------------------------------------------------------------------
# Hourly resampling
# -----------------------------------------------------------------------------
start_date = pd.to_datetime('2024-01-01')

_, sma_A_rs = resample_hourly(t_A, sma_A, start_date)
_, sma_B_rs = resample_hourly(t_B, sma_B, start_date)
_, sma_C_rs = resample_hourly(t_C, sma_C, start_date)

# Differences (rates)
diff_A = np.diff(sma_A_rs)
diff_B = np.diff(sma_B_rs)
diff_C = np.diff(sma_C_rs)

# -----------------------------------------------------------------------------
# Scatter plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(5, 3))
plt.scatter(diff_A, diff_B, label='A vs B')
plt.scatter(diff_A, diff_C, label='A vs C')
plt.xlabel('Change in Altitude [km]')
plt.ylabel('Change in Altitude [km]')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Density plots and statistics
# -----------------------------------------------------------------------------
plot_scatter_and_density(diff_A, diff_B, 'Swarm A vs Swarm B')
plot_scatter_and_density(diff_A, diff_C, 'Swarm A vs Swarm C')
