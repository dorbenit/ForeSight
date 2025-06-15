# === IMPORTS ===
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
import numpy as np
import threading
import time
import seaborn as sns
import matplotlib.ticker as mtick
from datetime import timedelta
import matplotlib.font_manager as fm
import xgboost as xgb
from xgboost import Booster
import pandas as pd

# === CONFIGURATION ===
SAMPLE_RATE = 256
WINDOW_SIZE = 256
CHANNEL_COUNT = 5

# Load trained XGBoost model
model = Booster()
model.load_model("xgb_model.json")
print("1")

# Initialize global data structures
timestamps = []
productivity_scores = []
inlet = None

# === GUI SETUP ===
sns.set_theme(style="white")
start_time_global = time.time()
root = tk.Tk()
root.title("Productivity Estimation")

fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_ylim(0, 1)
ax.set_xlim(0, 10)

# Light mode settings
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.tick_params(colors='black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.title.set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set font for ticks
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Segoe UI')

ax.set_xlabel("Time", fontname='Segoe UI')
ax.set_ylabel("Productivity", fontname='Segoe UI')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas.draw()

# Initial plot elements
fit_curve_past, = ax.plot([], [], '-', color='#b20ff7', label='Fitting (Past)')
fit_curve_future, = ax.plot([], [], '-', color='#b20ff7', label='Fitting (Future)')
fill_curve = None
future_fill = None
below_threshold_fill = None
vline = None
vline_text = None
redline = ax.axhline(0.2, color='#f7d00f')  # Threshold line at 20%


# === EEG FEATURE EXTRACTION ===
def extract_bandpowers(window):
    band_features = []
    for ch in range(CHANNEL_COUNT):
        freqs, psd = welch(window[:, ch], fs=SAMPLE_RATE, nperseg=WINDOW_SIZE)

        def bandpower(low, high):
            idx = np.logical_and(freqs >= low, freqs <= high)
            return np.mean(psd[idx])

        band_features += [
            bandpower(0.5, 4),
            bandpower(4, 8),
            bandpower(8, 12),
            bandpower(12, 30),
            bandpower(30, 45)
        ]
    return band_features


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def add_eeg_features(df):
    safe_cols = ['alpha_ch1', 'alpha_ch2', 'beta_ch1', 'beta_ch2']
    df[safe_cols] = df[safe_cols].replace(0, 1e-6)

    df['front_left_right_alph_diffrence'] = df['alpha_ch1'] - df['alpha_ch2']
    df['front_left_right_beta_diffrence'] = df['beta_ch1'] - df['beta_ch2']

    df['beta_gama_ratio_ch0'] = df['beta_ch0'] / df['gamma_ch0']
    df['beta_gama_ratio_ch1'] = df['beta_ch1'] / df['gamma_ch1']
    df['beta_gama_ratio_ch2'] = df['beta_ch2'] / df['gamma_ch2']
    df['beta_gama_ratio_ch3'] = df['beta_ch1'] / df['gamma_ch3']
    df['beta_gama_ratio_ch4'] = df['beta_ch1'] / df['gamma_ch4']

    for ch in range(5):
        df[f'alph_beta_ratio_ch{ch}'] = df[f'alpha_ch{ch}'] / df[f'beta_ch{ch}']

    df['frontal_parietal_beta_ratio'] = (df['beta_ch1'] + df['beta_ch2']) / (df['beta_ch0'] + df['beta_ch3'])
    return df

# === Graph Update ===
def update_plot():
    if not timestamps:
        return

    x = np.array(timestamps)
    y = np.array(productivity_scores)

    # === Add artificial decay points ===
    x_aug = x.copy().tolist()
    y_aug = y.copy().tolist()
    for t in x:
        future_t = t + 3600
        if t % 10 == 0:
            x_aug.append(future_t)
            y_aug.append(0.2)
    x_aug = np.array(x_aug)
    y_aug = np.array(y_aug)
    x_fit = np.linspace(0, max(x_aug) * 1.1, 300)
    # בדיקות בטיחות לפני Polynomial.fit
    if len(x_aug) >= 3 and not np.isnan(x_aug).any() and not np.isnan(y_aug).any() and not np.isinf(
            x_aug).any() and not np.isinf(y_aug).any():
        try:
            poly_model = Polynomial.fit(x_aug, y_aug, deg=2)
            y_poly = poly_model(x_fit)
            mse_poly = np.mean((poly_model(x_aug) - y_aug) ** 2)
        except Exception as e:
            print(f"⚠️ Polynomial fit failed: {e}")
            return
    else:
        print("⚠️ Not enough valid data for polynomial fit.")
        return

    y_poly = poly_model(x_fit)
    mse_poly = np.mean((poly_model(x_aug) - y_aug) ** 2)

    try:
        params, _ = curve_fit(exp_func, x_aug, y_aug, maxfev=10000)
        y_exp = exp_func(x_fit, *params)
        mse_exp = np.mean((exp_func(x_aug, *params) - y_aug) ** 2)
    except:
        y_exp = None
        mse_exp = float('inf')

    y_fit = y_poly if mse_poly < mse_exp else y_exp

    now_time = x[-1]
    past_mask = x_fit <= now_time
    future_mask = x_fit > now_time

    fit_curve_past.set_data(x_fit[past_mask], y_fit[past_mask])
    fit_curve_future.set_data(x_fit[future_mask], y_fit[future_mask])

    global fill_curve, future_fill, below_threshold_fill
    if fill_curve:
        fill_curve.remove()
    if future_fill:
        future_fill.remove()
    if below_threshold_fill:
        below_threshold_fill.remove()

    fill_curve = ax.fill_between(x_fit[past_mask], 0, y_fit[past_mask], color='#b20ff7', alpha=0.3)
    future_fill = ax.fill_between(x_fit[future_mask], 0, y_fit[future_mask], color='#b20ff7', alpha=0.1)

    # === Yellow fill below threshold (appears under 0.2) ===
    below_threshold_fill = ax.fill_between(x_fit, 0, np.minimum(y_fit, 0.2), color='#f7d00f', alpha=0.2)

    # === Time estimation ===
    idxs = np.where((y_fit[:-1] > 0.2) & (y_fit[1:] <= 0.2))[0]
    print("y_fit min:", np.min(y_fit), "| now_time:", now_time)

    est_minutes = "∞"
    if len(idxs) > 0:
        cross_idx = idxs[-1]
        x1, x2 = x_fit[cross_idx], x_fit[cross_idx + 1]
        y1, y2 = y_fit[cross_idx], y_fit[cross_idx + 1]
        slope = (y2 - y1) / (x2 - x1)
        intersect_x = x1 + (0.2 - y1) / slope
        est_sec = max(0, intersect_x - now_time)
        est_minutes = int(est_sec // 60)

    global vline, vline_text
    if vline:
        vline.remove()
    if vline_text:
        vline_text.remove()
    vline = ax.axvline(now_time, color='black', linestyle='-')
    vline_text = ax.text(now_time, 1.02, "Now", color='black', ha='center', va='bottom', fontname='Segoe UI')

    # # === Title ===
    # ax.set_title(f"Estimated productive time: {20} min", fontname='Segoe UI')

    # === Axes limits and ticks ===
    ax.set_xlim(0, max(10, x_fit[-1]))

    # Grid: 10-minute intervals (600 sec) and 10% productivity
    ax.set_xticks(np.arange(0, x_fit[-1] + 600, 600))  # every 10 minutes
    ax.set_yticks(np.arange(0, 1.01, 0.1))             # every 10%
    ax.grid(True, which='major', color='#b20ff7', alpha=0.05)

    # X-axis time formatting
    def format_time(x, pos):
        return str(timedelta(seconds=int(x)))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(format_time))

    canvas.draw()


# === Main Update Loop ===
def update_plot():
    if not timestamps:
        return

    x = np.array(timestamps)
    y = np.array(productivity_scores)

    # Add artificial decay points for better fitting
    x_aug = x.copy().tolist()
    y_aug = y.copy().tolist()
    for t in x:
        future_t = t + 3600
        if t % 10 == 0:
            x_aug.append(future_t)
            y_aug.append(0.2)

    x_aug = np.array(x_aug)
    y_aug = np.array(y_aug)
    x_fit = np.linspace(0, max(x_aug) * 1.1, 300)

    # Polynomial fitting
    if len(x_aug) >= 3 and not np.isnan(x_aug).any() and not np.isnan(y_aug).any() and not np.isinf(x_aug).any() and not np.isinf(y_aug).any():
        try:
            poly_model = Polynomial.fit(x_aug, y_aug, deg=2)
            y_poly = poly_model(x_fit)
            mse_poly = np.mean((poly_model(x_aug) - y_aug) ** 2)
        except Exception as e:
            print(f"⚠️ Polynomial fit failed: {e}")
            return
    else:
        print("⚠️ Not enough valid data for polynomial fit.")
        return

    # Exponential fitting (fallback)
    try:
        params, _ = curve_fit(exp_func, x_aug, y_aug, maxfev=10000)
        y_exp = exp_func(x_fit, *params)
        mse_exp = np.mean((exp_func(x_aug, *params) - y_aug) ** 2)
    except:
        y_exp = None
        mse_exp = float('inf')

    y_fit = y_poly if mse_poly < mse_exp else y_exp

    # Plot updates
    now_time = x[-1]
    past_mask = x_fit <= now_time
    future_mask = x_fit > now_time

    fit_curve_past.set_data(x_fit[past_mask], y_fit[past_mask])
    fit_curve_future.set_data(x_fit[future_mask], y_fit[future_mask])

    global fill_curve, future_fill, below_threshold_fill
    if fill_curve:
        fill_curve.remove()
    if future_fill:
        future_fill.remove()
    if below_threshold_fill:
        below_threshold_fill.remove()

    fill_curve = ax.fill_between(x_fit[past_mask], 0, y_fit[past_mask], color='#b20ff7', alpha=0.3)
    future_fill = ax.fill_between(x_fit[future_mask], 0, y_fit[future_mask], color='#b20ff7', alpha=0.1)
    below_threshold_fill = ax.fill_between(x_fit, 0, np.minimum(y_fit, 0.2), color='#f7d00f', alpha=0.2)

    # Time estimation until productivity drops below threshold (0.2)
    idxs = np.where((y_fit[:-1] > 0.2) & (y_fit[1:] <= 0.2))[0]
    print("y_fit min:", np.min(y_fit), "| now_time:", now_time)

    est_minutes = "∞"
    if len(idxs) > 0:
        cross_idx = idxs[-1]
        x1, x2 = x_fit[cross_idx], x_fit[cross_idx + 1]
        y1, y2 = y_fit[cross_idx], y_fit[cross_idx + 1]
        slope = (y2 - y1) / (x2 - x1)
        intersect_x = x1 + (0.2 - y1) / slope
        est_sec = max(0, intersect_x - now_time)
        est_minutes = int(est_sec // 60)

    global vline, vline_text
    if vline:
        vline.remove()
    if vline_text:
        vline_text.remove()
    vline = ax.axvline(now_time, color='black', linestyle='-')
    vline_text = ax.text(now_time, 1.02, "Now", color='black', ha='center', va='bottom', fontname='Segoe UI')

    # Axes limits and grid
    ax.set_xlim(0, max(10, x_fit[-1]))
    ax.set_xticks(np.arange(0, x_fit[-1] + 600, 600))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(True, which='major', color='#b20ff7', alpha=0.05)

    def format_time(x, pos):
        return str(timedelta(seconds=int(x)))

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(format_time))
    canvas.draw()

# === MAIN UPDATE LOOP ===
def update_loop():
    buffer = []
    while True:
        start = time.time()
        sample, _ = inlet.pull_sample(timeout=1.5)
        if sample is None or len(sample) != CHANNEL_COUNT:
            continue

        buffer.append(sample)
        window = np.array(buffer[-WINDOW_SIZE:])
        features_dict = compute_bandpowers_window(window)

        # Normalize with calibration means
        feature_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        features = {}
        for band in feature_names:
            for ch in range(CHANNEL_COUNT):
                key = f"{band}_ch{ch}"
                raw_val = features_dict.get(key, 0)
                mean_val = calibration_means.get(key, 1e-6)
                features[key] = raw_val - mean_val

        # Transform to DataFrame
        df_features = pd.DataFrame([features])
        df_processed = add_eeg_features(df_features)
        df_processed["trial"] = 0

        try:
            required_features = model.feature_names
            for col in required_features:
                if col not in df_processed.columns:
                    df_processed[col] = 0

            df_processed = df_processed[required_features]
            dmat = xgb.DMatrix(df_processed)
            score = model.predict(dmat)[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            score = 0.5

        score = np.clip(score, 0.7, 0.85)
        score = (score - 0.7) / 0.15

        t = time.time() - start_time_global
        timestamps.append(t)
        productivity_scores.append(score)

        print(f"[{t}] Productivity Score: {score:.2f}")
        update_plot()
        elapsed = time.time() - start
        time.sleep(max(0, 1 - elapsed))

# === Connect to EEG Stream ===
print("🔍 Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=10)
if not streams:
    raise RuntimeError("❌ No EEG stream found. Make sure BlueMuse is running and streaming has started.")
inlet = StreamInlet(streams[0])
print("✅ EEG stream connected.")


# === CALIBRATION PHASE ===
def compute_bandpowers_window(window):
    record = {}
    for ch in range(CHANNEL_COUNT):
        freqs, psd = welch(window[:, ch], fs=SAMPLE_RATE, nperseg=WINDOW_SIZE)

        def bandpower(low, high):
            idx = np.logical_and(freqs >= low, freqs <= high)
            return np.mean(psd[idx])

        record[f'delta_ch{ch}'] = bandpower(0.5, 4)
        record[f'theta_ch{ch}'] = bandpower(4, 8)
        record[f'alpha_ch{ch}'] = bandpower(8, 12)
        record[f'beta_ch{ch}'] = bandpower(12, 30)
        record[f'gamma_ch{ch}'] = bandpower(30, 45)
    return record

print("🧠 Calibrating for 2 minutes...")
raw_samples = []
raw_timestamps = []
start_time = time.time()

# Collect EEG samples for calibration
while time.time() - start_time < 10:
    sample, timestamp = inlet.pull_sample(timeout=1.0)
    if sample is not None and len(sample) == CHANNEL_COUNT:
        raw_samples.append(sample)
        raw_timestamps.append(timestamp)

raw_samples = np.array(raw_samples)
print(f"✅ Collected {len(raw_samples)} samples for calibration.")

# Compute sliding window bandpower features
band_rows = []
for i in range(len(raw_samples) - WINDOW_SIZE + 1):
    window = raw_samples[i:i + WINDOW_SIZE]
    features = compute_bandpowers_window(window)
    band_rows.append(features)

# Compute calibration means for normalization
df_calib = pd.DataFrame(band_rows)
calibration_means = df_calib.mean().to_dict()
print("✅ Calibration means calculated.")


# === START MAIN LOOP ===
# Launch background thread for prediction and plotting
threading.Thread(target=update_loop, daemon=True).start()

# Start Tkinter GUI main loop
root.mainloop()
