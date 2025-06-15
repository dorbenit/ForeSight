import tkinter as tk
import random
import time
import csv
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch

# === CONFIGURATION ===
WORDS = ['RED', 'GREEN', 'BLUE', 'YELLOW']
COLORS = ['red', 'green', 'blue', 'yellow']
COLOR_KEYS = {'r': 'red', 'g': 'green', 'b': 'blue', 'y': 'yellow'}
TRIALS = 400
SAMPLE_RATE = 256
WINDOW_SIZE = 256
EEG_OUTPUT = "muse_bandpowers_moving_stride1.csv"
STROOP_OUTPUT = "stroop_results.csv"
stop_event = threading.Event()

# === GLOBAL VARIABLES ===
stroop_results = []


# === EEG RECORDING THREAD FUNCTION ===
def record_eeg(inlet, channel_count):
    print("⏺ Starting EEG recording...")
    buffer = []
    timestamps = []
    start_time = time.time()

    while not stop_event.is_set():
        sample, timestamp = inlet.pull_sample(timeout=0.5)
        if sample and len(sample) == channel_count:
            buffer.append(sample)
            timestamps.append(timestamp)

    print("🛑 Stopped EEG recording.")

    # Compute band powers using moving window
    print("⚙ Computing band powers...")
    buffer = np.array(buffer)
    band_rows = []

    for i in range(0, len(buffer) - WINDOW_SIZE - 1, 16):
        window = buffer[i:i + WINDOW_SIZE]
        time_mid = timestamps[i + WINDOW_SIZE // 2]
        record = {'timestamp': time_mid}

        for ch in range(channel_count):
            freqs, psd = welch(window[:, ch], fs=SAMPLE_RATE, nperseg=WINDOW_SIZE)

            def bandpower(low, high):
                idx = np.logical_and(freqs >= low, freqs <= high)
                return np.mean(psd[idx])

            record[f'delta_ch{ch}'] = bandpower(0.5, 4)
            record[f'theta_ch{ch}'] = bandpower(4, 8)
            record[f'alpha_ch{ch}'] = bandpower(8, 12)
            record[f'beta_ch{ch}'] = bandpower(12, 30)
            record[f'gamma_ch{ch}'] = bandpower(30, 45)

        band_rows.append(record)

    # Save EEG bandpower results
    df = pd.DataFrame(band_rows)
    df.to_csv(EEG_OUTPUT, index=False)
    print(f"✅ EEG saved: {EEG_OUTPUT}")


# === STROOP TEST FUNCTION ===
def run_stroop():
    def next_trial():
        nonlocal trial
        if trial >= TRIALS:
            stop_event.set()
            root.destroy()
            return

        word = random.choice(WORDS)
        color = random.choice(COLORS)
        label.config(text=word, fg=color)
        root.update()
        trial_start = time.time()

        def on_key(event):
            nonlocal trial
            if event.char.lower() in COLOR_KEYS:
                response_color = COLOR_KEYS[event.char.lower()]
                correct = response_color == color
                stroop_results.append({
                    'trial': trial,
                    'timestamp': trial_start,
                    'word': word,
                    'color': color,
                    'response': response_color,
                    'correct': correct,
                    'response_time': time.time() - trial_start
                })
                label.unbind("<Key>")
                trial += 1
                next_trial()

        label.bind("<Key>", on_key)

    # Set up GUI
    trial = 0
    root = tk.Tk()
    root.title("Stroop Test")
    root.geometry("400x200")
    label = tk.Label(root, text="", font=("Helvetica", 32))
    label.pack(expand=True)
    root.after(1000, next_trial)
    label.focus_set()
    root.mainloop()


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Connect to EEG stream before starting threads
    print("🔍 Resolving EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        raise RuntimeError("❌ No EEG stream found. Please start BlueMuse and ensure streaming is active.")

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    channel_count = info.channel_count()
    ch_names = [info.desc().child("channels").child("channel").child_value("label")]
    print(f"✅ EEG stream connected. Channels: {channel_count}")

    # Start EEG recording in a separate thread
    eeg_thread = threading.Thread(target=record_eeg, args=(inlet, channel_count))
    eeg_thread.start()

    # Run Stroop task in main thread
    run_stroop()

    # Wait for EEG thread to finish
    eeg_thread.join()

    # Save Stroop test results
    with open(STROOP_OUTPUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stroop_results[0].keys())
        writer.writeheader()
        writer.writerows(stroop_results)
    print(f"✅ Stroop results saved: {STROOP_OUTPUT}")
