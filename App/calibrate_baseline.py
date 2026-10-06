from pathlib import Path
import csv
import json
import re
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# Files
SCRIPT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
CONFIG_FILE = SCRIPT_DIR / "baseline_config.json"


# Channels
USE_CH1 = True
USE_CH2 = True
USE_CH3 = True
USE_CH4 = True

CHANNELS = {
    "CH1": {"header_name": "EXG Channel 0", "numeric_column": 1, "enabled": USE_CH1},
    "CH2": {"header_name": "EXG Channel 1", "numeric_column": 2, "enabled": USE_CH2},
    "CH3": {"header_name": "EXG Channel 2", "numeric_column": 3, "enabled": USE_CH3},
    "CH4": {"header_name": "EXG Channel 3", "numeric_column": 4, "enabled": USE_CH4},
}

# Threshold multiplier for each channel
CHANNEL_K = {"CH1": 4.0, "CH2": 4.0, "CH3": 4.0, "CH4": 4.0}


# Calibration settings
DEFAULT_FS = 250.0
IGNORE_SECONDS = 10.0
BASELINE_SECONDS = 20.0

HIGHPASS_HZ = 20.0
LOWPASS_HZ = 100.0
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
FILTER_ORDER = 4

ACTIVITY_WINDOW_MS = 50.0
REARM_FRACTION = 0.30
FEATURE_NOISE_MULTIPLIER = 2.0


# Plot text sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 30,
    "axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})


def get_channels():
    active_channels = {}

    for name, info in CHANNELS.items():
        if info["enabled"]:
            active_channels[name] = info

    if not active_channels:
        raise ValueError("At least one channel must be enabled.")

    return active_channels


def choose_file():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    selected = filedialog.askopenfilename(
        parent=root,
        title="Select REST recording for calibration",
        initialdir=str(RECORDINGS_DIR),
        filetypes=[("OpenBCI recordings", "*.txt *.csv"), ("All files", "*.*")],
    )

    root.destroy()

    if not selected:
        raise SystemExit("No file selected.")

    return Path(selected)


def split_line(line):
    """Split one row from either a CSV, tab-separated or semicolon-separated file."""
    line = line.strip()

    if "\t" in line:
        return [item.strip() for item in line.split("\t")]

    if ";" in line and "," not in line:
        return [item.strip() for item in line.split(";")]

    return [item.strip() for item in next(csv.reader([line]))]


def is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def read_openbci_file(file_path, channels):
    """Read the selected OpenBCI recording and return the sample rate and EMG channels."""
    lines = file_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()

    if not lines:
        raise ValueError("Recording is empty.")

    # Read the sample rate from the file. Use 250 Hz if it is not written there.
    fs = DEFAULT_FS

    for line in lines:
        match = re.search(r"Sample\s*Rate\s*=\s*([0-9]+(?:\.[0-9]+)?)", line, re.I)

        if match:
            fs = float(match.group(1))
            break

    # Find the row containing the OpenBCI column names.
    header = None
    header_index = None

    for index, line in enumerate(lines):
        if "EXG Channel" not in line:
            continue

        possible_header = split_line(line)

        for item in possible_header:
            if "EXG Channel" in item:
                header = possible_header
                header_index = index
                break

        if header is not None:
            break

    data = {}
    for channel in channels:
        data[channel] = []

    # Normal OpenBCI file: find each channel using its column name.
    if header is not None:
        column_numbers = {}

        for channel, info in channels.items():
            wanted = info["header_name"]

            if wanted not in header:
                raise ValueError(f"Missing column: {wanted}")

            column_numbers[channel] = header.index(wanted)

        for line in lines[header_index + 1:]:
            if not line.strip() or line.lstrip().startswith("%"):
                continue

            row = split_line(line)

            try:
                row_values = {}
                for channel, column in column_numbers.items():
                    row_values[channel] = float(row[column])
            except (ValueError, IndexError):
                continue

            for channel, value in row_values.items():
                data[channel].append(value)

    # Backup for files without column names: use the saved column numbers.
    else:
        max_column = max(info["numeric_column"] for info in channels.values())
        valid_rows = 0

        for line in lines:
            if not line.strip() or line.lstrip().startswith("%"):
                continue

            row = split_line(line)

            if len(row) <= max_column or not is_number(row[0]):
                continue

            try:
                row_values = {}
                for channel, info in channels.items():
                    row_values[channel] = float(row[info["numeric_column"]])
            except (ValueError, IndexError):
                continue

            for channel, value in row_values.items():
                data[channel].append(value)

            valid_rows += 1

        if valid_rows == 0:
            raise ValueError("Could not recognise OpenBCI recording format.")

    # Convert the Python lists to NumPy arrays.
    for channel in data:
        data[channel] = np.asarray(data[channel], dtype=float)

        if data[channel].size == 0:
            raise ValueError(f"No usable samples for {channel}.")

    # Keep all channels the same length.
    shortest = min(len(values) for values in data.values())

    for channel in data:
        data[channel] = data[channel][:shortest]

    return fs, data


def filter_emg(raw_emg, fs):
    """Centre the EMG, remove 50 Hz noise and apply the 20-100 Hz band-pass filter."""
    filtered = np.asarray(raw_emg, dtype=float)
    filtered = filtered - np.mean(filtered)

    nyquist = fs / 2.0
    lowpass = min(LOWPASS_HZ, nyquist * 0.90)

    # 50 Hz notch filter
    b, a = signal.iirnotch(NOTCH_HZ, NOTCH_Q, fs=fs)
    filtered = signal.lfilter(b, a, filtered)

    # 20-100 Hz band-pass filter
    sos = signal.butter(
        FILTER_ORDER,
        [HIGHPASS_HZ, lowpass],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return signal.sosfilt(sos, filtered)


def activity_signal(filtered_emg, fs):
    """Turn the filtered EMG into the moving absolute-EMG activity signal."""
    window_samples = round(ACTIVITY_WINDOW_MS * fs / 1000.0)
    window_samples = max(1, window_samples)

    rectified_emg = np.abs(filtered_emg)
    window = np.ones(window_samples) / fs

    activity = np.convolve(rectified_emg, window, mode="full")
    return activity[:len(filtered_emg)]


def main():
    channels = get_channels()
    file_path = choose_file()
    fs, raw_data = read_openbci_file(file_path, channels)

    # First 10 s are ignored. The next 20 s are the rest baseline.
    baseline_start = round(IGNORE_SECONDS * fs)
    baseline_end = round((IGNORE_SECONDS + BASELINE_SECONDS) * fs)

    recording_length = min(len(values) for values in raw_data.values())

    if recording_length <= baseline_end:
        raise ValueError("Recording must be longer than 30 seconds.")

    config = {
        "version": 1,
        "source_file": file_path.name,
        "sample_rate": fs,
        "filter": {
            "highpass_hz": HIGHPASS_HZ,
            "lowpass_hz": LOWPASS_HZ,
            "notch_hz": NOTCH_HZ,
            "notch_q": NOTCH_Q,
            "filter_order": FILTER_ORDER,
        },
        "activity_window_ms": ACTIVITY_WINDOW_MS,
        "rearm_fraction": REARM_FRACTION,
        "feature_noise_multiplier": FEATURE_NOISE_MULTIPLIER,
        "channels": {},
    }

    filtered_data = {}
    activity_data = {}

    print("\nBASELINE CALIBRATION")
    print("=" * 72)

    for channel, info in channels.items():
        filtered = filter_emg(raw_data[channel], fs)
        activity = activity_signal(filtered, fs)

        filtered_data[channel] = filtered
        activity_data[channel] = activity

        # Only calculate the baseline values from the 10-30 s rest section.
        rest_activity = activity[baseline_start:baseline_end]
        rest_filtered = filtered[baseline_start:baseline_end]

        mean = float(np.mean(rest_activity))
        std = float(np.std(rest_activity))
        filtered_std = float(np.std(rest_filtered))

        k = float(CHANNEL_K[channel])
        threshold = mean + k * std
        reset_threshold = mean + REARM_FRACTION * (threshold - mean)

        config["channels"][channel] = {
            "header_name": info["header_name"],
            "numeric_column": info["numeric_column"],
            "k": k,
            "activity_mean": mean,
            "activity_std": std,
            "threshold": threshold,
            "reset_threshold": reset_threshold,
            "filtered_std": filtered_std,
        }

        print(
            f"{channel}: mean={mean:.6f}, std={std:.6f}, k={k:g}, "
            f"threshold={threshold:.6f}, reset={reset_threshold:.6f}"
        )

    # Save the calibration values for the other programs to use.
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nSaved: {CONFIG_FILE}")

    # Create the time axis once and use it for both plots.
    first_channel = list(channels.keys())[0]
    time = np.arange(len(filtered_data[first_channel])) / fs

    plt.figure(figsize=(16, 8))

    for channel in channels:
        plt.plot(time, filtered_data[channel], linewidth=1.0, label=channel)

    plt.axvline(IGNORE_SECONDS, linestyle="--", label="Baseline starts")
    plt.axvline(IGNORE_SECONDS + BASELINE_SECONDS, linestyle="--", label="Baseline ends")
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered EMG")
    plt.title("Centred filtered EMG used for calibration")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(16, 8))

    for channel in channels:
        plt.plot(time, activity_data[channel], linewidth=1.1, label=f"{channel} activity")
        plt.axhline(config["channels"][channel]["threshold"], linestyle="--", label=f"{channel} trigger")
        plt.axhline(config["channels"][channel]["reset_threshold"], linestyle=":", label=f"{channel} reset")

    plt.xlabel("Time (s)")
    plt.ylabel("Moving |EMG| integral")
    plt.title("Reference calibration thresholds")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
