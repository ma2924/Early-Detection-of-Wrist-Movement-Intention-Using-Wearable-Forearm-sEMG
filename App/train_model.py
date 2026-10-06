"""
Current gesture-training pipeline.

Trigger and re-arm thresholds are recalculated from the 3-10 s resting
period of each training recording rather than using the fixed thresholds
stored in baseline_config.json.
"""

from pathlib import Path
import csv
import json
import re
import tkinter as tk
from tkinter import filedialog

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project files

PROJECT_DIR = Path(__file__).resolve().parent

RECORDINGS_DIR = PROJECT_DIR / "recordings"
CONFIG_FILE = PROJECT_DIR / "baseline_config.json"
MODEL_FILE = PROJECT_DIR / "gesture_model.joblib"
FEATURES_FILE = PROJECT_DIR / "training_features.csv"


# Gestures the model will learn
GESTURES = ["Open", "Point", "Fist", "Pinch", "Spoon",]


# Timing

# 0-3 s is ignored
# 3-10 s is used as rest calibration
# movement detection starts at 10 s
CALIBRATION_START_SECONDS = 3.0
CALIBRATION_END_SECONDS = 10.0
DETECTION_START_SECONDS = 10.0

# Kept in the saved model information for compatibility
RECORDING_IGNORE_SECONDS = DETECTION_START_SECONDS

EXPECTED_REPS_PER_FILE = 20

# Features are taken from the first 300 ms after a movement is detected
FEATURE_WINDOW_MS = 300.0

# Signal must stay above threshold for 30 ms to count as a real trigger
PERSISTENCE_MS = 30.0

# Triggering channel must stay below its re-arm threshold for 150 ms
GLOBAL_REARM_MS = 150.0


# Threshold settings

DEFAULT_TRIGGER_K = 4.0
DEFAULT_REARM_K = 2.0

# Trigger threshold = mean + K * standard deviation
# Experimental per-channel threshold multipliers.
# These values were tuned for the development recordings and are not
# intended to be universal physiological thresholds.
TRIGGER_K_BY_CHANNEL = {
    "CH1": 4.0,
    "CH2": 4.0,
    "CH3": 10.0,
    "CH4": 4.0,
    "CH5": 4.0,
    "CH6": 4.0,
    "CH7": 4.0,
    "CH8": 4.0,
}

# Re-arm threshold = mean + K * standard deviation
REARM_K_BY_CHANNEL = {
    "CH1": 2.0,
    "CH2": 2.0,
    "CH3": 1.0,
    "CH4": 2.0,
    "CH5": 2.0,
    "CH6": 2.0,
    "CH7": 2.0,
    "CH8": 2.0,
}


CHANNEL_COLOURS = {
    "CH1": "#1f77b4",
    "CH2": "#ff7f0e",
    "CH3": "#2ca02c",
    "CH4": "#d62728",
    "CH5": "#9467bd",
    "CH6": "#8c564b",
    "CH7": "#e377c2",
    "CH8": "#17becf",
}

PREVIEW_EACH_FILE = True

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 14,
})


# Loading files

def split_line(line):
    """Split a CSV, tab-separated or semicolon-separated row."""
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


def load_config():
    """Load baseline_config.json and check that the needed information exists."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            "\nCould not find baseline_config.json.\n\n"
            f"Expected location:\n{CONFIG_FILE}\n\n"
            "Run calibrate_baseline.py first, and make sure "
            "baseline_config.json is saved in the App folder."
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    required_items = [
        "sample_rate",
        "filter",
        "activity_window_ms",
        "feature_noise_multiplier",
        "channels",
    ]

    missing_items = []

    for item in required_items:
        if item not in config:
            missing_items.append(item)

    if missing_items:
        missing_items.sort()
        raise ValueError("baseline_config.json is missing: " + ", ".join(missing_items))

    if not config["channels"]:
        raise ValueError("No channels are stored in baseline_config.json.")

    return config


def choose_training_files(gesture):
    """Open a file picker for one gesture."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    root.update()

    selected = filedialog.askopenfilenames(
        parent=root,
        title=f"Select {gesture} TRAINING recording(s)",
        initialdir=str(RECORDINGS_DIR),
        filetypes=[
            ("OpenBCI recordings", "*.txt *.csv"),
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    selected_files = []

    for file_name in selected:
        selected_files.append(Path(file_name))

    return selected_files


def read_openbci_file(file_path, config):
    """Read the EMG channels from an OpenBCI recording."""
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore",) as file:
        lines = file.readlines()

    if not lines:
        raise ValueError(f"{file_path.name} is empty.")

    expected_fs = float(config["sample_rate"])
    fs = expected_fs

    # Use the sample rate written in the recording if it is available
    for line in lines:
        match = re.search(r"Sample\s*Rate\s*=\s*([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE,)

        if match:
            fs = float(match.group(1))
            break

    if abs(fs - expected_fs) > 1e-9:
        raise ValueError(
            f"{file_path.name}: recording is {fs:g} Hz, "
            f"but calibration is {expected_fs:g} Hz."
        )

    channels = config["channels"]

    # Find the row containing the EXG (EMG) column names
    header = None
    header_index = None

    for index, line in enumerate(lines):
        if "EXG Channel" not in line:
            continue

        possible_header = split_line(line)
        has_exg_column = False

        for item in possible_header:
            if "EXG Channel" in item:
                has_exg_column = True
                break

        if has_exg_column:
            header = possible_header
            header_index = index
            break

    data = {}

    for channel in channels:
        data[channel] = []

    # Normal case: use the column names in the file
    if header_index is not None:
        column_indices = {}

        for channel, info in channels.items():
            wanted = info["header_name"]

            if wanted not in header:
                raise ValueError(f"{file_path.name}: missing column {wanted}")

            column_indices[channel] = header.index(wanted)

        for line in lines[header_index + 1:]:
            stripped = line.strip()

            if not stripped or stripped.startswith("%"):
                continue

            row = split_line(stripped)
            values = {}

            try:
                for channel, column in column_indices.items():
                    values[channel] = float(row[column])
            except (ValueError, IndexError):
                continue

            for channel, value in values.items():
                data[channel].append(value)

    # Backup case: use the numeric column positions stored in the config
    else:
        max_column = 0

        for info in channels.values():
            column = int(info["numeric_column"])

            if column > max_column:
                max_column = column

        valid_rows = 0

        for line in lines:
            stripped = line.strip()

            if not stripped or stripped.startswith("%"):
                continue

            row = split_line(stripped)

            if len(row) <= max_column:
                continue

            if not is_number(row[0]):
                continue

            values = {}

            try:
                for channel, info in channels.items():
                    column = int(info["numeric_column"])
                    values[channel] = float(row[column])
            except (ValueError, IndexError):
                continue

            for channel, value in values.items():
                data[channel].append(value)

            valid_rows += 1

        if valid_rows == 0:
            raise ValueError(f"{file_path.name}: could not recognise OpenBCI format.")

    # Convert lists to NumPy arrays
    for channel in data:
        data[channel] = np.asarray(data[channel], dtype=float)

        if data[channel].size == 0:
            raise ValueError(f"{file_path.name}: no usable samples for {channel}.")

    # Keep every channel the same length
    shortest = None

    for values in data.values():
        if shortest is None or len(values) < shortest:
            shortest = len(values)

    for channel in data:
        data[channel] = data[channel][:shortest]

    return fs, data


# Signal processing

def filter_emg(values, fs, config):
    """Apply the same filtering used during calibration."""
    filter_settings = config["filter"]

    highpass_hz = float(filter_settings["highpass_hz"])
    lowpass_hz = float(filter_settings["lowpass_hz"])
    notch_hz = float(filter_settings["notch_hz"])
    notch_q = float(filter_settings["notch_q"])
    filter_order = int(filter_settings["filter_order"])

    filtered = np.asarray(values, dtype=float)

    # Centre around zero
    filtered = filtered - np.mean(filtered)

    nyquist = fs / 2.0

    if notch_hz >= nyquist:
        raise ValueError(f"Cannot use {notch_hz:g} Hz notch at fs={fs:g} Hz.")

    actual_lowpass = min(lowpass_hz, nyquist * 0.90)

    # Remove mains noise
    b_notch, a_notch = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=fs,)

    filtered = signal.lfilter(b_notch, a_notch, filtered,)

    # Band-pass filter
    sos = signal.butter(filter_order, [highpass_hz, actual_lowpass], btype="bandpass", fs=fs, output="sos",)

    filtered = signal.sosfilt(sos, filtered,)

    return filtered


def calculate_activity(filtered_emg, fs, config):
    """Calculate the moving absolute-EMG activity signal."""
    activity_window_ms = float(config["activity_window_ms"])

    window_samples = round(activity_window_ms * fs / 1000.0)

    window_samples = max(1, window_samples)

    rectified = np.abs(filtered_emg)
    kernel = np.ones(window_samples, dtype=float) / fs

    activity = np.convolve(rectified, kernel, mode="full",)

    return activity[:len(rectified)]


def process_recording(file_path, config):
    """Load, filter and calculate activity for every enabled channel."""
    fs, raw_data = read_openbci_file(file_path, config,)

    processed = {}

    for channel in config["channels"]:
        filtered = filter_emg(raw_data[channel], fs, config,)

        activity = calculate_activity(filtered, fs, config,)

        processed[channel] = {"filtered": filtered, "activity": activity,}

    return fs, processed


# Per-recording thresholds

def get_channel_colour(channel, channel_index=0):
    fallback_colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]

    if channel in CHANNEL_COLOURS:
        return CHANNEL_COLOURS[channel]

    colour_index = channel_index % len(fallback_colours)
    return fallback_colours[colour_index]


def get_trigger_k(channel):
    if channel in TRIGGER_K_BY_CHANNEL:
        return float(TRIGGER_K_BY_CHANNEL[channel])

    return float(DEFAULT_TRIGGER_K)


def get_rearm_k(channel):
    if channel in REARM_K_BY_CHANNEL:
        return float(REARM_K_BY_CHANNEL[channel])

    return float(DEFAULT_REARM_K)


def calculate_recording_baseline(processed, fs, config):
    """Calculate the 3-10 s rest baseline for this recording."""
    channels = list(config["channels"].keys())

    start_index = round(CALIBRATION_START_SECONDS * fs)

    if start_index < 0:
        start_index = 0

    end_index = round(CALIBRATION_END_SECONDS * fs)

    if end_index < start_index + 1:
        end_index = start_index + 1

    calibration = {}

    for channel in channels:
        activity = processed[channel]["activity"]

        if len(activity) <= start_index:
            raise ValueError(
                f"{channel}: recording is too short for "
                f"{CALIBRATION_START_SECONDS:g}-"
                f"{CALIBRATION_END_SECONDS:g} s calibration."
            )

        actual_end = min(end_index, len(activity))

        values = np.asarray(activity[start_index:actual_end], dtype=float,)

        if values.size < 2:
            raise ValueError(f"{channel}: not enough samples in the " "recording-local baseline window.")

        mean = float(np.mean(values))
        std = float(np.std(values))
        variance = float(np.var(values))

        trigger_k = get_trigger_k(channel)
        rearm_k = get_rearm_k(channel)

        trigger_threshold = mean + trigger_k * std
        rearm_threshold = mean + rearm_k * std

        calibration[channel] = {
            "mean": mean,
            "std": std,
            "variance": variance,
            "trigger_k": trigger_k,
            "rearm_k": rearm_k,
            "trigger_threshold": float(trigger_threshold),
            "rearm_threshold": float(rearm_threshold),
        }

    return calibration


def print_recording_baseline(calibration):
    """Print the thresholds so they can be checked before training."""
    print(
        "\nRecording-local baseline "
        f"({CALIBRATION_START_SECONDS:g}-"
        f"{CALIBRATION_END_SECONDS:g} s)"
    )

    print("Channel | Mean | SD | Ktrig | Trigger | Krearm | Re-arm")

    print("-" * 76)

    for channel, values in calibration.items():
        print(
            f"{channel:>7} | "
            f"{values['mean']:.4f} | "
            f"{values['std']:.4f} | "
            f"{values['trigger_k']:.2f} | "
            f"{values['trigger_threshold']:.4f} | "
            f"{values['rearm_k']:.2f} | "
            f"{values['rearm_threshold']:.4f}"
        )

    print()


def get_trigger_threshold(calibration, channel):
    return float(calibration[channel]["trigger_threshold"])


def get_reset_threshold(calibration, channel):
    return float(calibration[channel]["rearm_threshold"])


# Finding movement events

def find_first_persistent_crossing(activity, threshold, start_index, end_index, persistence_samples,):
    """Find the first place the signal stays above threshold long enough."""
    above_count = 0
    stop_index = min(end_index, len(activity))

    for index in range(start_index, stop_index):
        if activity[index] >= threshold:
            above_count += 1

            if above_count >= persistence_samples:
                onset = index - persistence_samples + 1
                return onset
        else:
            above_count = 0

    return None


def make_zero_counts(channels):
    """Create a counter set to zero for each channel."""
    counts = {}

    for channel in channels:
        counts[channel] = 0

    return counts


def detect_gesture_events(processed, fs, config):
    """Find each movement and record when each channel activates."""
    channels = list(config["channels"].keys())

    calibration = calculate_recording_baseline(processed, fs, config,)

    print_recording_baseline(calibration)

    persistence_samples = round(PERSISTENCE_MS * fs / 1000.0)
    persistence_samples = max(1, persistence_samples)

    rearm_samples = round(GLOBAL_REARM_MS * fs / 1000.0)
    rearm_samples = max(1, rearm_samples)

    feature_samples = round(FEATURE_WINDOW_MS * fs / 1000.0)
    feature_samples = max(1, feature_samples)

    detection_start_index = round(DETECTION_START_SECONDS * fs)
    detection_start_index = max(0, detection_start_index)

    sample_count = None

    for channel in channels:
        channel_length = len(processed[channel]["activity"])

        if sample_count is None or channel_length < sample_count:
            sample_count = channel_length

    above_counts = make_zero_counts(channels)

    events = []

    # The detector starts ready to find a movement
    armed = True

    # These are only used after a movement has triggered
    trigger_channel = None
    trigger_below_reset_count = 0
    current_event = None

    # Re-arming cannot begin until the 300 ms feature window has ended
    lock_until = detection_start_index

    index = detection_start_index

    while index < sample_count:

        # -------------------------------------------------------------
        # ARMED: look for the next movement
        # -------------------------------------------------------------
        if armed:
            onset_candidates = []

            for channel in channels:
                activity_value = processed[channel]["activity"][index]

                trigger_threshold = get_trigger_threshold(calibration, channel,)

                if activity_value >= trigger_threshold:
                    above_counts[channel] += 1

                    if above_counts[channel] >= persistence_samples:
                        candidate_onset = (index - persistence_samples + 1)

                        onset_candidates.append((candidate_onset, channel))

                else:
                    above_counts[channel] = 0

            if onset_candidates:
                # Find the earliest valid channel crossing.
                # Keeping the first one also matches the old code if two tie.
                onset = None
                trigger_channel = None

                for candidate_onset, candidate_channel in onset_candidates:
                    if onset is None or candidate_onset < onset:
                        onset = candidate_onset
                        trigger_channel = candidate_channel

                feature_window_end = min(onset + feature_samples, sample_count,)

                # Find when every channel first activates inside this window
                activations = {}

                for channel in channels:
                    threshold = get_trigger_threshold(calibration, channel,)

                    activation_index = find_first_persistent_crossing(
                        processed[channel]["activity"],
                        threshold,
                        onset,
                        feature_window_end,
                        persistence_samples,
                    )

                    activations[channel] = activation_index

                current_event = {
                    "onset": onset,
                    "window_end": feature_window_end,
                    "activations": activations,
                    "trigger_channel": trigger_channel,
                    "reset_crossing": None,
                    "rearm": None,
                    "min_activity_after_window": None,
                }

                events.append(current_event)

                print(
                    f"  Event {len(events):>2}: "
                    f"TRIGGER at {onset / fs:.3f} s "
                    f"| {trigger_channel} "
                    f"| threshold="
                    f"{get_trigger_threshold(calibration, trigger_channel):.4f}"
                )

                # Stop looking for another movement until this one has re-armed
                armed = False
                trigger_below_reset_count = 0
                lock_until = feature_window_end
                above_counts = make_zero_counts(channels)


        # DISARMED: wait for the trigger channel to return to rest

        else:
            can_check_rearm = (index >= lock_until and trigger_channel is not None)

            if can_check_rearm:
                trigger_activity = (processed[trigger_channel]["activity"][index])

                rearm_threshold = get_reset_threshold(calibration, trigger_channel,)

                if current_event is not None:
                    old_min = current_event["min_activity_after_window"]

                    if old_min is None or trigger_activity < old_min:
                        current_event["min_activity_after_window"] = float(trigger_activity)

                if trigger_activity <= rearm_threshold:
                    trigger_below_reset_count += 1
                else:
                    # It has to stay below continuously
                    trigger_below_reset_count = 0

                if trigger_below_reset_count >= rearm_samples:
                    reset_crossing = (index - rearm_samples + 1)

                    if current_event is not None:
                        current_event["reset_crossing"] = reset_crossing

                        current_event["rearm"] = index

                    print(
                        "           RE-ARM threshold entered at "
                        f"{reset_crossing / fs:.3f} s "
                        "| armed again at "
                        f"{index / fs:.3f} s"
                    )

                    armed = True
                    trigger_channel = None
                    trigger_below_reset_count = 0
                    current_event = None
                    above_counts = make_zero_counts(channels)

        index += 1

    # Useful warning if one movement prevented the detector from re-arming
    detector_is_stuck = (not armed and trigger_channel is not None and current_event is not None)

    if detector_is_stuck:
        rearm_threshold = get_reset_threshold(calibration, trigger_channel,)

        minimum_seen = current_event["min_activity_after_window"]

        print("\n*** DETECTOR ENDED STILL DISARMED ***")

        print(f"Blocking channel: {trigger_channel}")

        print(f"Re-arm threshold: {rearm_threshold:.6f}")

        if minimum_seen is not None:
            print("Minimum activity after feature window: " f"{minimum_seen:.6f}")

            if minimum_seen > rearm_threshold:
                print("The blocking channel NEVER went below " "its re-arm threshold.")
            else:
                print(
                    "The blocking channel went below the "
                    "re-arm threshold but did not stay there "
                    f"for {GLOBAL_REARM_MS:g} ms continuously."
                )

        print()

    return events


# Feature extraction

def zero_crossings(values, threshold):
    """Count useful zero crossings while ignoring very small noisy changes."""
    count = 0

    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]

        crossed_zero = ((previous > 0 and current < 0) or (previous < 0 and current > 0))

        difference = abs(current - previous)

        large_enough = difference >= threshold

        if crossed_zero and large_enough:
            count += 1

    return float(count)


def slope_sign_changes(values, threshold):
    """Count useful changes in slope while ignoring small noisy changes."""
    count = 0

    for index in range(1, len(values) - 1):
        left_difference = (values[index] - values[index - 1])

        right_difference = (values[index] - values[index + 1])

        slope_changed = (
            (
                left_difference > 0
                and right_difference > 0
            )
            or
            (
                left_difference < 0
                and right_difference < 0
            )
        )

        left_is_large = (abs(left_difference) >= threshold)

        right_is_large = (abs(right_difference) >= threshold)

        large_enough = (left_is_large or right_is_large)

        if slope_changed and large_enough:
            count += 1

    return float(count)


def get_feature_names(config):
    """Build the CSV/model feature names in the same order as the data."""
    names = []

    for channel in config["channels"]:
        channel_names = [
            f"{channel}_MAV",
            f"{channel}_WL",
            f"{channel}_ZC",
            f"{channel}_SSC",
            f"{channel}_activation_flag",
            f"{channel}_activation_latency_ms",
        ]

        names.extend(channel_names)

    return names


def extract_event_features(event, processed, fs, config,):
    """Extract the six features used for each channel."""
    onset = int(event["onset"])

    feature_samples = round(FEATURE_WINDOW_MS * fs / 1000.0)

    feature_samples = max(1, feature_samples,)

    end = onset + feature_samples

    shortest = None

    for channel in config["channels"]:
        channel_length = len(processed[channel]["filtered"])

        if shortest is None or channel_length < shortest:
            shortest = channel_length

    # Ignore an event if its full 300 ms window is not available
    if end > shortest:
        return None

    features = []

    for channel in config["channels"]:
        segment = processed[channel]["filtered"][onset:end]

        # Mean Absolute Value
        absolute_segment = np.abs(segment)
        mav = float(np.mean(absolute_segment))

        # Waveform Length
        sample_differences = np.diff(segment)
        waveform_length = float(np.sum(np.abs(sample_differences)))

        # Noise threshold used by ZC and SSC
        noise_multiplier = float(config["feature_noise_multiplier"])

        filtered_std = float(config["channels"][channel]["filtered_std"])

        noise_threshold = (noise_multiplier * filtered_std)

        zc = zero_crossings(segment, noise_threshold,)

        ssc = slope_sign_changes(segment, noise_threshold,)

        # Activation timing for this channel
        activation_index = event["activations"][channel]

        if activation_index is None:
            activation_flag = 0.0
            activation_latency_ms = FEATURE_WINDOW_MS
        else:
            activation_flag = 1.0

            activation_delay_samples = (activation_index - onset)

            activation_latency_ms = (activation_delay_samples * 1000.0 / fs)

        channel_features = [mav, waveform_length, zc, ssc, activation_flag, float(activation_latency_ms),]

        features.extend(channel_features)

    return features


# Preview plots

def preview_recording(file_path, processed, events, fs, config, gesture,):
    """Show the recording and detector thresholds before accepting it."""
    channels = list(config["channels"].keys())

    calibration = calculate_recording_baseline(processed, fs, config,)

    sample_count = None

    for channel in channels:
        channel_length = len(processed[channel]["filtered"])

        if sample_count is None or channel_length < sample_count:
            sample_count = channel_length

    time_axis = (np.arange(sample_count) / fs)


    # Figure 1: filtered EMG and movement events

    fig1, ax1 = plt.subplots(figsize=(18, 8))

    for channel_index, channel in enumerate(channels):
        colour = get_channel_colour(channel, channel_index,)

        ax1.plot(
            time_axis,
            processed[channel]["filtered"][:sample_count],
            linewidth=0.9,
            color=colour,
            alpha=0.75,
            label=channel,
        )

    used_trigger_labels = set()
    used_rearm_labels = set()

    for event in events:
        trigger_channel = event["trigger_channel"]

        channel_index = channels.index(trigger_channel)

        colour = get_channel_colour(trigger_channel, channel_index,)

        # Solid line = trigger
        trigger_label = None

        if trigger_channel not in used_trigger_labels:
            trigger_label = (f"{trigger_channel} trigger")

        ax1.axvline(
            event["onset"] / fs,
            color=colour,
            linestyle="-",
            linewidth=3.2,
            alpha=0.95,
            label=trigger_label,
        )

        used_trigger_labels.add(trigger_channel)

        # Dashed line = detector re-armed
        if event.get("rearm") is not None:
            rearm_label = None

            if trigger_channel not in used_rearm_labels:
                rearm_label = (f"{trigger_channel} re-arm")

            ax1.axvline(
                event["rearm"] / fs,
                color=colour,
                linestyle="--",
                linewidth=3.2,
                alpha=0.95,
                label=rearm_label,
            )

            used_rearm_labels.add(trigger_channel)

    ax1.axvspan(
        CALIBRATION_START_SECONDS,
        CALIBRATION_END_SECONDS,
        color="gray",
        alpha=0.10,
        label="Rest calibration",
    )

    ax1.axvline(
        DETECTION_START_SECONDS,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Detection starts",
    )

    ax1.set_xlabel("Time (s)")

    ax1.set_ylabel("Filtered EMG")

    ax1.set_title(
        f"TRAIN {gesture}: filtered EMG + detector events\n"
        f"{file_path.name} | detected {len(events)} events"
    )

    ax1.legend(loc="upper right", ncol=2,)

    ax1.grid(alpha=0.12)

    fig1.tight_layout()


    # Figure 2: activity and thresholds for each channel

    plot_height = max(9, 3.4 * len(channels),)

    fig2, axes = plt.subplots(len(channels), 1, figsize=(19, plot_height), sharex=True,)

    if len(channels) == 1:
        axes = [axes]

    for channel_index, channel in enumerate(channels):
        ax = axes[channel_index]

        colour = get_channel_colour(channel, channel_index,)

        activity = processed[channel]["activity"][:sample_count]

        stats = calibration[channel]

        trigger_threshold = stats["trigger_threshold"]

        rearm_threshold = stats["rearm_threshold"]

        ax.plot(time_axis, activity, color=colour, linewidth=1.5, alpha=0.90, label=f"{channel} activity",)

        trigger_text = (f"Trigger = μ + " f"{stats['trigger_k']:.1f}σ " f"= {trigger_threshold:.4g}")

        ax.axhline(
            trigger_threshold,
            color=colour,
            linestyle="-",
            linewidth=3.0,
            alpha=0.95,
            label=trigger_text,
        )

        rearm_text = (f"Re-arm = μ + " f"{stats['rearm_k']:.1f}σ " f"= {rearm_threshold:.4g}")

        ax.axhline(
            rearm_threshold,
            color=colour,
            linestyle="--",
            linewidth=3.0,
            alpha=0.95,
            label=rearm_text,
        )

        ax.axvspan(CALIBRATION_START_SECONDS, CALIBRATION_END_SECONDS, color="gray", alpha=0.08,)

        ax.axvline(DETECTION_START_SECONDS, color="black", linestyle=":", linewidth=1.5, alpha=0.65,)

        # Only show event lines on the channel which caused the event
        for event in events:
            if event["trigger_channel"] != channel:
                continue

            ax.axvline(event["onset"] / fs, color=colour, linestyle="-", linewidth=4.2, alpha=0.85,)

            if event.get("rearm") is not None:
                ax.axvline(event["rearm"] / fs, color=colour, linestyle="--", linewidth=4.2, alpha=0.85,)

        ax.set_ylabel(channel, fontweight="bold",)

        ax.set_title(f"μ={stats['mean']:.4f}, " f"σ={stats['std']:.4f}", loc="left", fontsize=13,)

        ax.legend(loc="upper right", ncol=3,)

        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s)")

    fig2.suptitle(
        f"TRAIN {gesture}: recording-local threshold diagnosis\n"
        "3-10 s = rest calibration | "
        "SOLID = trigger | DASHED = re-arm",
        fontsize=20,
        fontweight="bold",
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


# Saving features and training the model

def save_feature_table(feature_rows, labels, feature_names,):
    """Save the extracted training features to a CSV file."""
    with open(FEATURES_FILE, "w", newline="", encoding="utf-8",) as file:
        writer = csv.writer(file)

        writer.writerow(feature_names + ["Gesture"])

        for features, gesture in zip(feature_rows, labels,):
            row = list(features)
            row.append(gesture)
            writer.writerow(row)


def main():
    config = load_config()

    print("\nLogistic Regression gesture trainer")

    print("=" * 70)

    print(f"Project folder:\n{PROJECT_DIR}")

    print(f"\nBaseline file:\n{CONFIG_FILE}")

    enabled_channels = ", ".join(config["channels"].keys())

    print("\nEnabled channels: " + enabled_channels)

    print(f"\nFirst {CALIBRATION_START_SECONDS:g} s " "of every training recording are ignored.")

    print(
        f"Seconds {CALIBRATION_START_SECONDS:g}-"
        f"{CALIBRATION_END_SECONDS:g} are used as "
        "recording-local REST calibration."
    )

    print("Movement detection starts at " f"{DETECTION_START_SECONDS:g} s.")

    print("Expected repetitions per recording: " f"{EXPECTED_REPS_PER_FILE}")

    print("Feature window after first trigger: " f"{FEATURE_WINDOW_MS:g} ms")

    print("Trigger-channel re-arm hold: " f"{GLOBAL_REARM_MS:g} ms")

    print("\nRest is treated as NO TRIGGER, " "so it is not an ML gesture class.")

    feature_names = get_feature_names(config)

    feature_rows = []
    labels = []

    # Ask for recordings one gesture at a time
    for gesture in GESTURES:
        print(f"\n--- {gesture.upper()} ---")

        selected_files = choose_training_files(gesture)

        if not selected_files:
            print(f"No {gesture} file selected. " "Skipping this gesture.")

            continue

        for file_path in selected_files:
            fs, processed = process_recording(file_path, config,)

            events = detect_gesture_events(processed, fs, config,)

            print(f"\n{gesture} | {file_path.name}")

            print("Detected events: " f"{len(events)} / " f"expected about {EXPECTED_REPS_PER_FILE}")

            if PREVIEW_EACH_FILE:
                preview_recording(file_path, processed, events, fs, config, gesture,)

            answer = input(f"\nUse these {len(events)} events " f"for {gesture}? [Y/n]: ")

            answer = answer.strip().lower()

            if answer == "n" or answer == "no":
                print("Recording rejected.")

                continue

            accepted = 0

            for event in events:
                features = extract_event_features(event, processed, fs, config,)

                if features is not None:
                    feature_rows.append(features)

                    labels.append(gesture)

                    accepted += 1

            print("Accepted complete feature windows: " f"{accepted}")

    if not feature_rows:
        raise SystemExit("No training events were accepted.")

    # Work out which gesture classes were actually supplied
    unique_labels = list(set(labels))

    unique_labels.sort()

    if len(unique_labels) < 2:
        raise SystemExit("At least two gesture classes " "are required to train Logistic Regression.")

    X = np.asarray(feature_rows, dtype=float,)

    y = np.asarray(labels,)

    save_feature_table(X, y, feature_names,)

    # StandardScaler puts all features onto a similar scale.
    # LogisticRegression is the actual gesture classifier.
    scaler = StandardScaler()

    classifier = LogisticRegression(max_iter=5000, class_weight="balanced",)

    model = Pipeline([("scaler", scaler), ("classifier", classifier),])

    print("\nTraining Logistic Regression...")

    model.fit(X, y,)

    # This is only a check of how well the model fits its training data.
    training_predictions = model.predict(X)

    correct_predictions = (training_predictions == y)

    training_accuracy = (100.0 * float(np.mean(correct_predictions)))

    # Save the model together with the settings needed later
    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "gestures": unique_labels,
        "sample_rate": float(config["sample_rate"]),
        "channels": list(config["channels"].keys()),
        "feature_window_ms": FEATURE_WINDOW_MS,
        "persistence_ms": PERSISTENCE_MS,
        "global_rearm_ms": GLOBAL_REARM_MS,
        "rearm_rule": "trigger_channel_only",
        "baseline_mode": "per_recording",
        "calibration_start_seconds": CALIBRATION_START_SECONDS,
        "calibration_end_seconds": CALIBRATION_END_SECONDS,
        "detection_start_seconds": DETECTION_START_SECONDS,
        "trigger_k_by_channel": dict(TRIGGER_K_BY_CHANNEL),
        "rearm_k_by_channel": dict(REARM_K_BY_CHANNEL),
        "recording_ignore_seconds": RECORDING_IGNORE_SECONDS,
    }

    joblib.dump(model_bundle, MODEL_FILE,)

    print("\nTRAINING COMPLETE")

    print("=" * 70)

    print("Total training examples: " f"{len(y)}")

    for gesture in unique_labels:
        gesture_count = 0

        for label in y:
            if label == gesture:
                gesture_count += 1

        print(f"{gesture}: {gesture_count}")

    print("\nFeatures per movement: " f"{X.shape[1]}")

    print("Training-set fit accuracy: " f"{training_accuracy:.1f}%")

    print("\nDo NOT use that training accuracy " "as your final result.")

    print("Use test_model.py with NEW recordings " "for the real test accuracy.")

    print(f"\nSaved model:\n{MODEL_FILE}")

    print(f"\nSaved extracted features:\n{FEATURES_FILE}")


if __name__ == "__main__":
    main()
