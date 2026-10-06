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
MODEL_FILE = PROJECT_DIR / "gesture_model_saved_thresholds.joblib"
FEATURES_FILE = PROJECT_DIR / "training_features_saved_thresholds.csv"

# Detection settings

# Thresholds come from baseline_config.json.
# The first 10 s of each training recording are ignored.

RECORDING_IGNORE_SECONDS = 10.0
DETECTION_START_SECONDS = RECORDING_IGNORE_SECONDS

# Aim for one event per movement
EXPECTED_REPS_PER_FILE = 20

# Feature window starts at the first trigger
FEATURE_WINDOW_MS = 300.0

# Signal must stay above threshold for this long to count
PERSISTENCE_MS = 30.0

# The triggering channel controls re-arming
GLOBAL_REARM_MS = 150.0


# Channel colours
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


# Plot settings

plt.rcParams.update(
    {
        "font.size": 17,
        "axes.titlesize": 21,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 14,
    }
)


# File loading

def split_line(line):
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
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            "\nCould not find baseline_config.json.\n\n"
            f"Expected location:\n{CONFIG_FILE}\n\n"
            "Run calibrate_baseline.py first, and make sure "
            "baseline_config.json is saved in the App folder."
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    required_top_level = {
        "sample_rate",
        "filter",
        "activity_window_ms",
        "feature_noise_multiplier",
        "channels",
    }

    missing = required_top_level.difference(config.keys())

    if missing:
        raise ValueError("baseline_config.json is missing: " + ", ".join(sorted(missing)))

    if not config["channels"]:
        raise ValueError("No channels are stored in baseline_config.json.")

    return config


def choose_training_files(gesture):
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
    for path in selected:
        selected_files.append(Path(path))
    return selected_files


def read_openbci_file(file_path, config):
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore",) as file:
        lines = file.readlines()

    if not lines:
        raise ValueError(f"{file_path.name} is empty.")

    expected_fs = float(config["sample_rate"])
    fs = expected_fs

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

    header_index = None
    header = None

    for index, line in enumerate(lines):
        if "EXG Channel" in line:
            possible_header = split_line(line)

            if any("EXG Channel" in item for item in possible_header):
                header_index = index
                header = possible_header
                break

    data = {}
    for channel in channels:
        data[channel] = []

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

            try:
                values = {}
                for channel, column in column_indices.items():
                    values[channel] = float(row[column])
            except (ValueError, IndexError):
                continue

            for channel, value in values.items():
                data[channel].append(value)

    else:
        max_column = max(int(info["numeric_column"]) for info in channels.values())

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

            try:
                values = {}
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

    for channel in data:
        data[channel] = np.asarray(data[channel], dtype=float,)

        if data[channel].size == 0:
            raise ValueError(f"{file_path.name}: no usable samples for {channel}.")

    shortest = None
    for values in data.values():
        if shortest is None or len(values) < shortest:
            shortest = len(values)

    for channel in data:
        data[channel] = data[channel][:shortest]

    return fs, data


# Signal processing

def filter_emg(values, fs, config):
    filter_settings = config["filter"]

    highpass_hz = float(filter_settings["highpass_hz"])
    lowpass_hz = float(filter_settings["lowpass_hz"])
    notch_hz = float(filter_settings["notch_hz"])
    notch_q = float(filter_settings["notch_q"])
    filter_order = int(filter_settings["filter_order"])

    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)

    nyquist = fs / 2.0

    if notch_hz >= nyquist:
        raise ValueError(f"Cannot use {notch_hz:g} Hz notch at fs={fs:g} Hz.")

    actual_lowpass = min(lowpass_hz, nyquist * 0.90,)

    b_notch, a_notch = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=fs,)

    filtered = signal.lfilter(b_notch, a_notch, values,)

    sos = signal.butter(filter_order, [highpass_hz, actual_lowpass], btype="bandpass", fs=fs, output="sos",)

    return signal.sosfilt(sos, filtered,)


def calculate_activity(filtered_emg, fs, config):
    activity_window_ms = float(config["activity_window_ms"])

    window_samples = max(1, round(activity_window_ms * fs / 1000.0),)

    rectified = np.abs(filtered_emg)

    kernel = np.ones(window_samples, dtype=float,) / fs

    activity = np.convolve(rectified, kernel, mode="full",)

    return activity[:len(rectified)]


def process_recording(file_path, config):
    fs, raw_data = read_openbci_file(file_path, config,)

    processed = {}

    for channel in config["channels"]:
        filtered = filter_emg(raw_data[channel], fs, config,)

        activity = calculate_activity(filtered, fs, config,)

        processed[channel] = {"filtered": filtered, "activity": activity,}

    return fs, processed


# Thresholds

def get_channel_colour(channel, channel_index=0):
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",]

    return CHANNEL_COLOURS.get(channel, fallback[channel_index % len(fallback)],)


def get_trigger_threshold(config, channel,):
    return float(config["channels"][channel]["threshold"])


def get_reset_threshold(config, channel,):
    return float(config["channels"][channel]["reset_threshold"])


def print_saved_thresholds(config,):
    print("\nThresholds loaded from baseline_config.json")

    print("Channel | Trigger | Re-arm")

    print("-" * 36)

    for channel in config["channels"]:

        print(
            f"{channel:>7} | "
            f"{get_trigger_threshold(config, channel):.6f} | "
            f"{get_reset_threshold(config, channel):.6f}"
        )

    print()


# Movement detection

def find_first_persistent_crossing(activity, threshold, start_index, end_index, persistence_samples,):
    above_count = 0

    for index in range(start_index, min(end_index, len(activity)),):
        if activity[index] >= threshold:
            above_count += 1

            if above_count >= persistence_samples:
                return (index - persistence_samples + 1)
        else:
            above_count = 0

    return None


def detect_gesture_events(processed, fs, config,):
    channels = list(config["channels"].keys())

    print_saved_thresholds(config)

    persistence_samples = max(1, round(PERSISTENCE_MS * fs / 1000.0),)

    rearm_samples = max(1, round(GLOBAL_REARM_MS * fs / 1000.0),)

    feature_samples = max(1, round(FEATURE_WINDOW_MS * fs / 1000.0),)

    detection_start_index = max(0, round(DETECTION_START_SECONDS * fs),)

    sample_count = None
    for channel in channels:
        channel_length = len(processed[channel]["activity"])
        if sample_count is None or channel_length < sample_count:
            sample_count = channel_length

    above_counts = {}
    for channel in channels:
        above_counts[channel] = 0

    events = []

    armed = True
    trigger_channel = None
    trigger_below_reset_count = 0
    current_event = None

    # Wait until the 300 ms feature window is over before re-arming
    lock_until = detection_start_index

    index = detection_start_index

    while index < sample_count:

        # ARMED: search for next gesture
        if armed:

            onset_candidates = []

            for channel in channels:

                activity_value = (processed[channel]["activity"][index])

                trigger_threshold = (get_trigger_threshold(config, channel,))

                if activity_value >= trigger_threshold:

                    above_counts[channel] += 1

                    if (above_counts[channel] >= persistence_samples):

                        candidate_onset = (index - persistence_samples + 1)

                        onset_candidates.append((candidate_onset, channel,))

                else:

                    above_counts[channel] = 0

            if onset_candidates:

                # Use the earliest valid channel crossing
                onset = None
                trigger_channel = None

                for candidate_onset, candidate_channel in onset_candidates:
                    if onset is None or candidate_onset < onset:
                        onset = candidate_onset
                        trigger_channel = candidate_channel

                feature_window_end = min(onset + feature_samples, sample_count,)

                activations = {}

                for channel in channels:

                    activations[channel] = (
                        find_first_persistent_crossing(
                            processed[channel][
                                "activity"
                            ],
                            get_trigger_threshold(
                                config,
                                channel,
                            ),
                            onset,
                            feature_window_end,
                            persistence_samples,
                        )
                    )

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
                    f"{get_trigger_threshold(config, trigger_channel):.4f}"
                )

                armed = False
                trigger_below_reset_count = 0
                lock_until = feature_window_end

                above_counts = {}
                for channel in channels:
                    above_counts[channel] = 0

        # DISARMED: wait for trigger channel to return to rest
        else:

            if (index >= lock_until and trigger_channel is not None):

                trigger_activity = (processed[trigger_channel]["activity"][index])

                rearm_threshold = (get_reset_threshold(config, trigger_channel,))

                if current_event is not None:

                    old_min = current_event["min_activity_after_window"]

                    if (old_min is None or trigger_activity < old_min):

                        current_event["min_activity_after_window"] = float(trigger_activity)

                if trigger_activity <= rearm_threshold:

                    trigger_below_reset_count += 1

                else:

                    # It has to stay below continuously
                    trigger_below_reset_count = 0

                if (trigger_below_reset_count >= rearm_samples):

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

                    above_counts = {}
                    for channel in channels:
                        above_counts[channel] = 0

        index += 1

    # Warn if the detector never re-arms
    if (not armed and trigger_channel is not None and current_event is not None):

        rearm_threshold = (get_reset_threshold(config, trigger_channel,))

        minimum_seen = current_event["min_activity_after_window"]

        print("\n*** DETECTOR ENDED STILL DISARMED ***")

        print(f"Blocking channel: {trigger_channel}")

        print(f"Saved re-arm threshold: " f"{rearm_threshold:.6f}")

        if minimum_seen is not None:

            print("Minimum activity after feature window: " f"{minimum_seen:.6f}")

            if minimum_seen > rearm_threshold:

                print("The blocking channel NEVER went below " "its saved re-arm threshold.")

            else:

                print(
                    "The blocking channel went below the "
                    "saved re-arm threshold but did not stay "
                    f"there for {GLOBAL_REARM_MS:g} ms."
                )

        print()

    return events


# Feature extraction

def zero_crossings(values, threshold):
    count = 0

    for index in range(1, len(values)):
        sign_changed = (
            (
                values[index - 1] > 0
                and values[index] < 0
            )
            or
            (
                values[index - 1] < 0
                and values[index] > 0
            )
        )

        large_enough = (abs(values[index] - values[index - 1]) >= threshold)

        if sign_changed and large_enough:
            count += 1

    return float(count)


def slope_sign_changes(values, threshold):
    count = 0

    for index in range(1, len(values) - 1,):
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

        large_enough = (abs(left_difference) >= threshold or abs(right_difference) >= threshold)

        if slope_changed and large_enough:
            count += 1

    return float(count)


def get_feature_names(config):
    names = []

    for channel in config["channels"]:
        names.extend(
            [
                f"{channel}_MAV",
                f"{channel}_WL",
                f"{channel}_ZC",
                f"{channel}_SSC",
                f"{channel}_activation_flag",
                f"{channel}_activation_latency_ms",
            ]
        )

    return names


def extract_event_features(event, processed, fs, config,):
    onset = int(event["onset"])

    feature_samples = max(1, round(FEATURE_WINDOW_MS * fs / 1000.0),)

    end = onset + feature_samples

    shortest = None
    for channel in config["channels"]:
        channel_length = len(processed[channel]["filtered"])
        if shortest is None or channel_length < shortest:
            shortest = channel_length

    if end > shortest:
        return None

    features = []

    for channel in config["channels"]:
        segment = processed[channel]["filtered"][onset:end]

        mav = float(np.mean(np.abs(segment)))

        waveform_length = float(np.sum(np.abs(np.diff(segment))))

        noise_threshold = (
            float(
                config["feature_noise_multiplier"]
            )
            * float(
                config["channels"][channel]["filtered_std"]
            )
        )

        zc = zero_crossings(segment, noise_threshold,)

        ssc = slope_sign_changes(segment, noise_threshold,)

        activation_index = (event["activations"][channel])

        if activation_index is None:
            activation_flag = 0.0
            activation_latency_ms = (FEATURE_WINDOW_MS)
        else:
            activation_flag = 1.0
            activation_latency_ms = (activation_index - onset) * 1000.0 / fs

        features.extend([mav, waveform_length, zc, ssc, activation_flag, float(activation_latency_ms),])

    return features


# Preview plots

def preview_recording(file_path, processed, events, fs, config, gesture,):
    channels = list(config["channels"].keys())

    sample_count = None
    for channel in channels:
        channel_length = len(processed[channel]["filtered"])
        if sample_count is None or channel_length < sample_count:
            sample_count = channel_length

    time_axis = (np.arange(sample_count) / fs)

    # Filtered EMG and event markers

    fig1, ax1 = plt.subplots(figsize=(18, 8))

    for channel_index, channel in enumerate(channels):

        colour = get_channel_colour(channel, channel_index,)

        ax1.plot(
            time_axis,
            processed[channel]["filtered"][
                :sample_count
            ],
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
        ax1.axvline(
            event["onset"] / fs,
            color=colour,
            linestyle="-",
            linewidth=3.2,
            alpha=0.95,
            label=(
                f"{trigger_channel} trigger"
                if trigger_channel
                not in used_trigger_labels
                else None
            ),
        )

        used_trigger_labels.add(trigger_channel)

        # Dashed line = re-arm
        if event.get("rearm") is not None:

            ax1.axvline(
                event["rearm"] / fs,
                color=colour,
                linestyle="--",
                linewidth=3.2,
                alpha=0.95,
                label=(
                    f"{trigger_channel} re-arm"
                    if trigger_channel
                    not in used_rearm_labels
                    else None
                ),
            )

            used_rearm_labels.add(trigger_channel)

    ax1.axvline(DETECTION_START_SECONDS, color="black", linestyle=":", linewidth=2.0, label="Detection starts",)

    ax1.set_xlabel("Time (s)")

    ax1.set_ylabel("Filtered EMG")

    ax1.set_title(
        f"TRAIN {gesture}: filtered EMG + detector events\n"
        f"{file_path.name} | detected {len(events)} events"
    )

    ax1.legend(loc="upper right", ncol=2,)

    ax1.grid(alpha=0.12)

    fig1.tight_layout()


    # Activity and saved thresholds

    fig2, axes = plt.subplots(len(channels), 1, figsize=(19, max(9, 3.4 * len(channels),),), sharex=True,)

    if len(channels) == 1:
        axes = [axes]

    for channel_index, channel in enumerate(channels):

        ax = axes[channel_index]

        colour = get_channel_colour(channel, channel_index,)

        activity = processed[channel]["activity"][:sample_count]

        trigger_threshold = (get_trigger_threshold(config, channel,))

        rearm_threshold = (get_reset_threshold(config, channel,))

        ax.plot(time_axis, activity, color=colour, linewidth=1.5, alpha=0.90, label=f"{channel} activity",)

        # Solid line = trigger threshold
        ax.axhline(
            trigger_threshold,
            color=colour,
            linestyle="-",
            linewidth=3.0,
            alpha=0.95,
            label=(
                f"Trigger = {trigger_threshold:.4g}"
            ),
        )

        # Dashed line = re-arm threshold
        ax.axhline(
            rearm_threshold,
            color=colour,
            linestyle="--",
            linewidth=3.0,
            alpha=0.95,
            label=(
                f"Re-arm = {rearm_threshold:.4g}"
            ),
        )

        ax.axvline(DETECTION_START_SECONDS, color="black", linestyle=":", linewidth=1.5, alpha=0.65,)

        # Only show event markers on the channel that caused them
        for event in events:

            if (event["trigger_channel"] != channel):
                continue

            ax.axvline(event["onset"] / fs, color=colour, linestyle="-", linewidth=4.2, alpha=0.85,)

            if event.get("rearm") is not None:

                ax.axvline(event["rearm"] / fs, color=colour, linestyle="--", linewidth=4.2, alpha=0.85,)

        ax.set_ylabel(channel, fontweight="bold",)

        ax.legend(loc="upper right", ncol=3,)

        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s)")

    fig2.suptitle(
        f"TRAIN {gesture}: thresholds from baseline_config.json\n"
        f"SOLID = trigger | DASHED = re-arm",
        fontsize=20,
        fontweight="bold",
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


# Model training

def save_feature_table(feature_rows, labels, feature_names,):
    with open(FEATURES_FILE, "w", newline="", encoding="utf-8",) as file:
        writer = csv.writer(file)

        writer.writerow(feature_names + ["Gesture"])

        for features, gesture in zip(feature_rows, labels,):
            writer.writerow(list(features) + [gesture])


def main():
    config = load_config()

    print("\nLogistic Regression gesture trainer")
    print("=" * 70)

    print(f"Project folder:\n{PROJECT_DIR}")

    print(f"\nBaseline file:\n{CONFIG_FILE}")

    print("\nEnabled channels: " + ", ".join(config["channels"].keys()))

    print(f"\nFirst {RECORDING_IGNORE_SECONDS:g} s " "of every training recording are ignored.")

    print("Trigger and re-arm thresholds are loaded " "from baseline_config.json.")

    print(f"Movement detection starts at " f"{DETECTION_START_SECONDS:g} s.")

    print(f"Expected repetitions per recording: " f"{EXPECTED_REPS_PER_FILE}")

    print(f"Feature window after first trigger: " f"{FEATURE_WINDOW_MS:g} ms")

    print(f"Trigger-channel re-arm hold: " f"{GLOBAL_REARM_MS:g} ms")

    print("\nRest is treated as NO TRIGGER, " "so it is not an ML gesture class.")

    feature_names = get_feature_names(config)

    feature_rows = []
    labels = []

    for gesture in GESTURES:
        print(f"\n--- {gesture.upper()} ---")

        selected_files = (choose_training_files(gesture))

        if not selected_files:
            print(f"No {gesture} file selected. " "Skipping this gesture.")
            continue

        for file_path in selected_files:
            fs, processed = (process_recording(file_path, config,))

            events = (detect_gesture_events(processed, fs, config,))

            print(f"\n{gesture} | {file_path.name}")
            print(f"Detected events: " f"{len(events)} / " f"expected about {EXPECTED_REPS_PER_FILE}")

            if PREVIEW_EACH_FILE:
                preview_recording(file_path, processed, events, fs, config, gesture,)

            answer = input(f"\nUse these {len(events)} events " f"for {gesture}? [Y/n]: ").strip().lower()

            if answer in {"n", "no",}:
                print("Recording rejected.")
                continue

            accepted = 0

            for event in events:
                features = (extract_event_features(event, processed, fs, config,))

                if features is not None:
                    feature_rows.append(features)
                    labels.append(gesture)
                    accepted += 1

            print(f"Accepted complete feature windows: " f"{accepted}")

    if not feature_rows:
        raise SystemExit("No training events were accepted.")

    unique_labels = sorted(set(labels))

    if len(unique_labels) < 2:
        raise SystemExit("At least two gesture classes " "are required to train Logistic Regression.")

    X = np.asarray(feature_rows, dtype=float,)

    y = np.asarray(labels,)

    save_feature_table(X, y, feature_names,)

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    model.fit(X, y,)

    training_predictions = (model.predict(X))

    training_accuracy = (100.0 * float(np.mean(training_predictions == y)))

    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "gestures": unique_labels,
        "sample_rate": float(
            config["sample_rate"]
        ),
        "channels": list(
            config["channels"].keys()
        ),
        "feature_window_ms": FEATURE_WINDOW_MS,
        "persistence_ms": PERSISTENCE_MS,
        "global_rearm_ms": GLOBAL_REARM_MS,
        "rearm_rule": "trigger_channel_only",
        "baseline_mode": "baseline_config_json",
        "baseline_config_file": CONFIG_FILE.name,
        "detection_start_seconds": DETECTION_START_SECONDS,
        "recording_ignore_seconds": RECORDING_IGNORE_SECONDS,
    }

    joblib.dump(model_bundle, MODEL_FILE,)

    print("\nTRAINING COMPLETE")
    print("=" * 70)

    print(f"Total training examples: " f"{len(y)}")

    for gesture in unique_labels:
        count = int(np.sum(y == gesture))

        print(f"{gesture}: {count}")

    print(f"\nFeatures per movement: " f"{X.shape[1]}")

    print(f"Training-set fit accuracy: " f"{training_accuracy:.1f}%")

    print("\nDo NOT use that training accuracy " "as your final result.")

    print("Evaluate the model on completely unseen recordings before reporting test accuracy.")

    print(f"\nSaved model:\n{MODEL_FILE}")

    print(f"\nSaved extracted features:\n{FEATURES_FILE}")


if __name__ == "__main__":
    main()
