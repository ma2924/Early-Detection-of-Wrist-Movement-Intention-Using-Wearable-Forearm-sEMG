# Software Development

This document describes the current software pipeline developed for:

**Early Detection of Wrist Movement Intention Using Forearm sEMG**

The software has three main stages:

1. live sEMG acquisition and recording;
2. baseline characterisation and movement-event detection;
3. transient feature extraction and gesture-classifier training.

The main objective is to identify the **early transient period of muscle activation**, rather than waiting until a hand or wrist movement has reached its final position.

---

## 1. Overall Software Architecture

```text
Electrodes
    |
    v
ADS1198
    |
    v
ESP32-S3 filtering
    |
    v
Bluetooth Low Energy
    |
    v
EMGApp.py
    |
    v
Recorded EMG
    |
    +---------------------------+
    |                           |
    v                           v
calibrate_baseline.py      train_model.py
    |                           |
    v                           v
baseline_config.json     per-recording baseline
                                |
                                v
                       movement detection
                                |
                                v
                       300 ms transient window
                                |
                                v
                         feature extraction
                                |
                                v
                        StandardScaler
                                |
                                v
                    Logistic Regression
                                |
                                v
                     gesture_model.joblib
```

---

# 2. Live Acquisition — `EMGApp.py`

`EMGApp.py` is based on the Python acquisition interface developed for the original wearable sEMG project by Abby Finka.

The application searches for an ESP32 BLE device named:

```text
EMG-Logger
```

and receives eight-channel sEMG data from the wearable.

The current acquisition system operates at:

```text
Sampling rate: 1000 Hz
Channels:      8
BLE packet:    15 samples per channel
```

Each BLE packet therefore contains multiple samples, but these are unpacked and processed sequentially by the Python application.

The received channel values are displayed using both:

- live time-domain plots;
- live FFT/frequency-domain plots.

The display maintains approximately two seconds of recent samples for each of the eight channels.

---

## 2.1 ESP32-side Filtering

The main signal filtering is performed on the ESP32 before transmission.

For this reason, additional Python-side real-time filtering is normally disabled:

```python
filtering = False
```

An optional Python band-pass filtering path remains available for development and debugging.

---

## 2.2 Recording Format

When logging is enabled, `EMGApp.py` saves the received data into the `Logs/` directory.

The output format intentionally resembles an OpenBCI recording, including channel headings such as:

```text
EXG Channel 0
EXG Channel 1
...
EXG Channel 7
```

This allows recorded data from the custom wearable to remain compatible with much of the processing code originally developed using OpenBCI Cyton recordings.

---

# 3. Baseline Configuration — `calibrate_baseline.py`

`calibrate_baseline.py` creates the reusable:

```text
baseline_config.json
```

file.

The script accepts OpenBCI-style `.txt` or `.csv` recordings and determines common signal-processing and channel information used elsewhere in the project.

Its original calibration recording structure is:

```text
0–10 s      ignored

10–30 s     resting EMG used for calibration
```

The script applies:

```text
20 Hz high-pass
50 Hz notch
100 Hz low-pass
```

and calculates a moving absolute-EMG activity signal using a **50 ms activity window**.

For each enabled channel it calculates quantities including:

- resting activity mean;
- resting activity standard deviation;
- filtered-signal standard deviation;
- initial threshold information.

These values, together with the signal-processing configuration and channel mappings, are written to `baseline_config.json`.

---

## 3.1 Role of `baseline_config.json` in the Current Pipeline

The current training pipeline no longer uses the stored trigger and reset thresholds from `baseline_config.json` to detect movements.

Instead, movement thresholds are recalculated independently for every training recording.

The configuration file is still used for information including:

- expected sample rate;
- filter settings;
- activity-window duration;
- enabled channels and their recording-column mappings;
- feature-noise multiplier;
- reference filtered-signal noise statistics.

The stored filtered-signal standard deviation is also used when applying noise rejection to Zero Crossing and Slope Sign Change features.

---

# 4. Training Recording Structure

Each training recording used by the current `train_model.py` follows:

```text
0–3 s       startup / ignored

3–10 s      REST calibration

10 s onward gesture repetitions
```

Movement detection begins only after the 10-second point.

Each gesture recording is therefore self-calibrating: its own resting period is used to calculate the thresholds used to detect movements later in that recording.

This reduces dependence on a single threshold set acquired during a separate recording session.

---

# 5. Per-Recording Baseline Calibration

For every selected training recording, the activity signal between **3 and 10 seconds** is used to calculate the resting statistics independently for every channel.

For channel `c`:

```text
μ_c = mean resting activity

σ_c = standard deviation of resting activity
```

The trigger threshold is then:

```text
TH_trigger,c = μ_c + K_trigger,c σ_c
```

and the re-arm threshold is:

```text
TH_rearm,c = μ_c + K_rearm,c σ_c
```

Each channel may use a different value of `K`.

The current values are experimental and are deliberately exposed near the top of the trainer so that they can be adjusted during development.

For example:

```python
TRIGGER_K_BY_CHANNEL = {
    "CH1": 4.0,
    "CH2": 4.0,
    "CH3": 10.0,
    "CH4": 4.0,
}
```

These values should therefore be treated as tuning parameters rather than universal physiological thresholds.

---

# 6. Persistent Movement Detection

A single threshold crossing is not enough to trigger a movement.

Short-duration noise spikes may briefly exceed a threshold even when no movement is occurring.

The detector therefore requires a channel to remain above its trigger threshold for approximately:

```text
30 ms
```

before the crossing is considered valid.

Conceptually:

```text
Activity crosses trigger threshold
              |
              v
     Remains above threshold?
          /          \
        No            Yes
        |              |
     Reject        >= 30 ms
                       |
                       v
                Valid activation
```

The earliest channel to produce a valid persistent threshold crossing defines the **global movement onset**.

The identity of that channel is stored as the:

```text
trigger_channel
```

for the movement.

---

# 7. Per-Channel Activation Timing

After global onset has been detected, the code searches the following transient window to determine when each remaining channel first produces a persistent activation.

This allows the system to record not only whether each muscle became active, but also **when it activated relative to the first detected muscle**.

For example:

```text
Global onset / CH1     0 ms
CH4 activation        17 ms
CH2 activation        31 ms
CH3 activation        48 ms
```

This relative timing becomes part of the feature vector used by the classifier.

---

# 8. Trigger-Channel Re-Arming

Earlier versions of the detector could become blocked because re-arming depended on the state of multiple EMG channels.

A noisy or slowly relaxing channel could therefore prevent the system from becoming ready for the next movement.

The current detector instead remembers the channel that originally triggered the movement.

Only that **triggering channel** controls re-arming.

After the 300 ms feature window has finished, the triggering channel must remain continuously below its own re-arm threshold for approximately:

```text
150 ms
```

before the detector becomes armed again.

```text
Movement triggered by CHx
          |
          v
300 ms feature window
          |
          v
Monitor CHx only
          |
          v
CHx < re-arm threshold
continuously for 150 ms
          |
          v
Detector armed again
```

If the recording ends while the detector is still disarmed, the software prints diagnostic information indicating:

- which channel is blocking re-arm;
- its re-arm threshold;
- the minimum activity reached after the transient window;
- whether it ever crossed the reset boundary.

This was added specifically to make threshold tuning easier.

---

# 9. Diagnostic Preview and Manual Recording Acceptance

Before data from a recording are added to the training set, the current trainer provides a visual diagnostic stage.

For each selected recording it generates two main figures.

## Figure 1 — Filtered EMG and Detected Events

The first plot shows:

- the filtered EMG signals;
- the 3–10 s resting calibration region;
- the point where movement detection begins;
- detected movement onsets;
- detector re-arm points.

The colour of each event corresponds to the channel that triggered that movement.

---

## Figure 2 — Per-Channel Threshold Diagnosis

A second figure shows the activity signal separately for each channel together with:

- the recording-specific trigger threshold;
- the recording-specific re-arm threshold;
- detected triggers;
- detected re-arm points.

This provides a quick visual check of whether the selected threshold values are behaving sensibly for that recording.

---

## Manual Acceptance

After viewing the diagnostic plots, the user is asked:

```text
Use these N events for <gesture>? [Y/n]:
```

The complete recording can therefore be:

```text
accepted
```

or:

```text
rejected
```

before any of its detected movements are added to the machine-learning dataset.

This was useful during development because event detection was still being tuned and obvious artefacts or incorrectly segmented recordings could be excluded before training.

The expected recording contains approximately **20 repetitions**, and the program displays:

```text
Detected events: N / expected about 20
```

to assist with this check.

---

# 10. Transient Window

For every accepted movement, the software extracts:

```text
300 ms
```

of EMG beginning at the global detected onset.

```text
                         300 ms
                  <---------------->

EMG --------------|================|--------------
                  ^
                  |
             global onset
```

Events are ignored if a complete 300 ms window is not available before the recording ends.

The aim is to determine whether sufficient information exists in this early transient period to identify the intended movement.

Future work may investigate shorter windows, particularly around **150 ms**, to reduce the time required to make a decision.

---

# 11. Feature Extraction

Six features are generated for every enabled channel.

## Mean Absolute Value — MAV

```text
MAV
```

represents the average magnitude of the filtered EMG within the transient window.

---

## Waveform Length — WL

```text
WL
```

is the sum of the absolute differences between consecutive samples.

It represents both signal amplitude and variation.

---

## Zero Crossings — ZC

```text
ZC
```

counts useful changes in signal polarity.

Small crossings are rejected using a noise threshold so that very small baseline fluctuations are not counted as meaningful signal behaviour.

---

## Slope Sign Changes — SSC

```text
SSC
```

counts significant changes in the direction of the waveform slope.

As with zero crossings, small changes are rejected using a noise threshold.

---

## Activation Flag

```text
activation_flag
```

records whether that channel produced a valid persistent activation during the 300 ms transient window.

```text
0 = not activated
1 = activated
```

---

## Activation Latency

```text
activation_latency_ms
```

records when the channel activated relative to global onset.

For channels that do not activate during the transient window, the latency is currently assigned the full feature-window duration:

```text
300 ms
```

This allows the classifier to use information about both:

- **which muscles activated**;
- **the order and relative timing of their activation**.

---

# 12. Feature Vector

Each channel contributes:

```text
[MAV,
 WL,
 ZC,
 SSC,
 activation_flag,
 activation_latency]
```

These are concatenated across the enabled channels.

For four channels, for example:

```text
CH1 features
     +
CH2 features
     +
CH3 features
     +
CH4 features
     |
     v
Combined movement feature vector
```

The feature names and extracted values are saved to:

```text
training_features.csv
```

together with the corresponding gesture label.

---

# 13. Gesture Classes

The current classifier is trained using five intended movements:

```text
Open
Point
Fist
Pinch
Spoon
```

Rest is not treated as an additional machine-learning class.

Instead, rest corresponds to the absence of a detected movement event.

---

# 14. Classifier

The current classifier pipeline is:

```text
Extracted features
        |
        v
StandardScaler
        |
        v
LogisticRegression
```

The Logistic Regression model uses:

```python
class_weight="balanced"
max_iter=5000
```

Balanced class weighting reduces the effect of different numbers of accepted examples between gesture classes.

The scaler and classifier are combined into one Scikit-learn pipeline.

---

# 15. Saved Model

After training, the resulting model is saved as:

```text
gesture_model.joblib
```

The saved object contains not only the trained model but also metadata required by later software, including:

- feature names;
- gesture classes;
- sample rate;
- channel order;
- feature-window duration;
- persistence duration;
- re-arm duration;
- re-arm rule;
- baseline mode;
- calibration period;
- movement-detection start time;
- trigger `K` values;
- re-arm `K` values.

This makes the trained model more self-describing and reduces the risk of later inference code using incompatible processing settings.

---

# 16. Training Accuracy

The training script reports the accuracy obtained when predicting the same examples used to fit the model.

This is provided only as a development check.

It is **not a valid estimate of generalisation performance**.

The software explicitly warns against reporting this value as final classification accuracy.

A meaningful classifier result requires evaluation using completely unseen recordings.

At the current stage of the project, reliable unseen-data classification accuracy has therefore not been reported.

---

# 17. Current Development Priority

The main unresolved software problem is still **robust movement-event detection**.

Current work primarily involves:

- tuning per-channel trigger multipliers;
- tuning per-channel re-arm multipliers;
- rejecting non-physiological artefacts;
- checking that approximately the expected number of movements is detected;
- confirming detected events visually before training.

This work is intentionally being prioritised before final classifier optimisation.

A classifier trained on incorrectly detected transient windows would otherwise learn from incorrectly segmented data.

The development priority is therefore:

```text
Reliable signal
      |
      v
Reliable baseline
      |
      v
Reliable onset detection
      |
      v
Reliable segmentation
      |
      v
Clean training dataset
      |
      v
Classifier evaluation
```

---

# 18. Future Development

The next software steps include:

- finalising robust per-channel trigger and re-arm parameters;
- adding stronger artefact rejection;
- collecting a cleaner training dataset;
- testing the trained model on fully unseen recordings;
- comparing different transient-window lengths;
- investigating shorter decision windows around 150 ms;
- comparing Logistic Regression against alternative classifiers;
- developing a complete real-time classification pipeline;
- eventually deploying movement-intention detection directly onto the ESP32-S3.

The longer-term target architecture is:

```text
Wearable electrodes
       |
       v
ADS1198
       |
       v
ESP32-S3
       |
       +--> filtering
       |
       +--> onset detection
       |
       +--> transient features
       |
       +--> movement classification
       |
       v
Real-time movement-intention output
```

---

# Related Work

This software builds on the wearable sEMG acquisition platform developed by:

**A. Finka — Development of a Flexible EMG Readout Board for Wearable Hand Movement Prediction**

Original repository:

https://github.com/abbyfinka/FYP-Knitted-EMG
