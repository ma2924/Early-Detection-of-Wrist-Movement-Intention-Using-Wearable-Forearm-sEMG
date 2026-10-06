# Installation and Setup

This project builds on the wearable sEMG acquisition platform developed by **Abby Finka** at Imperial College London.

The original project provides the underlying:

- ADS1198-based sEMG acquisition hardware;
- ESP32-S3 firmware;
- Bluetooth Low Energy communication;
- Python acquisition application.

Original repository:

**[Abby Finka — FYP-Knitted-EMG](https://github.com/abbyfinka/FYP-Knitted-EMG)**

## 1. Set Up the Original Project

First follow the installation and hardware setup instructions in the original repository.

Ensure that the ESP32-S3 acquisition system and BLE connection are working before adding the movement-intention processing software from this repository.

---

## 2. Install the Additional Python Dependencies

From the root of this repository, install the required Python packages using:

```bash
pip install -r requirements.txt
```

The supplied `requirements.txt` contains:

```text
bleak
joblib
matplotlib
numpy
scipy
scikit-learn
```

`tkinter` is also used for file-selection dialogs and is normally included with standard Python installations.

---

## 3. Add the UROP Software

Copy the files from this repository's:

```text
App/
```

directory into the original project's:

```text
FYP-Knitted-EMG/App/
```

The resulting application folder should contain:

```text
App/
├── EMGApp.py
├── calibrate_baseline.py
├── train_model.py
└── train_model_saved_thresholds.py
```

Additional files and folders are generated locally as the software is used.

---

## 4. Training Data

The training data used during development were recorded using an **OpenBCI Cyton** board.

The processing scripts are not restricted to Cyton recordings, however. Any text-based recording can be used provided that:

- the EMG samples are arranged in the expected columns;
- the column names or numeric column positions match those defined in the code;
- the recording sample rate matches the configuration used by the processing pipeline.

The scripts support common comma-separated, tab-separated and semicolon-separated text files.

For OpenBCI-style recordings, channels are normally identified using headings such as:

```text
EXG Channel 0
EXG Channel 1
EXG Channel 2
EXG Channel 3
```

---

## 5. Generate `baseline_config.json`

Run:

```bash
python calibrate_baseline.py
```


and select a resting EMG recording when prompted.

The script generates:

```text
baseline_config.json
```

inside the `App/` directory.

This file stores shared processing information including:

- sample rate;
- filtering parameters;
- activity-window duration;
- channel mappings;
- reference noise statistics;
- reference trigger and reset thresholds.

---

## 6. Choose a Training Method

Two versions of the gesture-training pipeline are included.

### Current method — `train_model.py`

```bash
python train_model.py
```

This is the current development version.

It uses information from `baseline_config.json`, but recalculates movement trigger and re-arm thresholds from the resting period contained within each individual training recording.

The expected recording structure is:

```text
0–3 s       startup / ignored
3–10 s      resting calibration
10 s onward gesture repetitions
```

This approach allows the event detector to adapt to the baseline activity of each recording.

### Saved-threshold method — `train_model_saved_thresholds.py`

```bash
python train_model_saved_thresholds.py
```

This version uses the trigger and re-arm thresholds stored directly in:

```text
baseline_config.json
```

It is retained as an earlier/alternative version of the event-detection pipeline and allows comparison with the later recording-specific calibration method.

---

## 7. Review Detected Events

During training, diagnostic plots are displayed for each selected recording.

These show:

- filtered EMG signals;
- detected movement onsets;
- trigger-channel information;
- re-arm points;
- channel activity signals;
- trigger and re-arm thresholds.

After viewing the plots, the program asks whether the detected events should be included in the training dataset.

For example:

```text
Use these 20 events for Fist? [Y/n]:
```

This allows incorrectly segmented recordings or obvious artefacts to be rejected before model training.

---

## 8. Generated Training Files

The training scripts generate files such as:

```text
training_features.csv
gesture_model.joblib
```

The saved feature table contains the extracted movement features and gesture labels.

The `.joblib` file contains the trained Scikit-learn pipeline together with processing metadata required to interpret the model.

These files are generated locally and do not need to be included in the repository.

---

## Current Gesture Classes

The current classifier uses five gesture classes:

```text
Open
Point
Fist
Pinch
Spoon
```

The classifier pipeline is:

```text
Detected movement
        ↓
300 ms transient window
        ↓
MAV + WL + ZC + SSC
+ activation flag
+ activation latency
        ↓
StandardScaler
        ↓
Logistic Regression
```

---

## Notes

> **Sample-rate note:** The training data used during development were recorded with the OpenBCI Cyton at **250 Hz**. The supplied processing workflow should therefore use a `baseline_config.json` generated from 250 Hz data. If adapting the pipeline to recordings from the custom ADS1198/ESP32-S3 wearable, which operates at **1000 Hz**, generate a new `baseline_config.json` using the 1000 Hz recordings.

The current software remains a research-development pipeline.

Event-detection thresholds are still being tuned, and final classification performance should be evaluated using completely unseen recordings before reporting test accuracy.
