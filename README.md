# Early Detection of Wrist Movement Intention Using Forearm sEMG

Software and wearable-electronics development from a UKESF-sponsored Undergraduate Research Opportunity at Imperial College London, investigating whether forearm surface electromyography (sEMG) can be used to detect voluntary wrist and hand movement intention before substantial mechanical movement occurs.

**Author:** Muhammad Abubakar  
**Supervisor:** Prof. Kristel Fobelets  
**Institution:** Imperial College London  
**Project:** Undergraduate Research Opportunity (UROP), 2026  

---

## Project Overview

This project builds on the wearable sEMG platform developed by **Abby Finka** during her final-year MEng project at Imperial College London:

**[Development of a Flexible EMG Readout Board for Wearable Hand Movement Prediction](https://github.com/abbyfinka/FYP-Knitted-EMG)**

Finka's project developed an eight-channel wearable sEMG acquisition system based on an **ADS1198 analogue front end** and **ESP32-S3 microcontroller**, with Bluetooth Low Energy transmission and a Python interface for visualising and recording EMG data.

My follow-on project focused on two main areas:

- **Wearable hardware redesign:** miniaturising and modularising the existing electronics to improve integration with the knitted forearm armband.
- **Movement-intention detection:** investigating the early transient period of the EMG signal to detect muscle activation before substantial mechanical movement, rather than focusing only on classification of the final hand or wrist position.

The longer-term motivation is the development of assistive wearable systems capable of identifying intended voluntary movement early enough to help distinguish it from involuntary motion such as tremor.

---

## Key Results

- Detected **19 of 20 intended muscle activations with zero false triggers** in the selected evaluation recording using channel-specific EMG onset detection.
- EMG onset typically occurred **110–150 ms before the subsequent peak acceleration change**, demonstrating that useful electrical activity was detectable before the later mechanical response.
- Implemented transient feature extraction using **Mean Absolute Value (MAV), Waveform Length (WL), Zero Crossings (ZC), Slope Sign Changes (SSC), Activation Flag and Activation Latency**.
- Developed an initial **Logistic Regression** pipeline for five hand gestures: **Open, Point, Fist, Pinch and Spoon**.
- Reduced the rigid wearable PCB from approximately **90 × 40 mm to 51 × 38 mm** and introduced a detachable flexible-PCB/FPC interface for the knitted electrode connections.

> The measured 110–150 ms interval is relative to the subsequent **peak acceleration change**, rather than the exact onset of physical movement, and should therefore not be interpreted as a direct measurement of electromechanical delay.

---

## Installation

This project extends the wearable sEMG platform developed by **Abby Finka**.

For the original ADS1198/ESP32-S3 hardware, firmware and BLE setup, first follow the instructions in:

**[Abby Finka — FYP-Knitted-EMG](https://github.com/abbyfinka/FYP-Knitted-EMG)**

The Python files in this repository can then be added to the original project's `App/` directory.

For the complete setup and software workflow, see:

**[Installation and Setup](docs/INSTALLATION.md)**

## Processing Pipeline

The software was developed as a staged pipeline so that baseline calibration, onset detection and classification could be tested independently.

```text
Resting EMG recording
        |
        v
Baseline calibration
        |
        v
Channel-specific thresholds
        |
        v
EMG onset detection
        |
        v
Transient-window extraction
        |
        v
Feature extraction
        |
        v
Gesture classification
```

### 1. Signal Processing

During algorithm development, recordings from an **OpenBCI Cyton** board were used as a controlled EMG source while the custom wearable hardware was being debugged.

The Cyton recordings were digitally filtered using approximately:

- 20 Hz high-pass filtering
- 50 Hz notch filtering
- 100 Hz low-pass filtering

A short moving absolute-EMG activity measure was then calculated for each channel.

### 2. Baseline Calibration

A dedicated resting recording is used to characterise the baseline activity of each muscle channel.

For each channel, the resting mean and standard deviation are used to define an activation threshold of the form:

```text
T_c = mean_c + k_c * standard_deviation_c
```

Separate channel-specific thresholds are required because baseline noise and contraction strength differ between muscles.

The calibration stage stores the resulting parameters so that identical baseline values can be reused when processing later gesture recordings.

### 3. EMG Onset Detection

The onset detector identifies the beginning of voluntary muscle activation using the calibrated channel-specific thresholds.

To reduce false detections caused by short noise spikes:

- activity must remain above the trigger threshold for a minimum persistence period;
- separate trigger and reset thresholds are used;
- the detector must re-arm before another movement can be accepted;
- re-arm behaviour is based on the channel responsible for the detected onset.

In the selected evaluation recording, this approach detected **19 of 20 intended movements with no false triggers**.

### 4. Transient Window Extraction

After detecting EMG onset, an early section of the signal is extracted for analysis.

The current implementation uses a **300 ms transient window** after onset.

Shorter windows are of interest because they could reduce the time required to make a movement-intention decision. Future work would investigate reducing the window towards approximately **150 ms**, and potentially lower if classification performance remains acceptable.

### 5. Feature Extraction

Six time-domain features are extracted from each active channel:

- **MAV** — Mean Absolute Value
- **WL** — Waveform Length
- **ZC** — Zero Crossings
- **SSC** — Slope Sign Changes
- **Activation Flag** — whether the channel activated
- **Activation Latency** — relative timing of channel activation

These features capture both the shape and magnitude of the transient EMG signal and the order/timing in which different forearm muscles become active.

### 6. Gesture Classification

The extracted features are standardised and passed to an initial **Logistic Regression** classifier.

The current gesture set consists of:

- Open
- Point
- Fist
- Pinch
- Spoon

Classifier validation on fully unseen recordings was not completed during the project, so a final classification accuracy is intentionally not reported.

Reliable onset detection and segmentation were prioritised before evaluating classifier performance, since incorrectly segmented transient windows would produce poorly labelled training data.

---

## Repository Structure

Only the final scripts relevant to the main processing pipeline are intended to be included publicly.

Earlier exploratory scripts used during development, including intermediate Cyton processing and trigger-testing programs, have been omitted to keep the repository focused.

```text
.
├── README.md
├── calibrate_baseline.py
├── baseline_config.json
├── train_model.py
└── EMGApp.py
```

### `calibrate_baseline.py`

Processes a dedicated resting EMG recording to characterise the baseline behaviour of each channel.

The calibration stage determines parameters used by the onset detector, including channel-specific baseline statistics, trigger thresholds and reset thresholds.

The resulting configuration is stored in `baseline_config.json`.

### `baseline_config.json`

Stores the calibrated parameters used by the later processing stages.

Keeping baseline calibration separate from gesture processing allows the same thresholds to be reused consistently across multiple recordings.

### `train_model.py`

Processes labelled gesture recordings using the saved calibration parameters.

The model-development pipeline includes:

1. EMG filtering
2. onset detection
3. transient-window extraction
4. time-domain feature extraction
5. feature standardisation
6. initial Logistic Regression training

### `EMGApp.py`

Main analysis/testing application used with the calibrated processing pipeline.

It applies the developed signal-processing and onset-detection approach to EMG recordings and supports inspection of the resulting movement detections and classification workflow.

---

## Hardware Development

Alongside the movement-intention software, the wearable electronics from the original project were redesigned to improve compactness and integration with the knitted armband.

The main acquisition architecture remained based on the **ADS1198 and ESP32-S3**, but the PCB was substantially reorganised in EasyEDA.

Changes included:

- reducing the rigid PCB from approximately **90 × 40 mm to 51 × 38 mm**;
- reorganising component placement across both sides of the PCB;
- retaining dedicated ground and 3.3 V planes;
- shortening analogue electrode paths;
- routing positive and negative electrode inputs as differential pairs;
- redesigning USB differential routing;
- adding test pads to improve hardware debugging;
- introducing a detachable flexible-PCB/FPC electrode interface.

A rigid main PCB was retained to provide mechanical support to the ESP32-S3, while flexibility was introduced through a separate flexible interface connecting the knitted electrode wiring to the electronics.

A snap-fit enclosure was also designed in **Autodesk Fusion**, with prototypes produced using PLA and TPU to investigate the balance between flexibility, rigidity and mechanical protection.

---

## Wearable Electrode Integration

The knitted armband contains eight electrode pairs together with reference connections, producing **18 electrode connections** in total.

The electrode wiring uses fine Litz wire connected to the detachable flexible-PCB interface.

During integration, each channel connection was individually identified and continuity-tested. Initial testing revealed that neighbouring textile electrodes could touch when the cylindrical armband was laid flat, creating misleading continuity measurements.

The complete channel mapping was therefore re-tested with electrodes physically separated, checking for both intended and unintended electrical continuity before correcting the final assignments.

---

## Current Status

The project demonstrated a working software pipeline for detecting early voluntary muscle activation from controlled forearm sEMG recordings.

The main completed software stages are:

- baseline calibration;
- channel-specific threshold generation;
- onset detection;
- persistence and re-arm logic;
- transient-window extraction;
- time-domain feature extraction;
- initial five-gesture classification (the classifier has not yet been deployed to the ESP32-S3 or integrated into the wearable for real-time inference.).

The custom wearable electronics were also manufactured and assembled successfully, but recordings from the wearable showed significant mains-frequency interference and channel-dependent noise.

Software development therefore continued primarily using the more controlled Cyton recordings while the custom wearable signal-quality issues were investigated.

---

## Future Work

Potential next steps include:

- implementing real-time onset detection and classification directly on the ESP32-S3;
- validating the gesture classifier using fully unseen recordings;
- reducing the transient classification window from 300 ms towards 150 ms or below;
- collecting larger and more controlled gesture datasets;
- improving signal quality from the knitted wearable;
- further investigating the source of noise in the custom PCB;
- testing the knitted electrodes independently from the custom acquisition electronics;
- implementing real-time onset detection and classification directly on the ESP32-S3;
- investigating methods for distinguishing voluntary movement from involuntary tremor.

---

## Related Work and Attribution

This project is a continuation of:

**A. Finka — Development of a Flexible EMG Readout Board for Wearable Hand Movement Prediction**  
Imperial College London, 2026

Original project repository:

**https://github.com/abbyfinka/FYP-Knitted-EMG**

The original project developed the underlying wearable sEMG acquisition architecture, including the ADS1198/ESP32-S3 electronics, BLE communication, Python acquisition interface and initial hand/wrist classification work.

This repository focuses on the subsequent hardware miniaturisation and the development of an early transient EMG movement-intention detection pipeline.

---

## Project Report & Disclaimer

The full official UROP project report is not hosted publicly in this repository.

For access to the report or further information about the project, please contact the author using the contact details available on this GitHub profile.

This project was developed as an undergraduate research prototype.

It is **not a medical device** and is not intended for clinical diagnosis, treatment or safety-critical use.

