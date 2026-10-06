import asyncio
from datetime import datetime
import threading
from bleak import BleakClient, BleakScanner
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from pathlib import Path
import struct
import numpy as np
from scipy import signal

# -------------------- BLE UUIDs -------------------- #
SERVICE_UUID = "cde33313-b7aa-4b32-b29f-9043b1d8e042"
DATA_CHARACTERISTIC_UUID = "89fea506-0482-4895-b474-843229dae557"
GESTURE_CHARACTERISTIC_UUID = "9122613f-3d96-4ba2-9bb5-382cbda24f02"
TARGET_DEVICE_NAME = "EMG-Logger"
PACKET_SIZE = 15

# -------------------- Display settings -------------------- #
PRINT_INTERVAL = 0.512
sampling_frequency = 1000
data_buffer_len = sampling_frequency * 2
time_step = 1 / sampling_frequency
ylim_time = 10
ylim_freq = 250
sample_index = 1

# Config variables
logging = True # Change to False to disable logging data to a text file
filtering = False # Change to True to enable real time bandpass filtering
fc_lowpass = 499
fc_highpass = 20

connect = True
maxRetry = 5

gestures = ("Rest", 
            "Fist", 
            "Flexion",
            "Extension",
            "Radial",
            "Ulnar")

# Create log file if logging enabled
if (logging):
    log_dir = Path("Logs")
    log_file_path = log_dir / (str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "_emg_log.txt")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_path, "w")
    log_file.write("%Raw EMG data\n%Number of channels = 8\n%Sample Rate = " + str(sampling_frequency) + "Hz\n%Board = ESP32\n")
    # headers matching Cyton board format for compatibility with existing processing pipelines, even though not all channels are used
    log_file.write("Sample Index, EXG Channel 0, EXG Channel 1, EXG Channel 2, EXG Channel 3, EXG Channel 4, EXG Channel 5, EXG Channel 6, EXG Channel 7, Accel Channel 0, Accel Channel 1, Accel Channel 2, Not Used, Digital Channel 0 (D11), Digital Channel 1 (D12), Digital Channel 2 (D13), Digital Channel 3 (D17), Not Used, Digital Channel 4 (D18), Analog Channel 0, Analog Channel 1, Analog Channel 2, Timestamp, Marker Channel, Timestamp (Formatted)\n")
    print(f"Logging enabled, writing to {log_file_path}")

# -------------------- Decode data recieved over BLE -------------------- #

async def process_data(queue):
    global sample_index
    while True:
        data = await queue.get() # get data from queue when available
        # print(len(data))
        # processing data
        try: 
            format_string = '<' + '8h' * PACKET_SIZE
            unpacked_data = struct.unpack(format_string, data)
    
            for i in range(0, PACKET_SIZE):
                # timestamp = unpacked_data[i * 9]
                timestamp = 0.001 * sample_index
                channels = unpacked_data[i * 8: (1 + i) * 8]
                channels_scaled = [0,0,0,0,0,0,0,0]
                for n in range(0,8):
                    # scale recieved values
                    channels_scaled[n] = channels[n] / 1000
                    # append new values
                    channel_data[n].append(channels_scaled[n])
                log_file.write(str(sample_index) + ", " + ", ".join(str(c) for c in channels_scaled) + ", " + ", ".join(str(c) for c in [0.0] * 13) + ", " + str(datetime.now().timestamp()) +  ", " + str(0.0) + ", " + str(datetime.now()) + "\n")
                # log_file.write(datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ": " + str(timestamp) + ": " + str(channels_scaled) + "\n")
                sample_index += 1

        except Exception as e:
            print("Error decoding data " + str(e))

        queue.task_done()

async def process_classifier(queue):
    while True:
        data = await queue.get()
        # print(data)
        try:
            format_string = '<' + 'b'
            unpacked_data = struct.unpack(format_string, data)
            current_pose = unpacked_data[0]
            fig.suptitle(f"{gestures[current_pose]}", fontsize=16, fontweight="bold")

        except Exception as e:
            print("Error decoding data " + str(e))

# -------------------- Recieve BLE -------------------- #

async def ble_receive():
    print('Starting BLE receive...')
    # asyncio queue to hold incoming data

    data_queue = asyncio.Queue()
    gesture_queue = asyncio.Queue()

    # The callback function just throws data in the bucket
    def data_notification_handler(sender, data):
        data_queue.put_nowait(data)

    def gesture_notification_handler(sender, data):
        gesture_queue.put_nowait(data)

    # Start the background processing task when data is put in queue
    process_task = asyncio.create_task(process_data(data_queue))
    process_classifier_task = asyncio.create_task(process_classifier(gesture_queue))

    print(f"Scanning for device named '{TARGET_DEVICE_NAME}'...")
    
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and d.name.lower() == TARGET_DEVICE_NAME.lower(),
        timeout=10.0 # 10s timeout
    )
    
    if not device:
        print(f"Error: Could not find '{TARGET_DEVICE_NAME}'")
        return

    print(f"Found '{device.name}' at address: {device.address}")

    print("Connecting...")
    async with BleakClient(device.address) as client:
        print("Connected!")
        
        await client.start_notify(DATA_CHARACTERISTIC_UUID, data_notification_handler)
        await client.start_notify(GESTURE_CHARACTERISTIC_UUID, gesture_notification_handler)
        
        print("Streaming data. Press Ctrl+C to stop.")
        try:
            while connect == True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await client.stop_notify(DATA_CHARACTERISTIC_UUID)
            await client.stop_notify(GESTURE_CHARACTERISTIC_UUID)
            process_task.cancel()
            client.disconnect()
            print("Disconnected.")




def run_ble_process():
    asyncio.run(ble_receive())


channel_data = [deque(maxlen=data_buffer_len) for _ in range(8)] # Data buffer

# Start BLE thread
print('Starting BLE thread...')
threading.Thread(target=run_ble_process, daemon=True).start()

# -------------------- Plotting -------------------- #

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(8, 2, figure=fig, width_ratios=[1,1])

time_plots = []
freq_plots = []

time_axes = []
freq_axes = []

print("Setting up plots...")
for i in range(8):
    # time domain plots
    ax_time = fig.add_subplot(gs[i, 0])
    time_plot, = ax_time.plot([], [], label=f"Channel {i+1}", linewidth=0.5)
    time_plots.append(time_plot)
    ax_time.set_ylim(-ylim_time, ylim_time)
    ax_time.set_xlim(0, data_buffer_len * time_step)
    ax_time.set_ylabel(f"Ch {i+1} / mV")
    time_axes.append(ax_time)

    # frequency domain plots
    ax_freq = fig.add_subplot(gs[i, 1])
    freq_plot, = ax_freq.plot([], [], label=f"Channel {i+1}", linewidth=0.5)
    freq_plots.append(freq_plot)
    ax_freq.set_xlim(0, sampling_frequency/2)
    ax_freq.set_ylim(0, ylim_freq)
    freq_axes.append(ax_freq)

fig.suptitle("- - -%", fontsize=16, fontweight="bold")
time_axes[-1].set_xlabel("Time / s")
freq_axes[-1].set_xlabel("Frequency / Hz")
plt.tight_layout()

def animate(frame):
    for idx, line in enumerate(time_plots):
        y = list(channel_data[idx])
        if filtering and len(y) > 1:
            sos_bandpass = signal.butter(8, [fc_highpass, fc_lowpass], 'bandpass', fs=sampling_frequency, output='sos')
            y = signal.sosfilt(sos_bandpass, y)
            
        x = list([s * time_step for s in range(len(y))])
        line.set_data(x, y)

    for idx, line in enumerate(freq_plots):
        y = list(channel_data[idx])
        if len(y) > 1:
            if filtering:
                sos_bandpass = signal.butter(8, [20, 499], 'bandpass', fs=sampling_frequency, output='sos')
                y = signal.sosfilt(sos_bandpass, y)
            freqs = np.fft.rfftfreq(len(y), d=time_step)
            fft_values = np.abs(np.fft.rfft(y))
            line.set_data(freqs, fft_values)

    return time_plots + freq_plots

ani = FuncAnimation(fig, animate, interval=50, blit=True)
plt.tight_layout()
plt.show()
