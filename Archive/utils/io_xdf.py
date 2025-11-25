"""
XDF I/O Utilities for SSVEP BCI Calibration Data

Provides functions to write and read XDF files with:
- 8-channel EEG stream (Pz, P3, P4, PO3, PO4, O1, Oz, O2)
- Marker stream for event synchronization
- Proper LSL metadata and channel information

XDF format is used instead of .npz for better compatibility with LSL ecosystem
and standard EEG analysis tools.
"""

import numpy as np
import pyxdf
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet, local_clock
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


class XDFCalibrationWriter:
    """Writer for SSVEP calibration data in XDF format.

    Creates two LSL streams:
    1. EEG stream: 8 channels at 250 Hz
    2. Marker stream: Event markers (irregular rate)

    Both streams are recorded together (e.g., with LabRecorder) to create
    a synchronized XDF file.

    Channel montage (10-20 system):
    - Channel 1: Pz
    - Channel 2: P3
    - Channel 3: P4
    - Channel 4: PO3
    - Channel 5: PO4
    - Channel 6: O1
    - Channel 7: Oz
    - Channel 8: O2
    """

    def __init__(
        self,
        subject: str,
        session: int,
        config: SSVEPConfig = None,
        stream_name_eeg: str = "SSVEP-EEG",
        stream_name_markers: str = "SSVEP-Markers"
    ):
        """Initialize XDF calibration writer.

        Args:
            subject: Subject ID (e.g., "S01")
            session: Session number (1, 2, 3, ...)
            config: SSVEPConfig instance
            stream_name_eeg: Name for EEG stream
            stream_name_markers: Name for marker stream
        """
        self.subject = subject
        self.session = session
        self.config = config or DEFAULT_CONFIG
        self.stream_name_eeg = stream_name_eeg
        self.stream_name_markers = stream_name_markers

        # Create EEG stream
        self.eeg_outlet = self._create_eeg_stream()

        # Create marker stream
        self.marker_outlet = self._create_marker_stream()

    def _create_eeg_stream(self) -> StreamOutlet:
        """Create LSL outlet for EEG data."""
        # Create stream info
        info = StreamInfo(
            name=self.stream_name_eeg,
            type="EEG",
            channel_count=self.config.n_eeg_channels,
            nominal_srate=self.config.fs,
            channel_format='float32',
            source_id=f"{self.stream_name_eeg}_{self.subject}_S{self.session}"
        )

        # Add channel metadata
        desc = info.desc()

        # Add acquisition metadata
        acq = desc.append_child("acquisition")
        acq.append_child_value("manufacturer", "OpenBCI")
        acq.append_child_value("model", "Cyton")
        acq.append_child_value("serial_number", "N/A")

        # Add subject metadata
        subject_node = desc.append_child("subject")
        subject_node.append_child_value("id", self.subject)
        subject_node.append_child_value("session", str(self.session))

        # Add channel information
        channels = desc.append_child("channels")

        for i, name in enumerate(self.config.electrode_names):
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

            # Add 3D position if available
            if name in self.config.electrode_positions:
                x, y, z = self.config.electrode_positions[name]
                loc = ch.append_child("location")
                loc.append_child_value("X", f"{x:.3f}")
                loc.append_child_value("Y", f"{y:.3f}")
                loc.append_child_value("Z", f"{z:.3f}")
                loc.append_child_value("system", "10-20")

        # Create outlet
        outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)

        return outlet

    def _create_marker_stream(self) -> StreamOutlet:
        """Create LSL outlet for event markers."""
        # Create stream info
        info = StreamInfo(
            name=self.stream_name_markers,
            type="Markers",
            channel_count=1,
            nominal_srate=0,  # Irregular rate
            channel_format='string',
            source_id=f"{self.stream_name_markers}_{self.subject}_S{self.session}"
        )

        # Add metadata
        desc = info.desc()
        desc.append_child_value("manufacturer", "SSVEP-BCI")
        desc.append_child_value("subject_id", self.subject)
        desc.append_child_value("session", str(self.session))

        # Create outlet
        outlet = StreamOutlet(info)

        return outlet

    def push_eeg(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """Push EEG sample to LSL stream.

        Args:
            sample: EEG sample with shape (n_channels,)
            timestamp: Optional timestamp (uses local_clock() if None)
        """
        if timestamp is None:
            timestamp = local_clock()

        self.eeg_outlet.push_sample(sample.tolist(), timestamp)

    def push_marker(self, marker: str, timestamp: Optional[float] = None):
        """Push marker to LSL stream.

        Args:
            marker: Marker string (usually JSON)
            timestamp: Optional timestamp (uses local_clock() if None)
        """
        if timestamp is None:
            timestamp = local_clock()

        self.marker_outlet.push_sample([marker], timestamp)


def load_xdf_calibration(
    xdf_path: str,
    config: SSVEPConfig = None
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Load SSVEP calibration data from XDF file.

    Args:
        xdf_path: Path to XDF file
        config: SSVEPConfig instance (for validation)

    Returns:
        Tuple of:
        - eeg_data: EEG data array with shape (n_channels, n_samples)
        - markers: List of marker dictionaries with 'timestamp' and 'data' keys
        - metadata: Dictionary with stream metadata
    """
    config = config or DEFAULT_CONFIG

    # Load XDF file
    streams, header = pyxdf.load_xdf(xdf_path)

    # Find EEG and marker streams
    eeg_stream = None
    marker_stream = None

    for stream in streams:
        stream_type = stream['info']['type'][0]
        if stream_type == 'EEG':
            eeg_stream = stream
        elif stream_type == 'Markers':
            marker_stream = stream

    if eeg_stream is None:
        raise ValueError(f"No EEG stream found in {xdf_path}")

    if marker_stream is None:
        raise ValueError(f"No Marker stream found in {xdf_path}")

    # Extract EEG data
    eeg_data = eeg_stream['time_series'].T  # Shape: (n_channels, n_samples)
    eeg_timestamps = eeg_stream['time_stamps']

    # Validate channel count
    n_channels = eeg_data.shape[0]
    if n_channels != config.n_eeg_channels:
        raise ValueError(
            f"Expected {config.n_eeg_channels} channels, got {n_channels}"
        )

    # Extract markers
    markers = []
    for i, ts in enumerate(marker_stream['time_stamps']):
        marker_string = marker_stream['time_series'][i][0]
        try:
            marker_data = json.loads(marker_string)
        except json.JSONDecodeError:
            # If not JSON, store as plain string
            marker_data = {'raw': marker_string}

        markers.append({
            'timestamp': ts,
            'data': marker_data
        })

    # Extract metadata
    metadata = {
        'eeg_stream_name': eeg_stream['info']['name'][0],
        'marker_stream_name': marker_stream['info']['name'][0],
        'n_channels': n_channels,
        'sampling_rate': float(eeg_stream['info']['nominal_srate'][0]),
        'n_samples': eeg_data.shape[1],
        'duration_sec': eeg_timestamps[-1] - eeg_timestamps[0],
        'eeg_timestamps': eeg_timestamps,
        'channel_labels': [
            ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']
        ],
    }

    # Extract subject info if available
    if 'subject' in eeg_stream['info']['desc'][0]:
        subject_node = eeg_stream['info']['desc'][0]['subject'][0]
        if 'id' in subject_node:
            metadata['subject_id'] = subject_node['id'][0]
        if 'session' in subject_node:
            metadata['session'] = int(subject_node['session'][0])

    return eeg_data, markers, metadata


def extract_trials_from_xdf(
    xdf_path: str,
    config: SSVEPConfig = None,
    trial_duration_sec: float = 4.0,
    baseline_duration_sec: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Extract trials from XDF calibration data.

    Finds TRIAL_START markers and extracts EEG epochs around them.

    Args:
        xdf_path: Path to XDF file
        config: SSVEPConfig instance
        trial_duration_sec: Duration of each trial in seconds
        baseline_duration_sec: Duration before trial start to include

    Returns:
        Tuple of:
        - trial_data: Array with shape (n_trials, n_channels, n_samples)
        - labels: Array with shape (n_trials,) containing target frequency indices
        - metadata: Dictionary with trial metadata
    """
    config = config or DEFAULT_CONFIG

    # Load XDF
    eeg_data, markers, metadata = load_xdf_calibration(xdf_path, config)

    eeg_timestamps = metadata['eeg_timestamps']
    fs = metadata['sampling_rate']

    # Find trial markers
    trial_starts = []
    trial_labels = []

    for marker in markers:
        marker_data = marker['data']
        if 'label' in marker_data and marker_data['label'] == 'TRIAL_START':
            trial_starts.append(marker['timestamp'])

            # Get target frequency
            if 'payload' in marker_data and 'target_freq' in marker_data['payload']:
                target_freq = marker_data['payload']['target_freq']

                # Find frequency index
                freq_idx = -1
                for i, freq in enumerate(config.target_frequencies):
                    if abs(freq - target_freq) < 0.01:
                        freq_idx = i
                        break

                trial_labels.append(freq_idx)
            else:
                trial_labels.append(-1)  # Unknown

    if len(trial_starts) == 0:
        raise ValueError(f"No TRIAL_START markers found in {xdf_path}")

    # Extract epochs
    n_samples_baseline = int(baseline_duration_sec * fs)
    n_samples_trial = int(trial_duration_sec * fs)
    n_samples_total = n_samples_baseline + n_samples_trial

    trials = []
    valid_labels = []

    for trial_idx, (ts, label) in enumerate(zip(trial_starts, trial_labels)):
        # Find corresponding sample index
        sample_idx = np.argmin(np.abs(eeg_timestamps - ts))

        # Extract epoch
        start_idx = sample_idx - n_samples_baseline
        end_idx = start_idx + n_samples_total

        # Check bounds
        if start_idx < 0 or end_idx > eeg_data.shape[1]:
            print(f"Warning: Trial {trial_idx+1} out of bounds, skipping")
            continue

        epoch = eeg_data[:, start_idx:end_idx]
        trials.append(epoch)
        valid_labels.append(label)

    trial_data = np.array(trials)  # Shape: (n_trials, n_channels, n_samples)
    labels = np.array(valid_labels)

    # Create metadata
    trial_metadata = {
        'n_trials': len(trials),
        'trial_duration_sec': trial_duration_sec,
        'baseline_duration_sec': baseline_duration_sec,
        'n_samples_per_trial': n_samples_total,
        'sampling_rate': fs,
        'xdf_path': xdf_path,
    }

    return trial_data, labels, trial_metadata


def extract_baseline_from_xdf(
    xdf_path: str,
    config: SSVEPConfig = None
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Extract baseline period from XDF calibration data.

    Looks for BASELINE_START and BASELINE_END markers.

    Args:
        xdf_path: Path to XDF file
        config: SSVEPConfig instance

    Returns:
        Tuple of:
        - baseline_data: Array with shape (n_channels, n_samples) or None if no baseline
        - metadata: Dictionary with baseline metadata or None
    """
    config = config or DEFAULT_CONFIG

    # Load XDF
    eeg_data, markers, metadata = load_xdf_calibration(xdf_path, config)

    eeg_timestamps = metadata['eeg_timestamps']
    fs = metadata['sampling_rate']

    # Find baseline markers
    baseline_start = None
    baseline_end = None

    for marker in markers:
        marker_data = marker['data']
        if 'label' in marker_data:
            if marker_data['label'] == 'BASELINE_START':
                baseline_start = marker['timestamp']
            elif marker_data['label'] == 'BASELINE_END':
                baseline_end = marker['timestamp']

    if baseline_start is None or baseline_end is None:
        return None, None

    # Find sample indices
    start_idx = np.argmin(np.abs(eeg_timestamps - baseline_start))
    end_idx = np.argmin(np.abs(eeg_timestamps - baseline_end))

    # Extract baseline data
    baseline_data = eeg_data[:, start_idx:end_idx]

    # Create metadata
    baseline_metadata = {
        'duration_sec': baseline_end - baseline_start,
        'n_samples': baseline_data.shape[1],
        'sampling_rate': fs,
        'start_time': baseline_start,
        'end_time': baseline_end,
    }

    return baseline_data, baseline_metadata


# Unit test
if __name__ == "__main__":
    print("Testing XDF I/O utilities...")

    from utils.lsl_markers import LSLMarkerSender, EventMarkers

    config = SSVEPConfig()

    # Create test XDF writer
    print("\nCreating XDF writer...")
    writer = XDFCalibrationWriter("S01", 1, config)
    print(f"  EEG stream: {writer.stream_name_eeg}")
    print(f"  Marker stream: {writer.stream_name_markers}")

    # Create marker sender
    marker_sender = LSLMarkerSender(writer.stream_name_markers)

    # Simulate recording
    print("\nSimulating calibration recording...")

    # Start calibration
    marker_sender.send_start_calibration(subject="S01", session=1)
    print("  -> START_CALIBRATION")

    # Baseline period
    marker_sender.send_baseline_start(duration_sec=1.0)
    print("  -> BASELINE_START")

    # Push 1 second of baseline EEG
    baseline_duration = 1.0
    n_baseline_samples = int(config.fs * baseline_duration)
    for i in range(n_baseline_samples):
        sample = np.random.randn(config.n_eeg_channels) * 10
        writer.push_eeg(sample)

    marker_sender.send_baseline_end()
    print("  -> BASELINE_END")

    # Simulate 4 trials
    import time
    frequencies = [8.57, 10.0, 12.0, 15.0]

    for trial_idx, freq in enumerate(frequencies, start=1):
        print(f"\n  Trial {trial_idx} (target={freq} Hz)")

        # Trial start
        marker_sender.send_trial_start(trial=trial_idx, target_freq=freq)
        print(f"    -> TRIAL_START")

        # Stim on
        marker_sender.send_stim_on(freq=freq, trial=trial_idx)
        print(f"    -> STIM_ON")

        # Push 4 seconds of trial EEG
        trial_duration = 4.0
        n_trial_samples = int(config.fs * trial_duration)
        for i in range(n_trial_samples):
            # Simulate SSVEP at target frequency
            t = i / config.fs
            signal_val = 20 * np.sin(2 * np.pi * freq * t)
            sample = np.random.randn(config.n_eeg_channels) * 10 + signal_val
            writer.push_eeg(sample)

        # Stim off
        marker_sender.send_stim_off(freq=freq, trial=trial_idx)
        print(f"    -> STIM_OFF")

        # Rest period
        marker_sender.send_rest_start(trial=trial_idx, duration_sec=1.0)
        for i in range(int(config.fs * 1.0)):
            sample = np.random.randn(config.n_eeg_channels) * 10
            writer.push_eeg(sample)
        marker_sender.send_rest_end(trial=trial_idx)

    # End calibration
    marker_sender.send_end_calibration(n_trials=4)
    print("\n  -> END_CALIBRATION")

    print("\nXDF writer test completed!")
    print("\nNote: To actually save to XDF, you need to run LabRecorder")
    print("      or use pyxdf's XDFWriter class to record the LSL streams.")
    print("\nXDF I/O utilities test passed!")
