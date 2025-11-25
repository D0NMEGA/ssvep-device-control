"""
SSVEP Preprocessing, XDF I/O, and LSL Markers - ALL-IN-ONE MODULE

Combines:
- Online (causal) preprocessing for real-time use
- Offline (zero-phase) preprocessing for training
- XDF file I/O for calibration data
- LSL marker streaming for event synchronization
"""

import numpy as np
import json
from scipy import signal
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# LSL/XDF imports
from pylsl import StreamInfo, StreamOutlet, local_clock
import pyxdf


# =============================================================================
# EVENT MARKERS
# =============================================================================

class EventMarkers:
    """LSL event marker IDs and labels."""

    # Session events
    START_CALIBRATION = 1
    BASELINE_START = 2
    BASELINE_END = 3
    END_CALIBRATION = 99

    # Trial events
    TRIAL_START = 10

    # Stimulation events (per frequency)
    STIM_ON_8p57 = 11
    STIM_OFF_8p57 = 12
    STIM_ON_10 = 13
    STIM_OFF_10 = 14
    STIM_ON_12 = 15
    STIM_OFF_12 = 16
    STIM_ON_15 = 17
    STIM_OFF_15 = 18

    # Rest events
    REST_START = 20
    REST_END = 21

    LABELS = {
        1: "START_CALIBRATION", 2: "BASELINE_START", 3: "BASELINE_END",
        10: "TRIAL_START",
        11: "STIM_ON_8p57", 12: "STIM_OFF_8p57",
        13: "STIM_ON_10", 14: "STIM_OFF_10",
        15: "STIM_ON_12", 16: "STIM_OFF_12",
        17: "STIM_ON_15", 18: "STIM_OFF_15",
        20: "REST_START", 21: "REST_END",
        99: "END_CALIBRATION"
    }

    @classmethod
    def get_stim_on_id(cls, freq: float) -> int:
        """Get STIM_ON event ID for frequency."""
        freq_map = {8.57: cls.STIM_ON_8p57, 10.0: cls.STIM_ON_10,
                    12.0: cls.STIM_ON_12, 15.0: cls.STIM_ON_15}
        for f, eid in freq_map.items():
            if abs(freq - f) < 0.01:
                return eid
        raise ValueError(f"Unknown frequency: {freq}")

    @classmethod
    def get_stim_off_id(cls, freq: float) -> int:
        """Get STIM_OFF event ID for frequency."""
        freq_map = {8.57: cls.STIM_OFF_8p57, 10.0: cls.STIM_OFF_10,
                    12.0: cls.STIM_OFF_12, 15.0: cls.STIM_OFF_15}
        for f, eid in freq_map.items():
            if abs(freq - f) < 0.01:
                return eid
        raise ValueError(f"Unknown frequency: {freq}")


class LSLMarkerSender:
    """Send LSL event markers."""

    def __init__(self, stream_name: str = "SSVEP-Markers"):
        info = StreamInfo(stream_name, "Markers", 1, 0, 'string', f"{stream_name}_events")
        self.outlet = StreamOutlet(info)

    def send_marker(self, event_id: int, payload: Optional[Dict] = None):
        """Send marker with event ID and payload."""
        marker_data = {
            "id": event_id,
            "label": EventMarkers.LABELS.get(event_id, f"UNKNOWN_{event_id}"),
            "payload": payload or {}
        }
        self.outlet.push_sample([json.dumps(marker_data)])

    def send_start_calibration(self, **kwargs):
        self.send_marker(EventMarkers.START_CALIBRATION, kwargs)

    def send_end_calibration(self, **kwargs):
        self.send_marker(EventMarkers.END_CALIBRATION, kwargs)

    def send_baseline_start(self, **kwargs):
        self.send_marker(EventMarkers.BASELINE_START, kwargs)

    def send_baseline_end(self, **kwargs):
        self.send_marker(EventMarkers.BASELINE_END, kwargs)

    def send_trial_start(self, trial: int, target_freq: float, **kwargs):
        payload = {"trial": trial, "target_freq": target_freq}
        payload.update(kwargs)
        self.send_marker(EventMarkers.TRIAL_START, payload)

    def send_stim_on(self, freq: float, trial: int, **kwargs):
        event_id = EventMarkers.get_stim_on_id(freq)
        payload = {"trial": trial, "event": "stim_on", "freq": freq}
        payload.update(kwargs)
        self.send_marker(event_id, payload)

    def send_stim_off(self, freq: float, trial: int, **kwargs):
        event_id = EventMarkers.get_stim_off_id(freq)
        payload = {"trial": trial, "event": "stim_off", "freq": freq}
        payload.update(kwargs)
        self.send_marker(event_id, payload)

    def send_rest_start(self, trial: int, **kwargs):
        payload = {"trial": trial, "event": "rest_start"}
        payload.update(kwargs)
        self.send_marker(EventMarkers.REST_START, payload)

    def send_rest_end(self, trial: int, **kwargs):
        payload = {"trial": trial, "event": "rest_end"}
        payload.update(kwargs)
        self.send_marker(EventMarkers.REST_END, payload)


# =============================================================================
# XDF I/O
# =============================================================================

class XDFWriter:
    """Write EEG and markers to LSL (for LabRecorder to save as XDF)."""

    def __init__(self, subject: str, session: int, config):
        self.subject = subject
        self.session = session
        self.config = config

        # Create EEG outlet
        info_eeg = StreamInfo("SSVEP-EEG", "EEG", config.n_eeg_channels,
                             config.fs, 'float32', f"SSVEP-EEG_{subject}_S{session}")

        desc = info_eeg.desc()
        channels = desc.append_child("channels")
        for name in config.electrode_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")

        self.eeg_outlet = StreamOutlet(info_eeg, chunk_size=32)

        # Marker sender (separate stream)
        self.marker_sender = LSLMarkerSender("SSVEP-Markers")

    def push_eeg(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """Push EEG sample to LSL."""
        if timestamp is None:
            timestamp = local_clock()
        self.eeg_outlet.push_sample(sample.tolist(), timestamp)


def load_xdf(xdf_path: str, config) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Load XDF file and extract EEG + markers.

    Returns:
        (eeg_data, markers, metadata)
        - eeg_data: (n_channels, n_samples)
        - markers: list of {timestamp, data} dicts
        - metadata: dict with stream info
    """
    streams, header = pyxdf.load_xdf(str(xdf_path))

    eeg_stream = None
    marker_stream = None

    for stream in streams:
        if stream['info']['type'][0] == 'EEG':
            eeg_stream = stream
        elif stream['info']['type'][0] == 'Markers':
            marker_stream = stream

    if eeg_stream is None:
        raise ValueError(f"No EEG stream in {xdf_path}")
    if marker_stream is None:
        raise ValueError(f"No Marker stream in {xdf_path}")

    eeg_data = eeg_stream['time_series'].T
    eeg_timestamps = eeg_stream['time_stamps']

    markers = []
    for i, ts in enumerate(marker_stream['time_stamps']):
        marker_str = marker_stream['time_series'][i][0]
        try:
            marker_data = json.loads(marker_str)
        except:
            marker_data = {'raw': marker_str}
        markers.append({'timestamp': ts, 'data': marker_data})

    metadata = {
        'n_channels': eeg_data.shape[0],
        'n_samples': eeg_data.shape[1],
        'sampling_rate': float(eeg_stream['info']['nominal_srate'][0]),
        'eeg_timestamps': eeg_timestamps,
        'duration_sec': eeg_timestamps[-1] - eeg_timestamps[0]
    }

    return eeg_data, markers, metadata


def extract_trials_from_xdf(xdf_path: str, config, trial_duration_sec: float = 4.0,
                           baseline_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Extract trial epochs from XDF using TRIAL_START markers.

    Returns:
        (trial_data, labels)
        - trial_data: (n_trials, n_channels, n_samples)
        - labels: (n_trials,) frequency indices
    """
    eeg_data, markers, metadata = load_xdf(xdf_path, config)

    fs = metadata['sampling_rate']
    timestamps = metadata['eeg_timestamps']

    trial_starts = []
    trial_labels = []

    for marker in markers:
        mdata = marker['data']
        if mdata.get('label') == 'TRIAL_START':
            trial_starts.append(marker['timestamp'])
            target_freq = mdata.get('payload', {}).get('target_freq', -1)

            # Find frequency index
            freq_idx = -1
            for i, freq in enumerate(config.target_frequencies):
                if abs(freq - target_freq) < 0.01:
                    freq_idx = i
                    break
            trial_labels.append(freq_idx)

    # Extract epochs
    n_baseline = int(baseline_sec * fs)
    n_trial = int(trial_duration_sec * fs)
    n_total = n_baseline + n_trial

    trials = []
    valid_labels = []

    for ts, label in zip(trial_starts, trial_labels):
        idx = np.argmin(np.abs(timestamps - ts))
        start_idx = idx - n_baseline
        end_idx = start_idx + n_total

        if start_idx >= 0 and end_idx <= eeg_data.shape[1] and label >= 0:
            trials.append(eeg_data[:, start_idx:end_idx])
            valid_labels.append(label)

    return np.array(trials), np.array(valid_labels)


# =============================================================================
# PREPROCESSING
# =============================================================================

class OnlinePreprocessor:
    """Causal (stateful) preprocessor for real-time use.

    Uses stateful IIR filters for online streaming.
    Bandpass: 5-50 Hz, Notch: 60 Hz
    """

    def __init__(self, config):
        self.config = config

        # Design bandpass filter (5-50 Hz)
        nyq = config.fs / 2.0
        low = max(0.001, min(5.0 / nyq, 0.999))
        high = max(0.001, min(50.0 / nyq, 0.999))
        self.sos_bp = signal.butter(5, [low, high], btype='band', output='sos')

        # Design notch filter (60 Hz)
        b, a = signal.iirnotch(60.0 / nyq, Q=30.0)
        self.sos_notch = signal.tf2sos(b, a)

        # Initialize filter states
        self.zi_bp = [signal.sosfilt_zi(self.sos_bp) for _ in range(config.n_eeg_channels)]
        self.zi_notch = [signal.sosfilt_zi(self.sos_notch) for _ in range(config.n_eeg_channels)]
        self._initialized = False

    def reset(self):
        """Reset filter states."""
        self.zi_bp = [signal.sosfilt_zi(self.sos_bp) for _ in range(self.config.n_eeg_channels)]
        self.zi_notch = [signal.sosfilt_zi(self.sos_notch) for _ in range(self.config.n_eeg_channels)]
        self._initialized = False

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data causally (for online use).

        Args:
            data: (n_channels, n_samples)

        Returns:
            Preprocessed data (n_channels, n_samples)
        """
        # CAR
        common_avg = np.mean(data, axis=0, keepdims=True)
        processed = data - common_avg

        # Bandpass filter (causal)
        n_channels, n_samples = processed.shape
        filtered = np.zeros_like(processed)

        for ch in range(n_channels):
            if not self._initialized:
                self.zi_bp[ch] = signal.sosfilt_zi(self.sos_bp) * processed[ch, 0]
            filtered[ch], self.zi_bp[ch] = signal.sosfilt(
                self.sos_bp, processed[ch], zi=self.zi_bp[ch]
            )

        self._initialized = True
        processed = filtered

        # Notch filter (causal)
        filtered = np.zeros_like(processed)
        for ch in range(n_channels):
            filtered[ch], self.zi_notch[ch] = signal.sosfilt(
                self.sos_notch, processed[ch], zi=self.zi_notch[ch]
            )

        return filtered


class OfflinePreprocessor:
    """Zero-phase (non-causal) preprocessor for training.

    Uses filtfilt for zero-phase filtering.
    Bandpass: 7-90 Hz, Notch: 60 Hz
    """

    def __init__(self, config):
        self.config = config

        # Design bandpass filter (7-90 Hz)
        nyq = config.fs / 2.0
        low = 7.0 / nyq
        high = 90.0 / nyq
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        self.sos_bp = signal.butter(5, [low, high], btype='band', output='sos')

        # Design notch filter (60 Hz)
        b, a = signal.iirnotch(60.0 / nyq, Q=30.0)
        self.sos_notch = signal.tf2sos(b, a)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data with zero-phase filtering (for offline training).

        Args:
            data: (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Preprocessed data (same shape)
        """
        if data.ndim == 2:
            # Single trial
            # CAR
            common_avg = np.mean(data, axis=0, keepdims=True)
            processed = data - common_avg

            # Bandpass + notch (zero-phase)
            filtered = np.zeros_like(processed)
            for ch in range(processed.shape[0]):
                filtered[ch] = signal.sosfiltfilt(self.sos_bp, processed[ch])
                filtered[ch] = signal.sosfiltfilt(self.sos_notch, filtered[ch])

            return filtered

        elif data.ndim == 3:
            # Multiple trials
            n_trials = data.shape[0]
            processed = np.zeros_like(data)

            for trial in range(n_trials):
                # CAR
                common_avg = np.mean(data[trial], axis=0, keepdims=True)
                processed[trial] = data[trial] - common_avg

                # Bandpass + notch (zero-phase)
                for ch in range(data.shape[1]):
                    processed[trial, ch] = signal.sosfiltfilt(self.sos_bp, processed[trial, ch])
                    processed[trial, ch] = signal.sosfiltfilt(self.sos_notch, processed[trial, ch])

            return processed

        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
