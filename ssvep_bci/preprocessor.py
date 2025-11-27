"""
SSVEP Preprocessing, XDF I/O, and LSL Markers

Combines:
- Online (causal) preprocessing for real-time use
- Offline (zero-phase) preprocessing for training
- XDF file I/O for calibration data
- LSL marker streaming for event synchronization
- Laplacian spatial filtering
- Filter bank analysis (Nakanishi et al. 2018)
"""

import numpy as np
import json
from scipy import signal
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from .filterbank import FilterBank

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

    def __init__(self, stream_name: str = "SSVEP-Markers", unique_id: str = None):
        """Initialize LSL marker sender with unique stream identifier.

        Args:
            stream_name: Name of the marker stream (default: "SSVEP-Markers")
            unique_id: Unique source_id to prevent stream conflicts.
                       If None, generates timestamp-based unique ID.
        """
        # Generate unique source_id to avoid zombie streams in LabRecorder
        if unique_id is None:
            import time
            unique_id = f"{stream_name}_{int(time.time())}"

        info = StreamInfo(stream_name, "Markers", 1, 0, 'string', unique_id)
        self.outlet = StreamOutlet(info)
        self.stream_name = stream_name
        self.unique_id = unique_id

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
        """Send TRIAL_START marker with trial metadata.

        Args:
            trial: Trial number within this session
            target_freq: Target frequency for this trial
            **kwargs: Optional fields (global_trial_id, led_index, etc.)
        """
        payload = {"trial": trial, "target_freq": target_freq}
        payload.update(kwargs)
        self.send_marker(EventMarkers.TRIAL_START, payload)

    def send_stim_on(self, freq: float, trial: int, **kwargs):
        """Send STIM_ON marker (accurate stimulation onset).

        Args:
            freq: Stimulation frequency
            trial: Trial number within this session
            **kwargs: Optional fields (global_trial_id, led_index, etc.)
        """
        event_id = EventMarkers.get_stim_on_id(freq)
        payload = {"trial": trial, "event": "stim_on", "freq": freq}
        payload.update(kwargs)
        self.send_marker(event_id, payload)

    def send_stim_off(self, freq: float, trial: int, **kwargs):
        """Send STIM_OFF marker (stimulation offset).

        Args:
            freq: Stimulation frequency
            trial: Trial number within this session
            **kwargs: Optional fields (global_trial_id, led_index, etc.)
        """
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

    def close(self):
        """Close LSL outlet to prevent zombie streams in LabRecorder.

        This ensures the stream is properly released and won't appear as
        a red (disconnected) stream that can't be deselected.
        """
        if hasattr(self, 'outlet'):
            # Explicitly delete outlet to ensure immediate LSL cleanup
            del self.outlet
            print(f"✓ Closed marker stream: {self.stream_name} ({self.unique_id})")

    def __enter__(self):
        """Context manager support for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-cleanup on context exit."""
        self.close()
        return False


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

        # Marker sender (separate stream with unique ID to prevent zombie streams)
        self.marker_sender = LSLMarkerSender(
            "SSVEP-Markers",
            unique_id=f"SSVEP-Markers_{subject}_S{session:03d}"
        )

    def push_eeg(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """Push EEG sample to LSL."""
        if timestamp is None:
            timestamp = local_clock()
        self.eeg_outlet.push_sample(sample.tolist(), timestamp)

    def close(self):
        """Close all LSL outlets to prevent zombie streams in LabRecorder.

        This ensures streams are properly released and won't appear as
        red (disconnected) streams that can't be deselected.
        """
        if hasattr(self, 'eeg_outlet'):
            del self.eeg_outlet
            print(f"✓ Closed EEG stream: SSVEP-EEG_{self.subject}_S{self.session:03d}")

        if hasattr(self, 'marker_sender'):
            self.marker_sender.close()

    def __enter__(self):
        """Context manager support for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-cleanup on context exit."""
        self.close()
        return False


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
    max_markers = 0

    for stream in streams:
        stream_type = stream['info']['type'][0]
        if stream_type == 'EEG':
            eeg_stream = stream
        elif stream_type == 'Markers':
            # Pick the marker stream with the most data (handles duplicates/empty streams)
            n_markers = len(stream['time_stamps'])
            if n_markers > max_markers:
                marker_stream = stream
                max_markers = n_markers

    if eeg_stream is None:
        raise ValueError(f"No EEG stream in {xdf_path}")
    if marker_stream is None or max_markers == 0:
        raise ValueError(
            f"No valid Marker stream in {xdf_path}. "
            f"Make sure to select BOTH 'SSVEP-EEG' and 'SSVEP-Markers' in LabRecorder."
        )

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
                           baseline_sec: float = 0.5, delay_sec: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract trial epochs from XDF using STIM_ON markers (accurate onset).

    CRITICAL: Epochs are extracted from STIM_ON (LED turns on), NOT TRIAL_START.
    This ensures accurate alignment with stimulation onset for TRCA.

    Args:
        xdf_path: Path to XDF file
        config: SSVEPConfig instance
        trial_duration_sec: Duration of stimulation in seconds (default: 4.0)
        baseline_sec: Baseline period before stimulation (default: 0.5)
        delay_sec: Optional additional delay (default: 0, since STIM_ON is accurate)

    Returns:
        (trial_data, labels)
        - trial_data: (n_trials, n_channels, n_samples)
        - labels: (n_trials,) frequency indices
    """
    # Use 130ms visual processing delay (reference default)
    # Even with STIM_ON markers, brain response has ~130ms latency
    if delay_sec is None:
        delay_sec = 0.13  # 130ms visual processing latency (TEST 3)

    eeg_data, markers, metadata = load_xdf(xdf_path, config)

    fs = metadata['sampling_rate']
    timestamps = metadata['eeg_timestamps']

    stim_onsets = []
    trial_labels = []

    # Extract STIM_ON markers (accurate stimulation onset)
    for marker in markers:
        mdata = marker['data']
        label = mdata.get('label', '')

        # Look for STIM_ON markers (e.g., "STIM_ON_10", "STIM_ON_8p57")
        if label.startswith('STIM_ON'):
            stim_onsets.append(marker['timestamp'])
            target_freq = mdata.get('payload', {}).get('freq', -1)

            # Find frequency index
            freq_idx = -1
            for i, freq in enumerate(config.target_frequencies):
                if abs(freq - target_freq) < 0.01:
                    freq_idx = i
                    break
            trial_labels.append(freq_idx)

    # Extract epochs from STIM_ON (LED onset)
    n_delay = int(delay_sec * fs)  # Usually 0 (STIM_ON is accurate)
    n_baseline = int(baseline_sec * fs)
    n_trial = int(trial_duration_sec * fs)
    n_total = n_baseline + n_trial

    trials = []
    valid_labels = []

    for ts, label in zip(stim_onsets, trial_labels):
        # Find closest EEG timestamp to STIM_ON
        idx = np.argmin(np.abs(timestamps - ts))
        start_idx = idx + n_delay - n_baseline
        end_idx = start_idx + n_total

        if start_idx >= 0 and end_idx <= eeg_data.shape[1] and label >= 0:
            epoch = eeg_data[:, start_idx:end_idx]

            # CRITICAL FIX: Remove DC offset and linear trend per channel
            # This fixes massive DC drift (10k-15k µV) that destroys TRCA
            from scipy.signal import detrend
            epoch_detrended = detrend(epoch, axis=1, type='linear')

            trials.append(epoch_detrended)
            valid_labels.append(label)

    return np.array(trials), np.array(valid_labels)


# =============================================================================
# LAPLACIAN SPATIAL FILTER
# =============================================================================

_LAPLACIAN_NEIGHBORS = {
    "Pz":  ["P3",  "P4",  "PO3", "PO4"],
    "P3":  ["Pz",  "PO3", "O1"],
    "P4":  ["Pz",  "PO4", "O2"],
    "PO3": ["P3",  "Pz",  "O1",  "Oz"],
    "PO4": ["P4",  "Pz",  "O2",  "Oz"],
    "O1":  ["PO3", "Oz",  "P3"],
    "Oz":  ["O1",  "O2",  "PO3", "PO4"],
    "O2":  ["PO4", "Oz",  "P4"],
}


def _neighbors_indices(electrode_names: Tuple[str, ...]) -> Dict[int, List[int]]:
    """Convert electrode name neighbor mapping to index mapping."""
    name_to_idx = {name: idx for idx, name in enumerate(electrode_names)}
    idx_map: Dict[int, List[int]] = {}
    for i, name in enumerate(electrode_names):
        neighbors = _LAPLACIAN_NEIGHBORS.get(name, [])
        neighbor_idxs = []
        for n in neighbors:
            if n in name_to_idx:
                neighbor_idxs.append(name_to_idx[n])
        idx_map[i] = neighbor_idxs
    return idx_map


def apply_laplacian(data: np.ndarray, neighbor_idx_map: Dict[int, List[int]]) -> np.ndarray:
    """
    Compute surface Laplacian:
        out[ch, :] = data[ch, :] - mean(data[neighbors_of_ch, :])

    Args:
        data: (n_channels, n_samples)
        neighbor_idx_map: {ch_idx: [neighbor indices]}
    Returns:
        laplacian_applied_data: same shape
    """
    if data.ndim != 2:
        raise ValueError("apply_laplacian expects 2D array (n_channels, n_samples)")

    n_ch, n_samps = data.shape
    out = np.empty_like(data, dtype=float)

    for ch in range(n_ch):
        neigh = neighbor_idx_map.get(ch, [])
        if not neigh:
            out[ch] = data[ch]
        else:
            out[ch] = data[ch] - np.mean(data[neigh], axis=0)
    return out


# =============================================================================
# PREPROCESSING
# =============================================================================

class OnlinePreprocessor:
    """Causal (stateful) preprocessor for real-time use.

    Uses stateful filter bank with Laplacian spatial filtering.
    - Laplacian spatial filter
    - Filter bank: Multiple Chebyshev bandpass filters (6-90 Hz, 14-90 Hz, ...)
    - Notch: 60 Hz (applied within each filter band)
    """

    def __init__(self, config):
        self.config = config

        # Laplacian neighbor map
        self.neigh_map = _neighbors_indices(config.electrode_names)

        # Filter bank
        self.filterbank = FilterBank(fs=config.fs, num_bands=config.num_fbs)
        self.filter_states = []  # Will be initialized on first call

    def reset(self):
        """Reset filter states."""
        self.filter_states = []

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data causally (for online use).

        Args:
            data: (n_channels, n_samples)

        Returns:
            Preprocessed data (n_channels, n_samples, n_filterbanks)
        """
        # Laplacian spatial filter
        processed = apply_laplacian(data, self.neigh_map)

        # Apply filter bank (causal)
        filtered, self.filter_states = self.filterbank.apply_online(
            processed, self.filter_states
        )

        return filtered


class OfflinePreprocessor:
    """Zero-phase (non-causal) preprocessor for training.

    Uses filter bank with zero-phase filtering (filtfilt).
    - Laplacian spatial filter
    - Filter bank: Multiple Chebyshev bandpass filters
    - Notch: 60 Hz (applied within each filter band)
    """

    def __init__(self, config):
        self.config = config

        # Laplacian neighbor map
        self.neigh_map = _neighbors_indices(config.electrode_names)

        # Filter bank
        self.filterbank = FilterBank(fs=config.fs, num_bands=config.num_fbs)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data with zero-phase filtering (for offline training).

        Args:
            data: (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Preprocessed data with filter bank dimension:
                - 2D input: (n_channels, n_samples, n_filterbanks)
                - 3D input: (n_trials, n_channels, n_samples, n_filterbanks)
        """
        if data.ndim == 2:
            # Single trial: (n_channels, n_samples)
            # Laplacian
            processed = apply_laplacian(data, self.neigh_map)

            # Apply filter bank (zero-phase)
            filtered = self.filterbank.apply_offline(processed)

            return filtered

        elif data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_samples)
            n_trials = data.shape[0]
            filtered_trials = []

            for trial in range(n_trials):
                # Laplacian
                trial_lap = apply_laplacian(data[trial], self.neigh_map)

                # Apply filter bank (zero-phase)
                trial_filtered = self.filterbank.apply_offline(trial_lap)
                filtered_trials.append(trial_filtered)

            # Stack to (n_trials, n_channels, n_samples, n_filterbanks)
            return np.stack(filtered_trials, axis=0)

        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")