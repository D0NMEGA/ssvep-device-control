"""
SSVEP Calibration Module

Collects calibration data for personalized SSVEP classification.
Uses single-LED paradigm: one frequency at a time with cued trials.

Data Format:
- .npz file with EEG epochs and labels
- .json file with metadata (frequencies, timestamps, channel info)

Event IDs:
- 1: 15.0 Hz
- 2: 12.0 Hz
- 3: 10.0 Hz
- 4: 8.57 Hz
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import SSVEPConfig


# Event ID mapping
FREQ_TO_EVENT_ID = {
    15.0: 1,
    12.0: 2,
    10.0: 3,
    8.57: 4,
}

EVENT_ID_TO_FREQ = {v: k for k, v in FREQ_TO_EVENT_ID.items()}


@dataclass
class CalibrationTrial:
    """Single calibration trial."""
    frequency: float
    event_id: int
    start_time: float  # Epoch timestamp
    duration: float    # Seconds
    data: np.ndarray = None  # (n_channels, n_samples)


@dataclass
class CalibrationSession:
    """Complete calibration session."""
    subject_id: str
    session_time: str
    config: SSVEPConfig

    # Session parameters
    trial_duration: float = 4.0      # Seconds per trial
    rest_duration: float = 2.0       # Seconds between trials
    trials_per_frequency: int = 5    # Repetitions per frequency

    # Collected data
    trials: List[CalibrationTrial] = field(default_factory=list)

    # State
    is_complete: bool = False
    current_trial: int = 0
    total_trials: int = 0

    def __post_init__(self):
        """Calculate total trials."""
        self.total_trials = len(self.config.target_frequencies) * self.trials_per_frequency


@dataclass
class CalibrationData:
    """Processed calibration data for saving/loading."""
    # EEG data
    epochs: np.ndarray  # (n_trials, n_channels, n_samples)
    labels: np.ndarray  # (n_trials,) - frequency values
    event_ids: np.ndarray  # (n_trials,) - event IDs

    # Metadata
    subject_id: str
    session_time: str
    fs: int
    channel_names: Tuple[str, ...]
    frequencies: Tuple[float, ...]
    trial_duration: float
    trials_per_frequency: int


class CalibrationCollector:
    """Collects calibration data for SSVEP classification.

    Paradigm:
    1. For each frequency (randomized order):
       a. Show cue indicating which LED to look at
       b. Turn on only that LED for trial_duration seconds
       c. Rest period with all LEDs off
    2. Repeat trials_per_frequency times per frequency
    3. Save data to .npz and metadata to .json
    """

    def __init__(
        self,
        config: SSVEPConfig = None,
        trial_duration: float = 4.0,
        rest_duration: float = 2.0,
        trials_per_frequency: int = 5,
        subject_id: str = "default"
    ):
        """Initialize calibration collector.

        Args:
            config: SSVEPConfig instance
            trial_duration: Duration of each trial in seconds
            rest_duration: Rest period between trials in seconds
            trials_per_frequency: Number of trials per frequency
            subject_id: Subject identifier
        """
        self.config = config or SSVEPConfig()

        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session = CalibrationSession(
            subject_id=subject_id,
            session_time=session_time,
            config=self.config,
            trial_duration=trial_duration,
            rest_duration=rest_duration,
            trials_per_frequency=trials_per_frequency
        )

        # Create trial sequence (randomized)
        self._trial_sequence = self._create_trial_sequence()

        # Callbacks
        self._on_trial_start: Optional[Callable] = None
        self._on_trial_end: Optional[Callable] = None
        self._on_rest_start: Optional[Callable] = None
        self._on_progress: Optional[Callable] = None

        # Current trial buffer
        self._current_data: List[np.ndarray] = []

    def _create_trial_sequence(self) -> List[float]:
        """Create randomized trial sequence.

        Returns:
            List of frequencies in trial order
        """
        sequence = []
        for freq in self.config.target_frequencies:
            sequence.extend([freq] * self.session.trials_per_frequency)

        # Shuffle
        np.random.shuffle(sequence)
        return sequence

    def get_trial_sequence(self) -> List[Tuple[int, float]]:
        """Get the complete trial sequence with indices.

        Returns:
            List of (trial_index, frequency) tuples
        """
        return [(i, freq) for i, freq in enumerate(self._trial_sequence)]

    def get_current_trial_info(self) -> Tuple[int, float, int]:
        """Get info about current trial.

        Returns:
            Tuple of (trial_index, frequency, event_id)
        """
        if self.session.current_trial >= len(self._trial_sequence):
            return (-1, 0.0, 0)

        freq = self._trial_sequence[self.session.current_trial]
        event_id = FREQ_TO_EVENT_ID.get(freq, 0)
        return (self.session.current_trial, freq, event_id)

    def start_trial(self) -> Tuple[float, int]:
        """Start the current trial.

        Returns:
            Tuple of (frequency, event_id) for this trial
        """
        trial_idx, freq, event_id = self.get_current_trial_info()

        if trial_idx < 0:
            return (0.0, 0)

        # Clear buffer
        self._current_data = []

        # Create trial
        trial = CalibrationTrial(
            frequency=freq,
            event_id=event_id,
            start_time=time.time(),
            duration=self.session.trial_duration
        )
        self.session.trials.append(trial)

        # Callback
        if self._on_trial_start:
            self._on_trial_start(trial_idx, freq, event_id)

        return (freq, event_id)

    def add_data(self, data: np.ndarray) -> None:
        """Add EEG data to current trial buffer.

        Args:
            data: Array of shape (n_channels, n_samples)
        """
        if len(self.session.trials) > 0:
            self._current_data.append(data.copy())

    def end_trial(self) -> bool:
        """End the current trial and save data.

        Returns:
            True if more trials remain
        """
        if len(self.session.trials) == 0:
            return False

        # Concatenate trial data
        if self._current_data:
            trial_data = np.concatenate(self._current_data, axis=1)
            self.session.trials[-1].data = trial_data

        # Callback
        if self._on_trial_end:
            freq = self.session.trials[-1].frequency
            self._on_trial_end(self.session.current_trial, freq)

        # Advance
        self.session.current_trial += 1

        # Report progress
        if self._on_progress:
            self._on_progress(self.session.current_trial, self.session.total_trials)

        # Check if complete
        if self.session.current_trial >= self.session.total_trials:
            self.session.is_complete = True
            return False

        return True

    def start_rest(self) -> None:
        """Signal start of rest period."""
        if self._on_rest_start:
            self._on_rest_start(self.session.rest_duration)

    def set_callbacks(
        self,
        on_trial_start: Callable = None,
        on_trial_end: Callable = None,
        on_rest_start: Callable = None,
        on_progress: Callable = None
    ):
        """Set callback functions.

        Args:
            on_trial_start: Called with (trial_idx, freq, event_id)
            on_trial_end: Called with (trial_idx, freq)
            on_rest_start: Called with (rest_duration)
            on_progress: Called with (current, total)
        """
        self._on_trial_start = on_trial_start
        self._on_trial_end = on_trial_end
        self._on_rest_start = on_rest_start
        self._on_progress = on_progress

    def get_calibration_data(self) -> Optional[CalibrationData]:
        """Process and return calibration data.

        Returns:
            CalibrationData object or None if not complete
        """
        if not self.session.is_complete:
            return None

        # Find minimum trial length (for consistent epoch size)
        min_samples = min(t.data.shape[1] for t in self.session.trials if t.data is not None)
        n_channels = self.config.n_eeg_channels

        # Create epochs array
        n_trials = len(self.session.trials)
        epochs = np.zeros((n_trials, n_channels, min_samples))
        labels = np.zeros(n_trials)
        event_ids = np.zeros(n_trials, dtype=int)

        for i, trial in enumerate(self.session.trials):
            if trial.data is not None:
                epochs[i] = trial.data[:n_channels, :min_samples]
                labels[i] = trial.frequency
                event_ids[i] = trial.event_id

        return CalibrationData(
            epochs=epochs,
            labels=labels,
            event_ids=event_ids,
            subject_id=self.session.subject_id,
            session_time=self.session.session_time,
            fs=self.config.fs,
            channel_names=self.config.electrode_names,
            frequencies=self.config.target_frequencies,
            trial_duration=self.session.trial_duration,
            trials_per_frequency=self.session.trials_per_frequency
        )

    def save(self, directory: str = "calibration") -> Tuple[str, str]:
        """Save calibration data to files, organized by subject.

        Args:
            directory: Base output directory

        Returns:
            Tuple of (npz_path, json_path)
        """
        data = self.get_calibration_data()
        if data is None:
            raise ValueError("Calibration not complete")

        # Create subject-specific directory
        subject_dir = Path(directory) / data.subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        # File names (session timestamp)
        base_name = f"session_{data.session_time}"
        npz_path = subject_dir / f"{base_name}.npz"
        json_path = subject_dir / f"{base_name}.json"

        # Save epochs
        np.savez_compressed(
            npz_path,
            epochs=data.epochs,
            labels=data.labels,
            event_ids=data.event_ids
        )

        # Save metadata
        metadata = {
            "subject_id": data.subject_id,
            "session_time": data.session_time,
            "fs": data.fs,
            "channel_names": list(data.channel_names),
            "frequencies": list(data.frequencies),
            "trial_duration": data.trial_duration,
            "trials_per_frequency": data.trials_per_frequency,
            "n_trials": len(data.labels),
            "n_channels": data.epochs.shape[1],
            "n_samples_per_trial": data.epochs.shape[2],
            "event_id_mapping": FREQ_TO_EVENT_ID
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return (str(npz_path), str(json_path))

    @staticmethod
    def load(npz_path: str) -> CalibrationData:
        """Load calibration data from files.

        Args:
            npz_path: Path to .npz file

        Returns:
            CalibrationData object
        """
        npz_path = Path(npz_path)
        json_path = npz_path.with_suffix('.json')

        # Load epochs
        with np.load(npz_path) as data:
            epochs = data['epochs']
            labels = data['labels']
            event_ids = data['event_ids']

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        return CalibrationData(
            epochs=epochs,
            labels=labels,
            event_ids=event_ids,
            subject_id=metadata['subject_id'],
            session_time=metadata['session_time'],
            fs=metadata['fs'],
            channel_names=tuple(metadata['channel_names']),
            frequencies=tuple(metadata['frequencies']),
            trial_duration=metadata['trial_duration'],
            trials_per_frequency=metadata['trials_per_frequency']
        )

    @staticmethod
    def combine_sessions(session_paths: List[str], output_path: str = None) -> CalibrationData:
        """Combine multiple calibration sessions into one dataset.

        This enables incremental learning - each new session adds to your
        template library for more robust classification.

        Args:
            session_paths: List of paths to session .npz files
            output_path: Optional path to save combined data

        Returns:
            Combined CalibrationData
        """
        if not session_paths:
            raise ValueError("No session paths provided")

        # Load all sessions
        sessions = [CalibrationCollector.load(p) for p in session_paths]

        # Verify compatibility
        first = sessions[0]
        for s in sessions[1:]:
            if s.fs != first.fs:
                raise ValueError(f"Sampling rate mismatch: {s.fs} vs {first.fs}")
            if s.channel_names != first.channel_names:
                raise ValueError("Channel configuration mismatch")

        # Find minimum epoch length (for consistency)
        min_samples = min(s.epochs.shape[2] for s in sessions)

        # Combine epochs
        all_epochs = []
        all_labels = []
        all_event_ids = []

        for s in sessions:
            # Trim to min length
            trimmed = s.epochs[:, :, :min_samples]
            all_epochs.append(trimmed)
            all_labels.append(s.labels)
            all_event_ids.append(s.event_ids)

        combined_epochs = np.concatenate(all_epochs, axis=0)
        combined_labels = np.concatenate(all_labels)
        combined_event_ids = np.concatenate(all_event_ids)

        # Count trials per frequency
        total_trials_per_freq = sum(s.trials_per_frequency for s in sessions)

        combined = CalibrationData(
            epochs=combined_epochs,
            labels=combined_labels,
            event_ids=combined_event_ids,
            subject_id=first.subject_id,
            session_time="combined",
            fs=first.fs,
            channel_names=first.channel_names,
            frequencies=first.frequencies,
            trial_duration=first.trial_duration,
            trials_per_frequency=total_trials_per_freq
        )

        # Save if output path specified
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                out_path,
                epochs=combined.epochs,
                labels=combined.labels,
                event_ids=combined.event_ids
            )

            # Metadata
            json_path = out_path.with_suffix('.json')
            metadata = {
                "subject_id": combined.subject_id,
                "session_time": "combined",
                "fs": combined.fs,
                "channel_names": list(combined.channel_names),
                "frequencies": list(combined.frequencies),
                "trial_duration": combined.trial_duration,
                "trials_per_frequency": combined.trials_per_frequency,
                "n_trials": len(combined.labels),
                "n_channels": combined.epochs.shape[1],
                "n_samples_per_trial": combined.epochs.shape[2],
                "source_sessions": session_paths,
                "event_id_mapping": FREQ_TO_EVENT_ID
            }
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Combined {len(sessions)} sessions -> {out_path}")

        return combined

    @staticmethod
    def get_subject_sessions(subject_id: str, calibration_dir: str = "calibration") -> List[str]:
        """Get all calibration sessions for a subject.

        Args:
            subject_id: Subject identifier
            calibration_dir: Base calibration directory

        Returns:
            List of session .npz file paths, sorted by date (oldest first)
        """
        subject_dir = Path(calibration_dir) / subject_id
        if not subject_dir.exists():
            return []

        sessions = list(subject_dir.glob("session_*.npz"))
        # Sort by filename (which includes timestamp)
        sessions.sort()
        return [str(s) for s in sessions]

    @staticmethod
    def get_latest_session(subject_id: str, calibration_dir: str = "calibration") -> Optional[str]:
        """Get the most recent calibration session for a subject.

        Args:
            subject_id: Subject identifier
            calibration_dir: Base calibration directory

        Returns:
            Path to latest session .npz file, or None
        """
        sessions = CalibrationCollector.get_subject_sessions(subject_id, calibration_dir)
        return sessions[-1] if sessions else None

    @staticmethod
    def get_combined_calibration(
        subject_id: str,
        calibration_dir: str = "calibration",
        max_sessions: int = None
    ) -> Optional[CalibrationData]:
        """Get combined calibration from all (or recent) sessions.

        Args:
            subject_id: Subject identifier
            calibration_dir: Base calibration directory
            max_sessions: Maximum number of recent sessions to use (None = all)

        Returns:
            Combined CalibrationData or None if no sessions
        """
        sessions = CalibrationCollector.get_subject_sessions(subject_id, calibration_dir)

        if not sessions:
            return None

        # Limit to recent sessions if specified
        if max_sessions and len(sessions) > max_sessions:
            sessions = sessions[-max_sessions:]

        if len(sessions) == 1:
            return CalibrationCollector.load(sessions[0])

        return CalibrationCollector.combine_sessions(sessions)


# Unit test
if __name__ == "__main__":
    print("Calibration Module Test")
    print("=" * 50)

    # Create collector
    collector = CalibrationCollector(
        trial_duration=2.0,
        rest_duration=1.0,
        trials_per_frequency=2,
        subject_id="test"
    )

    print(f"\nTotal trials: {collector.session.total_trials}")
    print("Trial sequence:")
    for idx, freq in collector.get_trial_sequence():
        event_id = FREQ_TO_EVENT_ID[freq]
        print(f"  Trial {idx}: {freq:.2f} Hz (event_id={event_id})")

    # Simulate data collection
    print("\nSimulating data collection...")
    for i in range(collector.session.total_trials):
        freq, event_id = collector.start_trial()
        print(f"  Trial {i}: {freq:.2f} Hz started")

        # Simulate EEG data (4 seconds at 250 Hz = 1000 samples)
        fake_data = np.random.randn(8, 1000)
        collector.add_data(fake_data)

        collector.end_trial()

    print(f"\nSession complete: {collector.session.is_complete}")

    # Get data
    data = collector.get_calibration_data()
    if data:
        print(f"\nEpochs shape: {data.epochs.shape}")
        print(f"Labels: {data.labels}")
        print(f"Event IDs: {data.event_ids}")

        # Save
        npz_path, json_path = collector.save("calibration_test")
        print(f"\nSaved to:")
        print(f"  {npz_path}")
        print(f"  {json_path}")

        # Load and verify
        loaded = CalibrationCollector.load(npz_path)
        print(f"\nLoaded epochs shape: {loaded.epochs.shape}")
        print(f"Subject: {loaded.subject_id}")
