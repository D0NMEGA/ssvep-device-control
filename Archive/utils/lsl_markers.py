"""
LSL Marker Stream Utilities

Provides LSL marker streaming for event synchronization with EEG data.
All markers include an integer ID, short label, and JSON payload for detailed metadata.

Event ID Table:
- 1:  START_CALIBRATION
- 2:  BASELINE_START
- 3:  BASELINE_END
- 10: TRIAL_START
- 11: STIM_ON_8p57
- 12: STIM_OFF_8p57
- 13: STIM_ON_10
- 14: STIM_OFF_10
- 15: STIM_ON_12
- 16: STIM_OFF_12
- 17: STIM_ON_15
- 18: STIM_OFF_15
- 20: REST_START
- 21: REST_END
- 99: END_CALIBRATION
"""

import json
from typing import Optional, Dict, Any
from pylsl import StreamInfo, StreamOutlet


class EventMarkers:
    """Event ID constants and labels."""

    # Session-level events
    START_CALIBRATION = 1
    BASELINE_START = 2
    BASELINE_END = 3
    END_CALIBRATION = 99

    # Trial-level events
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

    # Rest period events
    REST_START = 20
    REST_END = 21

    # Event labels (for human readability)
    LABELS = {
        1: "START_CALIBRATION",
        2: "BASELINE_START",
        3: "BASELINE_END",
        10: "TRIAL_START",
        11: "STIM_ON_8p57",
        12: "STIM_OFF_8p57",
        13: "STIM_ON_10",
        14: "STIM_OFF_10",
        15: "STIM_ON_12",
        16: "STIM_OFF_12",
        17: "STIM_ON_15",
        18: "STIM_OFF_15",
        20: "REST_START",
        21: "REST_END",
        99: "END_CALIBRATION",
    }

    @classmethod
    def get_label(cls, event_id: int) -> str:
        """Get label for an event ID."""
        return cls.LABELS.get(event_id, f"UNKNOWN_{event_id}")

    @classmethod
    def get_stim_on_id(cls, freq: float) -> int:
        """Get STIM_ON event ID for a frequency."""
        freq_map = {
            8.57: cls.STIM_ON_8p57,
            10.0: cls.STIM_ON_10,
            12.0: cls.STIM_ON_12,
            15.0: cls.STIM_ON_15,
        }
        # Try exact match first
        if freq in freq_map:
            return freq_map[freq]
        # Try approximate match (within 0.01 Hz)
        for f, event_id in freq_map.items():
            if abs(freq - f) < 0.01:
                return event_id
        raise ValueError(f"Unknown frequency: {freq}")

    @classmethod
    def get_stim_off_id(cls, freq: float) -> int:
        """Get STIM_OFF event ID for a frequency."""
        freq_map = {
            8.57: cls.STIM_OFF_8p57,
            10.0: cls.STIM_OFF_10,
            12.0: cls.STIM_OFF_12,
            15.0: cls.STIM_OFF_15,
        }
        # Try exact match first
        if freq in freq_map:
            return freq_map[freq]
        # Try approximate match (within 0.01 Hz)
        for f, event_id in freq_map.items():
            if abs(freq - f) < 0.01:
                return event_id
        raise ValueError(f"Unknown frequency: {freq}")


class LSLMarkerSender:
    """LSL marker stream sender for event synchronization.

    Creates an LSL outlet for sending event markers with:
    - Integer event ID
    - Short label string
    - JSON payload with detailed metadata

    Example usage:
        sender = LSLMarkerSender("SSVEPMarkers")
        sender.send_start_calibration(subject="S01", session=1)
        sender.send_trial_start(trial=1, target_freq=10.0)
        sender.send_stim_on(freq=10.0, trial=1)
        # ... EEG recording ...
        sender.send_stim_off(freq=10.0, trial=1)
        sender.send_end_calibration()
    """

    def __init__(self, stream_name: str = "SSVEPMarkers"):
        """Initialize LSL marker sender.

        Args:
            stream_name: Name of the LSL marker stream
        """
        self.stream_name = stream_name

        # Create LSL stream info
        # Type: "Markers" (standard for event markers)
        # Format: string (will send JSON strings)
        # Channels: 1 (single marker stream)
        info = StreamInfo(
            name=stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,  # Irregular rate (event-based)
            channel_format='string',
            source_id=f"{stream_name}_events"
        )

        # Add metadata
        desc = info.desc()
        desc.append_child_value("manufacturer", "SSVEP-BCI")
        desc.append_child_value("version", "1.0")

        # Add event ID descriptions
        events = desc.append_child("events")
        for event_id, label in EventMarkers.LABELS.items():
            event = events.append_child("event")
            event.append_child_value("id", str(event_id))
            event.append_child_value("label", label)

        # Create outlet
        self.outlet = StreamOutlet(info)

    def send_marker(self, event_id: int, payload: Optional[Dict[str, Any]] = None):
        """Send a marker with event ID and optional payload.

        Args:
            event_id: Integer event ID (see EventMarkers constants)
            payload: Optional dictionary with event-specific data
        """
        # Get label
        label = EventMarkers.get_label(event_id)

        # Build marker data structure
        marker_data = {
            "id": event_id,
            "label": label,
            "payload": payload or {}
        }

        # Convert to JSON string
        marker_string = json.dumps(marker_data)

        # Send to LSL
        self.outlet.push_sample([marker_string])

    # Session-level events

    def send_start_calibration(self, **kwargs):
        """Send START_CALIBRATION marker.

        Args:
            **kwargs: Additional metadata (e.g., subject, session, date)
        """
        self.send_marker(EventMarkers.START_CALIBRATION, kwargs)

    def send_end_calibration(self, **kwargs):
        """Send END_CALIBRATION marker.

        Args:
            **kwargs: Additional metadata (e.g., n_trials, duration_sec)
        """
        self.send_marker(EventMarkers.END_CALIBRATION, kwargs)

    def send_baseline_start(self, **kwargs):
        """Send BASELINE_START marker.

        Args:
            **kwargs: Additional metadata (e.g., duration_sec)
        """
        self.send_marker(EventMarkers.BASELINE_START, kwargs)

    def send_baseline_end(self, **kwargs):
        """Send BASELINE_END marker.

        Args:
            **kwargs: Additional metadata
        """
        self.send_marker(EventMarkers.BASELINE_END, kwargs)

    # Trial-level events

    def send_trial_start(self, trial: int, target_freq: float, **kwargs):
        """Send TRIAL_START marker.

        Args:
            trial: Trial number (1-indexed)
            target_freq: Target frequency in Hz
            **kwargs: Additional metadata
        """
        payload = {"trial": trial, "target_freq": target_freq}
        payload.update(kwargs)
        self.send_marker(EventMarkers.TRIAL_START, payload)

    # Stimulation events

    def send_stim_on(self, freq: float, trial: int, **kwargs):
        """Send STIM_ON marker for a frequency.

        Args:
            freq: Stimulation frequency (8.57, 10, 12, or 15 Hz)
            trial: Trial number
            **kwargs: Additional metadata
        """
        event_id = EventMarkers.get_stim_on_id(freq)
        payload = {"trial": trial, "event": "stim_on", "freq": freq}
        payload.update(kwargs)
        self.send_marker(event_id, payload)

    def send_stim_off(self, freq: float, trial: int, **kwargs):
        """Send STIM_OFF marker for a frequency.

        Args:
            freq: Stimulation frequency (8.57, 10, 12, or 15 Hz)
            trial: Trial number
            **kwargs: Additional metadata
        """
        event_id = EventMarkers.get_stim_off_id(freq)
        payload = {"trial": trial, "event": "stim_off", "freq": freq}
        payload.update(kwargs)
        self.send_marker(event_id, payload)

    # Rest period events

    def send_rest_start(self, trial: int, **kwargs):
        """Send REST_START marker.

        Args:
            trial: Trial number
            **kwargs: Additional metadata (e.g., duration_sec)
        """
        payload = {"trial": trial, "event": "rest_start"}
        payload.update(kwargs)
        self.send_marker(EventMarkers.REST_START, payload)

    def send_rest_end(self, trial: int, **kwargs):
        """Send REST_END marker.

        Args:
            trial: Trial number
            **kwargs: Additional metadata
        """
        payload = {"trial": trial, "event": "rest_end"}
        payload.update(kwargs)
        self.send_marker(EventMarkers.REST_END, payload)


# Unit test
if __name__ == "__main__":
    print("Testing LSLMarkerSender...")

    # Create sender
    sender = LSLMarkerSender("SSVEPMarkers_Test")
    print(f"Created LSL marker stream: {sender.stream_name}")

    # Test session-level events
    print("\nSending session-level events:")
    sender.send_start_calibration(subject="S01", session=1, date="2025-01-15")
    print("  -> START_CALIBRATION")

    sender.send_baseline_start(duration_sec=30)
    print("  -> BASELINE_START")

    sender.send_baseline_end()
    print("  -> BASELINE_END")

    # Test trial events
    print("\nSending trial events:")
    frequencies = [8.57, 10.0, 12.0, 15.0]

    for i, freq in enumerate(frequencies, start=1):
        print(f"\n  Trial {i} (target={freq} Hz):")

        sender.send_trial_start(trial=i, target_freq=freq)
        print(f"    -> TRIAL_START")

        sender.send_stim_on(freq=freq, trial=i)
        print(f"    -> STIM_ON_{freq}")

        # Simulate stimulation period
        import time
        time.sleep(0.1)

        sender.send_stim_off(freq=freq, trial=i)
        print(f"    -> STIM_OFF_{freq}")

        sender.send_rest_start(trial=i, duration_sec=2.0)
        print(f"    -> REST_START")

        time.sleep(0.05)

        sender.send_rest_end(trial=i)
        print(f"    -> REST_END")

    # End calibration
    print("\nSending end calibration:")
    sender.send_end_calibration(n_trials=4, duration_sec=60.0)
    print("  -> END_CALIBRATION")

    print("\nLSLMarkerSender test passed!")
    print("\nEvent ID Reference:")
    print("-" * 50)
    for event_id in sorted(EventMarkers.LABELS.keys()):
        label = EventMarkers.LABELS[event_id]
        print(f"  {event_id:2d}: {label}")
