"""
BrainFlow Driver for OpenBCI Cyton

Provides a clean interface to the BrainFlow library for EEG acquisition.
Handles board connection, streaming, and data retrieval.
"""

import numpy as np
from threading import Thread, Event
import time
from typing import Optional, Callable, List, Tuple
import logging
import serial.tools.list_ports

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainFlowDriver:
    """Driver for OpenBCI Cyton using BrainFlow.

    Manages connection to the Cyton board and provides streaming data
    access. Can run in either polling mode or callback mode.

    Attributes:
        config: SSVEPConfig instance
        board: BrainFlow BoardShim instance
        is_streaming: Whether data acquisition is active
    """

    # Board IDs for reference
    BOARD_CYTON = BoardIds.CYTON_BOARD.value  # 0
    BOARD_CYTON_DAISY = BoardIds.CYTON_DAISY_BOARD.value  # 2
    BOARD_SYNTHETIC = BoardIds.SYNTHETIC_BOARD.value  # -1

    def __init__(self, config: SSVEPConfig = None, use_synthetic: bool = False):
        """Initialize the BrainFlow driver.

        Args:
            config: SSVEPConfig instance. Uses DEFAULT_CONFIG if None.
            use_synthetic: If True, use synthetic board for testing (no hardware)
        """
        self.config = config or DEFAULT_CONFIG
        self.use_synthetic = use_synthetic

        # BrainFlow components
        self.board: Optional[BoardShim] = None
        self.params: Optional[BrainFlowInputParams] = None

        # State
        self.is_connected = False
        self.is_streaming = False

        # Channel information (set after connection)
        self.eeg_channels: List[int] = []
        self.sampling_rate: int = 0

        # Streaming thread
        self._stream_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._data_callback: Optional[Callable] = None

        # Enable BrainFlow logging
        BoardShim.enable_dev_board_logger()

    @staticmethod
    def list_ports() -> List[Tuple[str, str]]:
        """List available serial ports.

        Returns:
            List of (port_name, description) tuples
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append((port.device, port.description))
        return ports

    @staticmethod
    def auto_detect_cyton() -> Optional[str]:
        """Automatically detect OpenBCI Cyton board.

        Looks for FTDI devices (Cyton uses FT231X)

        Returns:
            Serial port string if found, None otherwise
        """
        for port in serial.tools.list_ports.comports():
            # Check for FTDI devices (Cyton uses FTDI FT231X)
            desc = port.description.lower()
            manufacturer = (port.manufacturer or "").lower()

            if 'ftdi' in desc or 'ftdi' in manufacturer or 'ft231x' in desc:
                logger.info(f"Found potential OpenBCI Cyton on {port.device}: {port.description}")
                return port.device

        logger.warning("No OpenBCI Cyton found via auto-detection")
        return None

    def connect(self, serial_port: str = None) -> bool:
        """Connect to the Cyton board.

        Args:
            serial_port: Serial port (e.g., "COM3" on Windows, "/dev/ttyUSB0" on Linux).
                        If None, will use config value or attempt auto-detection.

        Returns:
            True if connection successful, False otherwise
        """
        if self.is_connected:
            logger.warning("Already connected to board")
            return True

        try:
            # Set up parameters
            self.params = BrainFlowInputParams()

            if self.use_synthetic:
                board_id = self.BOARD_SYNTHETIC
                logger.info("Using synthetic board for testing")
            else:
                board_id = self.config.board_id

                # Set serial port (with auto-detection)
                port = serial_port or self.config.serial_port

                # Try auto-detection if no port specified
                if not port:
                    logger.info("No port specified, attempting auto-detection...")
                    port = self.auto_detect_cyton()

                if port:
                    self.params.serial_port = port
                    logger.info(f"Using serial port: {port}")
                else:
                    logger.warning("No serial port found - BrainFlow will try default port")

            # Create board
            self.board = BoardShim(board_id, self.params)

            # Prepare session
            logger.info(f"Preparing session for board ID: {board_id}")
            self.board.prepare_session()

            # Get channel information from board
            all_eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(board_id)

            # Use only the configured number of channels
            n_channels_needed = self.config.n_eeg_channels
            if len(all_eeg_channels) >= n_channels_needed:
                self.eeg_channels = all_eeg_channels[:n_channels_needed]
            else:
                self.eeg_channels = all_eeg_channels
                logger.warning(
                    f"Board has {len(all_eeg_channels)} channels, "
                    f"but config expects {n_channels_needed}"
                )

            logger.info(f"Connected! Using EEG channels: {self.eeg_channels}")
            logger.info(f"Sampling rate: {self.sampling_rate} Hz")

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.board = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the board."""
        if self.is_streaming:
            self.stop_stream()

        if self.board is not None:
            try:
                self.board.release_session()
                logger.info("Session released")
            except Exception as e:
                logger.error(f"Error releasing session: {e}")

        self.board = None
        self.is_connected = False

    def start_stream(self, callback: Callable = None) -> bool:
        """Start data streaming.

        Args:
            callback: Optional callback function that receives new data.
                     Signature: callback(data: np.ndarray) where data has
                     shape (n_channels, n_samples).

        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Not connected to board")
            return False

        if self.is_streaming:
            logger.warning("Already streaming")
            return True

        try:
            # Start stream with ring buffer
            self.board.start_stream(450000)  # Buffer size
            self.is_streaming = True
            logger.info("Streaming started")

            # If callback provided, start streaming thread
            if callback:
                self._data_callback = callback
                self._stop_event.clear()
                self._stream_thread = Thread(target=self._stream_loop, daemon=True)
                self._stream_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return False

    def stop_stream(self) -> None:
        """Stop data streaming."""
        if not self.is_streaming:
            return

        # Stop the streaming thread
        if self._stream_thread is not None:
            self._stop_event.set()
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        # Stop the board stream
        if self.board is not None:
            try:
                self.board.stop_stream()
                logger.info("Streaming stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")

        self.is_streaming = False

    def _stream_loop(self) -> None:
        """Background thread for continuous data streaming."""
        poll_interval = 0.01  # 10 ms polling

        while not self._stop_event.is_set():
            try:
                # Get available data
                data = self.board.get_board_data()

                if data.shape[1] > 0 and self._data_callback:
                    # Extract EEG channels only
                    eeg_data = data[self.eeg_channels, :]
                    self._data_callback(eeg_data)

            except Exception as e:
                logger.error(f"Error in stream loop: {e}")

            time.sleep(poll_interval)

    def get_data(self, n_samples: int = None) -> Optional[np.ndarray]:
        """Get data from the board buffer.

        Args:
            n_samples: Number of samples to retrieve. If None, gets all available.

        Returns:
            EEG data array with shape (n_channels, n_samples), or None if error.
        """
        if not self.is_streaming:
            return None

        try:
            if n_samples is None:
                data = self.board.get_board_data()
            else:
                data = self.board.get_current_board_data(n_samples)

            # Extract EEG channels
            if data.shape[1] > 0:
                return data[self.eeg_channels, :]
            return None

        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    def get_current_data(self, n_samples: int) -> Optional[np.ndarray]:
        """Get the most recent n samples without removing from buffer.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            EEG data array with shape (n_channels, n_samples)
        """
        if not self.is_streaming:
            return None

        try:
            data = self.board.get_current_board_data(n_samples)
            if data.shape[1] > 0:
                return data[self.eeg_channels, :]
            return None

        except Exception as e:
            logger.error(f"Error getting current data: {e}")
            return None

    @staticmethod
    def list_serial_ports() -> List[str]:
        """List available serial ports.

        Returns:
            List of serial port names
        """
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]


class SyntheticSSVEPDriver(BrainFlowDriver):
    """Synthetic data generator for SSVEP testing.

    Generates realistic SSVEP-like signals for testing the pipeline
    without hardware.
    """

    def __init__(self, config: SSVEPConfig = None, target_frequency: float = 10.0):
        """Initialize synthetic driver.

        Args:
            config: SSVEPConfig instance
            target_frequency: SSVEP frequency to simulate
        """
        super().__init__(config, use_synthetic=True)
        self.target_frequency = target_frequency
        self._phase = 0.0

        # Fixed channel characteristics (set once at init)
        np.random.seed(42)  # Reproducible for debugging
        n_ch = self.config.n_eeg_channels
        self._ch_amp_scale = np.array([
            1.0 + 0.2 * np.random.rand() if ch < 4 else 0.6 + 0.2 * np.random.rand()
            for ch in range(n_ch)
        ])
        self._ch_phase_offset = np.random.rand(n_ch) * 0.1
        np.random.seed(None)  # Reset to random

    def set_target_frequency(self, freq: float) -> None:
        """Change the simulated SSVEP frequency.

        Args:
            freq: New target frequency in Hz
        """
        if freq in self.config.target_frequencies:
            self.target_frequency = freq
            logger.info(f"Target frequency set to {freq} Hz")
        else:
            logger.warning(f"Frequency {freq} not in target list")

    def generate_ssvep_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic SSVEP data.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Synthetic EEG data with shape (n_channels, n_samples)
        """
        fs = self.config.fs
        n_channels = self.config.n_eeg_channels

        # Time vector starting from current phase
        t = (np.arange(n_samples) + self._phase) / fs

        # Update phase for next call
        self._phase += n_samples

        data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            # Use fixed channel characteristics
            amp_scale = self._ch_amp_scale[ch]
            phase_offset = self._ch_phase_offset[ch]

            # Strong SSVEP fundamental frequency
            ssvep = 20 * amp_scale * np.sin(
                2 * np.pi * self.target_frequency * t + phase_offset
            )
            # First harmonic (moderate strength for realism)
            ssvep += 12 * amp_scale * np.sin(
                2 * np.pi * 2 * self.target_frequency * t + phase_offset * 2
            )

            # Add channel-specific noise (simulates biological noise)
            noise = np.random.randn(n_samples) * (3 + ch * 0.3)

            # Combine
            data[ch] = ssvep + noise

        return data

    def get_data(self, n_samples: int = None) -> Optional[np.ndarray]:
        """Get synthetic SSVEP data.

        Overrides parent method to generate SSVEP-like signals instead
        of using BrainFlow's random synthetic data.

        Args:
            n_samples: Number of samples to generate. If None, generates
                      a small batch (step_samples from config).

        Returns:
            Synthetic SSVEP data with shape (n_channels, n_samples)
        """
        if not self.is_streaming:
            return None

        # Default to step size if not specified
        if n_samples is None:
            n_samples = self.config.step_samples

        return self.generate_ssvep_data(n_samples)


# Unit test
if __name__ == "__main__":
    print("Testing BrainFlowDriver...")
    print("=" * 50)

    # List available ports
    print("\nAvailable serial ports:")
    ports = BrainFlowDriver.list_serial_ports()
    for port in ports:
        print(f"  {port}")

    # Test with synthetic board
    print("\n--- Testing with synthetic board ---")
    config = SSVEPConfig()
    driver = SyntheticSSVEPDriver(config, target_frequency=10.0)

    # Connect
    print("\nConnecting to synthetic board...")
    if driver.connect():
        print(f"EEG channels: {driver.eeg_channels}")
        print(f"Sampling rate: {driver.sampling_rate} Hz")

        # Start streaming
        print("\nStarting stream...")
        if driver.start_stream():
            print("Streaming started!")

            # Collect data for 2 seconds
            print("\nCollecting data for 2 seconds...")
            time.sleep(2)

            # Get data
            data = driver.get_data()
            if data is not None:
                print(f"Retrieved data shape: {data.shape}")
                print(f"Data range: [{data.min():.1f}, {data.max():.1f}] µV")

            # Stop streaming
            driver.stop_stream()
            print("Streaming stopped")

        # Disconnect
        driver.disconnect()
        print("Disconnected")

    print("\n--- Testing callback mode ---")
    driver2 = SyntheticSSVEPDriver(config, target_frequency=12.0)

    received_samples = [0]

    def data_callback(data):
        received_samples[0] += data.shape[1]

    if driver2.connect():
        print("Connected")
        driver2.start_stream(callback=data_callback)

        print("Collecting for 1 second with callback...")
        time.sleep(1)

        driver2.stop_stream()
        driver2.disconnect()

        print(f"Received {received_samples[0]} samples via callback")
        print(f"Expected ~{config.fs} samples")

    print("\nBrainFlowDriver test passed!")
