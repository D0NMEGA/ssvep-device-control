"""
Hardware Drivers - BrainFlow (EEG) and Arduino (LED Control)

Consolidated module for all hardware interfaces.
"""

import numpy as np
import time
import serial
import serial.tools.list_ports
from threading import Thread
from typing import Optional, List, Tuple
import logging

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# BRAINFLOW DRIVER (OpenBCI Cyton EEG)
# =============================================================================

class BrainFlowDriver:
    """Driver for OpenBCI Cyton using BrainFlow."""

    def __init__(self, config, use_synthetic: bool = False):
        self.config = config
        self.use_synthetic = use_synthetic
        self.board = None
        self.is_connected = False
        self.is_streaming = False
        self.eeg_channels = []
        self.sampling_rate = 0

        BoardShim.enable_dev_board_logger()

    @staticmethod
    def auto_detect_cyton() -> Optional[str]:
        """Auto-detect OpenBCI Cyton (looks for FTDI devices)."""
        for port in serial.tools.list_ports.comports():
            desc = port.description.lower()
            manufacturer = (port.manufacturer or "").lower()
            if 'ftdi' in desc or 'ftdi' in manufacturer or 'ft231x' in desc:
                logger.info(f"Found Cyton on {port.device}")
                return port.device
        return None

    def connect(self, serial_port: str = None) -> bool:
        """Connect to Cyton board."""
        if self.is_connected:
            return True

        try:
            params = BrainFlowInputParams()

            if self.use_synthetic:
                board_id = BoardIds.SYNTHETIC_BOARD.value
            else:
                board_id = self.config.board_id
                port = serial_port or self.config.serial_port or self.auto_detect_cyton()
                if port:
                    params.serial_port = port

            self.board = BoardShim(board_id, params)
            self.board.prepare_session()

            # Get channel info
            self.eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.sampling_rate = BoardShim.get_sampling_rate(board_id)

            self.is_connected = True
            logger.info(f"Connected! Sampling rate: {self.sampling_rate} Hz")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def start_stream(self) -> bool:
        """Start EEG data streaming."""
        if not self.is_connected:
            return False
        try:
            self.board.start_stream()
            self.is_streaming = True
            return True
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return False

    def stop_stream(self):
        """Stop EEG streaming."""
        if self.is_streaming:
            self.board.stop_stream()
            self.is_streaming = False

    def get_data(self) -> Optional[np.ndarray]:
        """Get available EEG data.

        Returns:
            (n_channels, n_samples) array or None
        """
        if not self.is_streaming:
            return None

        try:
            data = self.board.get_board_data()
            if data.shape[1] == 0:
                return None

            # Extract EEG channels
            eeg_data = data[self.eeg_channels, :]
            return eeg_data

        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    def disconnect(self):
        """Disconnect from board."""
        if self.is_streaming:
            self.stop_stream()
        if self.is_connected:
            self.board.release_session()
            self.is_connected = False


class SyntheticSSVEPDriver(BrainFlowDriver):
    """Synthetic SSVEP generator for testing without hardware."""

    def __init__(self, config, target_frequency: float = 10.0):
        super().__init__(config, use_synthetic=True)
        self.target_frequency = target_frequency
        self.sample_count = 0

    def get_data(self) -> Optional[np.ndarray]:
        """Generate synthetic SSVEP data."""
        if not self.is_streaming:
            return None

        # Generate ~10 samples
        n_samples = 10
        n_channels = len(self.eeg_channels)

        t = (self.sample_count + np.arange(n_samples)) / self.sampling_rate
        self.sample_count += n_samples

        # Synthetic SSVEP signal
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            ssvep = 20 * np.sin(2 * np.pi * self.target_frequency * t)
            noise = 10 * np.random.randn(n_samples)
            data[ch] = ssvep + noise

        return data


# =============================================================================
# ARDUINO CONTROLLER (LED Stimulation & Feedback)
# =============================================================================

class ArduinoController:
    """Arduino LED controller for SSVEP stimulation and feedback."""

    def __init__(self, port: str = None, baud_rate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial = None
        self.is_connected = False
        self.is_running = False

        # Button detection
        self.button_pressed = False
        self._expected_stop = False

    @staticmethod
    def auto_detect() -> Optional[str]:
        """Auto-detect Arduino (looks for CH340 or Arduino devices)."""
        for port in serial.tools.list_ports.comports():
            desc = port.description.lower()
            if 'arduino' in desc or 'ch340' in desc or 'ch341' in desc:
                logger.info(f"Found Arduino on {port.device}")
                return port.device
        return None

    def connect(self) -> bool:
        """Connect to Arduino."""
        if self.is_connected:
            return True

        try:
            port = self.port or self.auto_detect()
            if not port:
                logger.warning("No Arduino port found")
                return False

            self.serial = serial.Serial(port, self.baud_rate, timeout=self.timeout)
            time.sleep(2)  # Arduino resets on connection

            # Clear buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            self.port = port
            self.is_connected = True
            logger.info(f"Connected to Arduino on {port}")

            # Start response reader thread
            self._reader_thread = Thread(target=self._read_responses, daemon=True)
            self._reader_thread.start()

            return True

        except Exception as e:
            logger.error(f"Arduino connection failed: {e}")
            return False

    def _read_responses(self):
        """Background thread to read Arduino responses."""
        while self.is_connected:
            try:
                if self.serial and self.serial.in_waiting:
                    response = self.serial.readline().decode('utf-8').strip()
                    if response:
                        self._handle_response(response)
            except:
                pass
            time.sleep(0.01)

    def _handle_response(self, response: str):
        """Handle Arduino response."""
        logger.debug(f"Arduino: {response}")

        if response == "RUNNING":
            self.is_running = True
        elif response == "STOPPED":
            # Check if unexpected (button press)
            if not self._expected_stop and self.is_running:
                logger.warning("Button pressed - emergency stop!")
                self.button_pressed = True
            self.is_running = False
            self._expected_stop = False

    def send_command(self, command: str) -> bool:
        """Send command to Arduino."""
        if not self.is_connected:
            return False
        try:
            self.serial.write(f"{command}\n".encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return False

    def start_stimulation(self) -> bool:
        """Start LED flickering."""
        return self.send_command("START")

    def stop_stimulation(self) -> bool:
        """Stop all LEDs."""
        self._expected_stop = True
        return self.send_command("STOP")

    def show_feedback(self, frequency: float) -> bool:
        """Light feedback LED for frequency.

        Args:
            frequency: 8.57, 10.0, 12.0, or 15.0 Hz
        """
        freq_map = {8.57: 0, 10.0: 1, 12.0: 2, 15.0: 3}
        for f, idx in freq_map.items():
            if abs(frequency - f) < 0.01:
                return self.send_command(f"FEEDBACK:{idx}")
        return False

    def clear_feedback(self) -> bool:
        """Turn off all feedback LEDs."""
        return self.send_command("CLEAR")

    def check_button_pressed(self) -> bool:
        """Check if emergency stop button was pressed."""
        if self.button_pressed:
            self.button_pressed = False
            return True
        return False

    def disconnect(self):
        """Disconnect from Arduino."""
        if self.is_connected:
            self.stop_stimulation()
            time.sleep(0.1)
            self.serial.close()
            self.is_connected = False
