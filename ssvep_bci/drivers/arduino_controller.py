"""
Arduino Serial Controller for SSVEP BCI

Provides bidirectional communication with the Arduino LED stimulator:
- Send commands to start/stop LED flickering
- Send feedback commands to light up red LEDs based on classification
- Receive status updates from Arduino

Protocol:
    Commands (Python -> Arduino):
        "START"      - Start LED flickering
        "STOP"       - Stop all LEDs
        "FEEDBACK:N" - Light red LED N (0-3) for feedback
        "CLEAR"      - Clear feedback LEDs

    Responses (Arduino -> Python):
        "OK"         - Command acknowledged
        "RUNNING"    - LEDs are flickering
        "STOPPED"    - LEDs are stopped
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Optional, Callable
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import SSVEPConfig, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArduinoController:
    """Serial controller for Arduino LED stimulator.

    Handles communication with Arduino for:
    - Starting/stopping LED stimulation
    - Providing visual feedback based on BCI classification
    """

    # Map frequency to LED index (matching Arduino whitePins order)
    FREQ_TO_LED = {
        8.57: 0,   # D2
        10.0: 1,   # D3
        12.0: 2,   # D4
        15.0: 3,   # D5
    }

    def __init__(self, port: str = None, baudrate: int = 115200, timeout: float = 1.0):
        """Initialize Arduino controller.

        Args:
            port: Serial port (e.g., "COM3"). Auto-detects if None.
            baudrate: Serial baud rate (must match Arduino sketch)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.serial: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_running = False

        # Callback for status updates
        self._status_callback: Optional[Callable] = None

        # Reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()

    @staticmethod
    def list_ports() -> list:
        """List available serial ports.

        Returns:
            List of (port, description) tuples
        """
        ports = serial.tools.list_ports.comports()
        return [(p.device, p.description) for p in ports]

    @staticmethod
    def find_arduino() -> Optional[str]:
        """Try to auto-detect Arduino port.

        Returns:
            Port name if found, None otherwise
        """
        for port, desc in ArduinoController.list_ports():
            desc_lower = desc.lower()
            if 'arduino' in desc_lower or 'ch340' in desc_lower or 'usb serial' in desc_lower:
                logger.info(f"Found Arduino on {port}: {desc}")
                return port
        return None

    def connect(self, port: str = None) -> bool:
        """Connect to Arduino.

        Args:
            port: Serial port. Uses stored port or auto-detects if None.

        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.warning("Already connected")
            return True

        # Determine port
        target_port = port or self.port or self.find_arduino()
        if not target_port:
            logger.error("No Arduino port specified and auto-detect failed")
            return False

        try:
            self.serial = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )

            # Wait for Arduino to reset after serial connection
            time.sleep(2.0)

            # Flush any startup messages
            self.serial.reset_input_buffer()

            self.port = target_port
            self.is_connected = True
            logger.info(f"Connected to Arduino on {target_port}")

            # Start reader thread
            self._start_reader()

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Arduino."""
        self._stop_reader.set()

        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        if self.serial and self.serial.is_open:
            try:
                self.send_command("STOP")
                time.sleep(0.1)
                self.serial.close()
            except:
                pass

        self.serial = None
        self.is_connected = False
        self.is_running = False
        logger.info("Disconnected from Arduino")

    def _start_reader(self) -> None:
        """Start background reader thread."""
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        """Background thread to read Arduino responses."""
        while not self._stop_reader.is_set():
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()
                    if line:
                        self._handle_response(line)
            except Exception as e:
                if not self._stop_reader.is_set():
                    logger.error(f"Reader error: {e}")
            time.sleep(0.01)

    def _handle_response(self, response: str) -> None:
        """Handle response from Arduino.

        Args:
            response: Response string from Arduino
        """
        logger.debug(f"Arduino: {response}")

        if response == "RUNNING":
            self.is_running = True
        elif response == "STOPPED":
            self.is_running = False

        if self._status_callback:
            self._status_callback(response)

    def send_command(self, command: str) -> bool:
        """Send command to Arduino.

        Args:
            command: Command string

        Returns:
            True if sent successfully
        """
        if not self.is_connected or not self.serial:
            logger.warning("Not connected to Arduino")
            return False

        try:
            self.serial.write(f"{command}\n".encode('utf-8'))
            self.serial.flush()
            logger.debug(f"Sent: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def start_stimulation(self) -> bool:
        """Start LED flickering.

        Returns:
            True if command sent successfully
        """
        return self.send_command("START")

    def stop_stimulation(self) -> bool:
        """Stop all LEDs.

        Returns:
            True if command sent successfully
        """
        success = self.send_command("STOP")
        if success:
            self.is_running = False
        return success

    def show_feedback(self, frequency: float) -> bool:
        """Light up red LED corresponding to detected frequency.

        Args:
            frequency: Detected SSVEP frequency (8.57, 10.0, 12.0, or 15.0)

        Returns:
            True if command sent successfully
        """
        led_index = self.FREQ_TO_LED.get(frequency)
        if led_index is None:
            logger.warning(f"Unknown frequency: {frequency}")
            return False

        return self.send_command(f"FEEDBACK:{led_index}")

    def clear_feedback(self) -> bool:
        """Turn off all red feedback LEDs.

        Returns:
            True if command sent successfully
        """
        return self.send_command("CLEAR")

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for Arduino status updates.

        Args:
            callback: Function that takes a status string
        """
        self._status_callback = callback


class BCIFeedbackController:
    """High-level controller that connects BCI classification to Arduino feedback.

    Automatically sends feedback commands based on committed predictions.
    """

    def __init__(self, arduino: ArduinoController, feedback_duration: float = 0.5):
        """Initialize feedback controller.

        Args:
            arduino: ArduinoController instance
            feedback_duration: How long to show feedback LED (seconds)
        """
        self.arduino = arduino
        self.feedback_duration = feedback_duration

        self._last_feedback_time = 0
        self._last_feedback_freq = None
        self._feedback_timer: Optional[threading.Timer] = None

    def on_prediction(self, committed_prediction: Optional[float]) -> None:
        """Handle new committed prediction from BCI.

        Args:
            committed_prediction: Detected frequency or None
        """
        current_time = time.time()

        # Only show feedback for new predictions
        if committed_prediction is not None:
            if (committed_prediction != self._last_feedback_freq or
                current_time - self._last_feedback_time > self.feedback_duration):

                # Cancel any pending clear
                if self._feedback_timer:
                    self._feedback_timer.cancel()

                # Show feedback
                self.arduino.show_feedback(committed_prediction)
                self._last_feedback_freq = committed_prediction
                self._last_feedback_time = current_time

                # Schedule clear after duration
                self._feedback_timer = threading.Timer(
                    self.feedback_duration,
                    self._clear_feedback
                )
                self._feedback_timer.start()

    def _clear_feedback(self) -> None:
        """Clear feedback after timeout."""
        self.arduino.clear_feedback()
        self._last_feedback_freq = None


# Unit test
if __name__ == "__main__":
    print("Arduino Controller Test")
    print("=" * 50)

    # List ports
    print("\nAvailable serial ports:")
    for port, desc in ArduinoController.list_ports():
        print(f"  {port}: {desc}")

    # Try to find Arduino
    arduino_port = ArduinoController.find_arduino()
    if arduino_port:
        print(f"\nFound Arduino on: {arduino_port}")

        # Test connection
        controller = ArduinoController(arduino_port)

        if controller.connect():
            print("Connected!")

            # Test commands
            print("\nSending START...")
            controller.start_stimulation()
            time.sleep(2)

            print("Sending FEEDBACK:1 (10 Hz LED)...")
            controller.show_feedback(10.0)
            time.sleep(1)

            print("Clearing feedback...")
            controller.clear_feedback()
            time.sleep(1)

            print("Sending STOP...")
            controller.stop_stimulation()

            controller.disconnect()
            print("Test complete!")
        else:
            print("Failed to connect")
    else:
        print("\nNo Arduino found. Connect Arduino and try again.")
        print("Or specify port manually: ArduinoController('COM3')")
