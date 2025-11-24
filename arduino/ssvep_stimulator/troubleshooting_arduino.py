#!/usr/bin/env python3
"""Arduino troubleshooting script - run from anywhere"""

import sys
from pathlib import Path

# Add project root to path so we can find ssvep_bci
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ssvep_bci.drivers.arduino_controller import ArduinoController
import time

arduino = ArduinoController('COM4')
if arduino.connect():
    print('Connected!')
    
    print('Starting LEDs...')
    arduino.start_stimulation()
    time.sleep(3)
    
    print('Showing feedback on 8.57 Hz (far left LED)...')
    arduino.show_feedback(8.57)
    time.sleep(2)
    
    print('Stopping...')
    arduino.stop_stimulation()
    arduino.disconnect()
    print('Done!')