from ssvep_bci.drivers.arduino_controller import ArduinoController
import time

arduino = ArduinoController('COM4')
if arduino.connect():
    print('Connected!')
    
    print('Starting LEDs...')
    arduino.start_stimulation()
    time.sleep(3)
    
    print('Showing feedback on LED 0 (8.57 Hz)...')
    arduino.show_feedback(8.57)
    time.sleep(2)
    
    print('Stopping...')
    arduino.stop_stimulation()
    arduino.disconnect()
    print('Done!')