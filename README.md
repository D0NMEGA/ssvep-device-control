# SSVEP Brain-Computer Interface

A real-time Steady-State Visual Evoked Potential (SSVEP) Brain-Computer Interface system using OpenBCI Cyton and Arduino-based LED stimulation.

## Overview

This project implements a 4-class SSVEP BCI that:
- Acquires 8-channel EEG at 250 Hz from OpenBCI Cyton
- Classifies user attention to flickering LEDs using Canonical Correlation Analysis (CCA)
- Achieves <300ms total latency with 250ms analysis windows
- Targets ≥90% classification accuracy

### Target Frequencies (Arduino LED Flicker Rates)
| LED Pin | Frequency | Half-Period |
|---------|-----------|-------------|
| D2 | 8.57 Hz | 58,333 µs |
| D3 | 10.0 Hz | 50,000 µs |
| D4 | 12.0 Hz | 41,667 µs |
| D5 | 15.0 Hz | 33,333 µs |

## Hardware Setup

### Required Components
- **EEG**: OpenBCI Cyton (8 channels, 250 Hz)
- **Stimulator**: Arduino Mega with LED array
  - 4 white LEDs on pins D2-D5 (SSVEP targets)
  - 4 red LEDs on pins D6-D9 (visual feedback chaser)
  - Push button on D10 (start/stop)

### EEG Electrode Montage
8-channel occipital/parietal placement optimized for SSVEP:

| Channel | Electrode | X | Y | Z |
|---------|-----------|-------|-------|-------|
| 1 | Pz | 0.000 | -0.587 | 0.809 |
| 2 | P3 | -0.444 | -0.587 | 0.678 |
| 3 | P4 | +0.444 | -0.587 | 0.678 |
| 4 | PO3 | -0.518 | -0.743 | 0.425 |
| 5 | PO4 | +0.518 | -0.743 | 0.425 |
| 6 | O1 | -0.309 | -0.951 | 0.000 |
| 7 | Oz | 0.000 | -1.000 | 0.000 |
| 8 | O2 | +0.309 | -0.951 | 0.000 |

*Coordinates are 10-20 system unit sphere positions (X: left-right, Y: front-back, Z: up-down)*

## Installation

```bash
# Clone the repository
git clone https://github.com/d0nmega/ssvep-device-control.git
cd ssvep-device-control

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- numpy, scipy, pandas, scikit-learn
- brainflow (EEG acquisition)
- PyQt6 (GUI - Phase 2)
- mne (optional, for offline analysis)

## Quick Start

### Test with Synthetic Data (No Hardware)
```bash
# Run with default 10 Hz simulated SSVEP
python run_cli.py

# Test different frequencies
python run_cli.py --target 8.57
python run_cli.py --target 12.0
python run_cli.py --target 15.0

# Adjust detection thresholds
python run_cli.py --confidence 0.55 --margin 0.10
```

### Run with Real Hardware
```bash
# Connect to Cyton on specific port
python run_cli.py --port COM3          # Windows
python run_cli.py --port /dev/ttyUSB0  # Linux

# Run for specific duration with logging
python run_cli.py --port COM3 --duration 60

# Disable logging
python run_cli.py --port COM3 --no-log
```

### CLI Options
```
--port, -p      Serial port for Cyton (e.g., COM3)
--duration, -d  Duration in seconds (default: 30)
--target, -t    Target frequency for synthetic mode (8.57, 10.0, 12.0, 15.0)
--confidence    Confidence threshold (default: 0.55)
--margin        Margin threshold (default: 0.15)
--no-log        Disable CSV logging
--list-ports    List available serial ports
```

### Run with Arduino Visual Feedback
The BCI can control the Arduino to provide real-time visual feedback - when you look at a white LED and the system detects it, the corresponding red LED lights up!

```bash
# Full system: Arduino + Cyton + Visual Feedback
python run_bci_with_feedback.py

# Specify ports manually
python run_bci_with_feedback.py --arduino COM3 --cyton COM4

# Test with synthetic EEG (Arduino only, no Cyton)
python run_bci_with_feedback.py --synthetic

# List available ports
python run_bci_with_feedback.py --list-ports
```

**Arduino Setup:**
1. Upload `arduino/ssvep_stimulator/ssvep_stimulator.ino` to your Arduino Mega
2. Connect LEDs as specified in Hardware Setup
3. The Python script will automatically start/stop stimulation

## Project Structure

```
ssvep-device-control/
├── run_cli.py              # CLI classifier (no Arduino control)
├── run_bci_with_feedback.py # Full BCI with Arduino feedback
├── run_gui.py              # GUI entry point (Phase 2)
├── requirements.txt        # Python dependencies
│
├── arduino/                # Arduino sketches
│   └── ssvep_stimulator/   # LED stimulator with serial control
├── README.md
│
└── ssvep_bci/              # Main package
    ├── main.py             # Application entry point
    ├── cli_test.py         # Real-time CLI with visualization
    │
    ├── utils/
    │   └── config.py       # Configuration parameters
    │
    ├── models/
    │   ├── eeg_buffer.py   # Ring buffer with sliding windows
    │   ├── preprocessor.py # CAR + Butterworth bandpass
    │   └── cca_decoder.py  # CCA classification + voting
    │
    ├── drivers/
    │   ├── brainflow_driver.py  # OpenBCI Cyton interface
    │   ├── arduino_controller.py # Arduino serial communication
    │   └── data_logger.py       # CSV logging utilities
    │
    └── gui/                # Phase 2 (coming soon)
        └── main_window.py
```

## Algorithm

### Signal Processing Pipeline
1. **Acquisition**: 250 Hz streaming from Cyton via BrainFlow
2. **Buffering**: 63-sample windows (252 ms) with 12-sample steps (48 ms, ~80% overlap)
3. **Preprocessing** (optional):
   - Common Average Reference (CAR)
   - 5th-order Butterworth bandpass (6-40 Hz)
4. **Classification**: CCA with synthetic reference signals
   - Fundamental + 1st harmonic for each target frequency
   - Correlation computed for all 4 frequencies per window
5. **Decision Logic**:
   - Confidence threshold: ρ_max ≥ 0.55
   - Margin threshold: (ρ_max - ρ_second) ≥ 0.15
   - Temporal voting: 2 consecutive agreements required

### Performance Metrics
- **Processing latency**: 2-4 ms per window (target: <20 ms)
- **Throughput**: ~70 windows/second
- **Accuracy**: 100% on synthetic data (with margin=0.10)

## Arduino Sketch

The Arduino runs independently with this behavior:
- **Button (D10)**: Toggle LED activity on/off
- **White LEDs (D2-D5)**: Non-blocking square wave flicker at target frequencies
- **Red LEDs (D6-D9)**: 1-second chaser pattern (visual feedback)

**Note**: The Arduino sketch is pre-loaded and NOT modified by this software. The Python BCI only reads EEG and classifies - it does not control the Arduino.

## Logging

Session data is saved to `logs/` directory:
- `ssvep_session_YYYYMMDD_HHMMSS_classification.csv`
  - Timestamps, correlations, predictions, latencies
- Analyze with included `SessionAnalyzer` class for accuracy/ITR metrics

## Configuration

Edit `ssvep_bci/utils/config.py` to customize:

```python
@dataclass
class SSVEPConfig:
    fs: int = 250                    # Sampling rate
    window_samples: int = 63         # ~252 ms window
    step_samples: int = 12           # ~48 ms step

    target_frequencies = (8.57, 10.0, 12.0, 15.0)
    n_harmonics: int = 2             # Fundamental + 1 harmonic

    confidence_threshold: float = 0.55
    margin_threshold: float = 0.15
    agreement_window: int = 2        # Windows that must agree

    bandpass_low: float = 6.0        # Hz
    bandpass_high: float = 40.0      # Hz
```

## Troubleshooting

### Cyton Connection Issues
```bash
# List available ports
python run_cli.py --list-ports

# Check if port is in use by another application
# On Windows: Device Manager > Ports
```

### Low Classification Accuracy
1. Check electrode impedance (<10 kΩ recommended)
2. Ensure user is fixating on one LED
3. Try lowering margin threshold: `--margin 0.10`
4. Increase window size in config for better frequency resolution

### High Latency
- Close other CPU-intensive applications
- Check USB connection (use USB 3.0 if available)

## Development Roadmap

- [x] **Phase 1**: Core pipeline (buffer, CCA, decision logic)
- [x] **Phase 1**: CLI test interface with real-time display
- [x] **Phase 1**: Synthetic data generator for testing
- [ ] **Phase 2**: PyQt6 GUI with live EEG plots
- [ ] **Phase 2**: Calibration mode with per-subject templates
- [ ] **Phase 2**: Device control output (keyboard/mouse simulation)

## References

- Lin, Z., et al. (2007). Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. *IEEE Trans. Biomed. Eng.*
- Chen, X., et al. (2015). Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface. *J. Neural Eng.*

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
