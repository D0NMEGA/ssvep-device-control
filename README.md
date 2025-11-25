# SSVEP Brain-Computer Interface

A real-time Steady-State Visual Evoked Potential (SSVEP) Brain-Computer Interface system using OpenBCI Cyton and Arduino-based LED stimulation.

## Overview

This project implements a 4-class SSVEP BCI that:
- Acquires 8-channel EEG at 250 Hz from OpenBCI Cyton
- Classifies user attention to flickering LEDs using Canonical Correlation Analysis (CCA)
- Achieves <300ms total latency with 250ms analysis windows
- Targets ≥90% classification accuracy

### Target Frequencies (Arduino LED Flicker Rates)
| Physical Position | LED Pin | Frequency | Half-Period |
|-------------------|---------|-----------|-------------|
| Far Left | D5 | 8.57 Hz | 58,333 µs |
| Center-Left | D4 | 10.0 Hz | 50,000 µs |
| Center-Right | D3 | 12.0 Hz | 41,667 µs |
| Far Right | D2 | 15.0 Hz | 33,333 µs |

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

## Hardware Troubleshooting

Run comprehensive hardware diagnostics to verify EEG and Arduino connections:

```bash
# Full hardware test (auto-detects ports)
python troubleshoot_hardware.py

# Specify ports manually
python troubleshoot_hardware.py --cyton COM3 --arduino COM4

# Adjust signal analysis duration
python troubleshoot_hardware.py --cyton COM3 --duration 20
```

**What it checks:**
- Arduino connection and LED control (white + red LEDs)
- Cyton EEG acquisition and sampling rate
- **Signal quality analysis** - detects common issues:
  - Flat channels (broken/disconnected electrodes)
  - DC offset (reference electrode issues)
  - Excessive noise (poor contact, EMG, electrical interference)
  - Saturation (amplifier clipping)

**Common Issues:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| All channels bad/noisy | **Reference (SRB2) not connected** | Connect SRB2 to earlobe/mastoid |
| One channel flat | Broken electrode or poor contact | Re-seat electrode, check wire |
| High noise on all channels | Electrical interference | Move away from electronics, check grounding |
| DC offset | Poor reference contact | Clean reference electrode, improve contact |

## Real-time Monitoring & Optimization

The real-time monitor provides comprehensive live monitoring of your BCI system for validation and optimization:

```bash
# With calibration templates (recommended)
python run_realtime_monitor.py --subject YOUR_NAME --cyton COM3 --arduino COM4

# Without calibration (standard CCA)
python run_realtime_monitor.py --cyton COM3 --arduino COM4

# Larger window for higher accuracy (500ms instead of 252ms)
python run_realtime_monitor.py --subject YOUR_NAME --cyton COM3 --window-ms 500

# Quick signal quality test only
python run_realtime_monitor.py --cyton COM3 --test-only
```

**What it shows:**
- **Signal quality:** Raw vs filtered data comparison
- **Live correlations:** Real-time correlation scores for all 4 frequencies
- **Classification results:** Which LED the system thinks you're looking at
- **Performance metrics:** Accuracy, prediction distribution, average correlations
- **Optimization recommendations:** Personalized suggestions to improve accuracy

**Live Display Format:**
```
Time  │ Win#  │ 15.00Hz (Far ) │ 12.00Hz (Cent) │ 10.00Hz (Cent) │ 8.57Hz (Far ) │ Best   │ Margin │ Pred   │ Status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  3.2s│   145 │  0.342         │  0.289         │  0.712         │  0.301         │  10.00 │  0.223 │  10.00 │ Center-Left
```

**Window Size Optimization:**
- Default: 252ms (63 samples @ 250Hz) - Low latency
- For >90% accuracy: Try 400-500ms windows (`--window-ms 500`)
- Tradeoff: Larger windows = better frequency resolution but higher latency

**Using Calibration Templates:**
All BCI scripts now support `--subject` parameter to automatically load your calibration data:
```bash
# BCI with Arduino feedback + calibration
python run_bci_with_feedback.py --subject YOUR_NAME --cyton COM3 --arduino COM4

# Real-time monitor + calibration
python run_realtime_monitor.py --subject YOUR_NAME --cyton COM3

# Standard calibrated BCI
python run_bci_calibrated.py --subject YOUR_NAME --cyton COM3
```

## Calibration (Personalized Templates)

Run calibration to create personalized templates for improved accuracy:

```bash
# Run calibration (5 trials per frequency, 4 seconds each)
python run_calibration.py --subject YOUR_NAME --cyton COM3

# Shorter calibration for testing
python run_calibration.py --subject YOUR_NAME --cyton COM3 --trials 3 --duration 3
```

Data is saved to `calibration/YOUR_NAME/session_YYYYMMDD_HHMMSS.npz`

### Run BCI with Calibrated Templates
```bash
# Uses all your calibration sessions (incremental learning)
python run_bci_calibrated.py --subject YOUR_NAME --cyton COM3

# Use only latest session
python run_bci_calibrated.py --subject YOUR_NAME --cyton COM3 --latest-only

# List available calibrations
python run_bci_calibrated.py --list-calibrations
```

**Incremental Learning**: Each calibration session adds to your template library. More sessions = more robust templates!

## Project Structure

```
ssvep-device-control/
├── run_cli.py               # CLI classifier (no Arduino)
├── run_bci_with_feedback.py # BCI with Arduino feedback (supports --subject)
├── run_bci_calibrated.py    # BCI with personalized templates
├── run_calibration.py       # Calibration data collection
├── run_realtime_monitor.py  # Real-time monitoring & optimization
├── run_gui.py               # PyQt6 GUI
├── troubleshoot_hardware.py # Hardware diagnostics
├── requirements.txt
│
├── calibration/             # Calibration data (per subject)
│   └── YOUR_NAME/
│       ├── session_*.npz    # EEG epochs
│       └── session_*.json   # Metadata
│
├── arduino/
│   └── ssvep_stimulator/    # LED stimulator sketch
│
└── ssvep_bci/
    ├── main.py
    ├── cli_test.py
    │
    ├── utils/
    │   └── config.py
    │
    ├── models/
    │   ├── eeg_buffer.py    # Ring buffer
    │   ├── preprocessor.py  # CAR + bandpass + notch + smoothing
    │   ├── cca_decoder.py   # Standard CCA
    │   ├── calibration.py   # Calibration data collection
    │   └── template_cca.py  # Template-based CCA
    │
    ├── drivers/
    │   ├── brainflow_driver.py
    │   ├── arduino_controller.py
    │   └── data_logger.py
    │
    └── gui/
        ├── main_window.py
        └── widgets/
            ├── signal_plot.py
            ├── classification_panel.py
            └── control_panel.py
```

## Algorithm

### Signal Processing Pipeline
1. **Acquisition**: 250 Hz streaming from Cyton via BrainFlow
2. **Buffering**: 63-sample windows (252 ms) with 12-sample steps (48 ms, ~80% overlap)
3. **Preprocessing** (enabled by default):
   - Common Average Reference (CAR)
   - 5th-order Butterworth bandpass (5-50 Hz)
   - Notch filter (60 Hz for US powerline noise)
   - Moving average smoothing (5 samples = 20ms)
4. **Classification**: CCA with synthetic reference signals (or template-based if calibrated)
   - Fundamental + 1st harmonic for each target frequency
   - Correlation computed for all 4 frequencies per window
   - Template-based CCA: 70% template matching + 30% standard CCA (when using `--subject`)
5. **Decision Logic**:
   - Confidence threshold: ρ_max ≥ 0.55
   - Margin threshold: (ρ_max - ρ_second) ≥ 0.15
   - Temporal voting: 2 consecutive agreements required

### Performance Metrics
- **Processing latency**: 2-4 ms per window (target: <20 ms)
- **Throughput**: ~70 windows/second
- **Accuracy**: 100% on synthetic data (with margin=0.10)

## Arduino Sketch

The Arduino communicates with Python via serial (115200 baud):
- **Button (D10)**: Manual toggle LED activity on/off
- **White LEDs (D2-D5)**: Non-blocking square wave flicker at target frequencies
- **Red LEDs (D6-D9)**: Feedback LEDs controlled by Python

**Serial Commands** (Python -> Arduino):
- `START` - Start LED flickering
- `STOP` - Stop all LEDs
- `FEEDBACK:N` - Light red LED N (0-3) for visual feedback
- `CLEAR` - Turn off all red LEDs

Upload `arduino/ssvep_stimulator/ssvep_stimulator.ino` to your Arduino Mega before use.

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

    target_frequencies = (15.0, 12.0, 10.0, 8.57)
    # Physical layout (left to right): 8.57, 10, 12, 15 Hz
    n_harmonics: int = 2             # Fundamental + 1 harmonic

    confidence_threshold: float = 0.55
    margin_threshold: float = 0.15
    agreement_window: int = 2        # Windows that must agree

    # Default filters (applied to ALL BCI modes)
    bandpass_low: float = 5.0        # Hz
    bandpass_high: float = 50.0      # Hz
    notch_freq: float = 60.0         # Hz (US powerline noise)
    smoothing_enabled: bool = True   # Moving average smoothing
    smoothing_window: int = 5        # 5 samples = 20ms
```

## Troubleshooting

### Hardware Issues
For hardware diagnostics (EEG signal quality, electrode contact, etc.), use:
```bash
python troubleshoot_hardware.py --cyton COM3 --arduino COM4
```
See [Hardware Troubleshooting](#hardware-troubleshooting) section for details.

### Cyton Connection Issues
```bash
# List available ports
python run_cli.py --list-ports

# Check if port is in use by another application
# On Windows: Device Manager > Ports
```

### Low Classification Accuracy
1. Run hardware troubleshooter to verify signal quality
2. Check electrode impedance (<10 kΩ recommended)
3. Ensure user is fixating on one LED (avoid eye movements)
4. Try lowering margin threshold: `--margin 0.10`
5. Run calibration to create personalized templates
6. Increase window size in config for better frequency resolution

### High Latency
- Close other CPU-intensive applications
- Check USB connection (use USB 3.0 if available)
- Reduce GUI update rate if using PyQt interface

## Development Roadmap

- [x] **Phase 1**: Core pipeline (buffer, CCA, decision logic)
- [x] **Phase 1**: CLI test interface with real-time display
- [x] **Phase 1**: Synthetic data generator for testing
- [x] **Phase 1**: Arduino serial control with visual feedback
- [x] **Phase 2**: PyQt6 GUI with live EEG plots
- [x] **Phase 2**: Calibration mode with per-subject templates
- [x] **Phase 2**: Incremental learning (combine multiple sessions)
- [ ] **Phase 3**: Device control output (keyboard/mouse simulation)
- [ ] **Phase 3**: Filter bank CCA for improved accuracy

## References

- Lin, Z., et al. (2007). Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. *IEEE Trans. Biomed. Eng.*
- Chen, X., et al. (2015). Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface. *J. Neural Eng.*

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
