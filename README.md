# SSVEP Brain-Computer Interface

Real-time SSVEP-BCI using OpenBCI Cyton, Arduino LED stimulation, and TRCA classification.

## System Overview

**Hardware:**
- OpenBCI Cyton (8-channel EEG @ 250 Hz)
- Arduino Mega (4 white LEDs @ 8.57, 10, 12, 15 Hz + 4 red feedback LEDs)

**Algorithm:** Task-Related Component Analysis (TRCA) with zero-phase offline preprocessing

**Electrode Montage:** Pz, P3, P4, PO3, PO4, O1, Oz, O2 (occipital/parietal)

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, scikit-learn, brainflow, pylsl, pyxdf, pyserial

## Quick Start

### 1. Hardware Diagnostics

Test connections and signal quality:

```bash
python troubleshoot.py
python troubleshoot.py --cyton COM3 --arduino COM4
```

**Checks:** Arduino LED control, Cyton connection, signal quality, real-time visualization

### 2. Calibration & Training

Records EEG data with LSL markers, trains TRCA model:

```bash
python calibration.py --subject alice --trials 5 --baseline 30
```

**Workflow:**
1. Start LabRecorder to capture LSL streams
2. Script records 30s baseline + 20 calibration trials
3. Automatically trains TRCA model from XDF file
4. Saves model to `models/alice_trca_TIMESTAMP.pkl`

### 3. Real-Time Classification

Uses trained model for live inference:

```bash
python classify.py models/alice_trca_20250115_143022.pkl
```

**Features:**
- Real-time TRCA classification (250ms windows)
- Causal preprocessing (5-50 Hz + 60 Hz notch)
- Arduino LED feedback
- Emergency stop button (D10)

## File Structure

```
ssvep-device-control/
├── calibration.py              # Offline: Data collection + TRCA training
├── classify.py                 # Online: Real-time classification
├── troubleshoot.py             # Hardware diagnostics
├── requirements.txt            # Python dependencies
├── README.md
├── MINIMAL_STRUCTURE.md        # Detailed documentation
│
├── ssvep_bci/                  # Core modules
│   ├── config.py               # System configuration
│   ├── preprocessor.py         # Online/offline preprocessing + XDF I/O + LSL
│   ├── drivers.py              # BrainFlow (EEG) + Arduino (LED control)
│   ├── buffer.py               # Ring buffer for real-time windowing
│   └── trca.py                 # TRCA classifier
│
├── models/                     # Trained TRCA models (.pkl)
├── arduino/ssvep_stimulator/   # Arduino LED controller firmware
└── Archive/                    # Legacy code (CCA, GUI, old scripts)
```

## Algorithm Details

### Preprocessing

| Stage | Filters | Method | Use |
|-------|---------|--------|-----|
| **Offline** | 7-90 Hz + 60 Hz notch | Zero-phase (`filtfilt`) | Training |
| **Online** | 5-50 Hz + 60 Hz notch | Causal (stateful IIR) | Real-time |

### TRCA (Task-Related Component Analysis)

Maximizes inter-trial covariance to extract reproducible task-related components. Superior to CCA for SSVEP classification.

**Reference:** Nakanishi et al. (2018). IEEE Trans. Biomed. Eng., 65(1), 104-112.

## Hardware Setup

### Arduino Connections

**White LEDs (Stimulation):**
- D2: 15 Hz (far right)
- D3: 12 Hz (center-right)
- D4: 10 Hz (center-left)
- D5: 8.57 Hz (far left)

**Red LEDs (Feedback):**
- D6-D9: Correspond to white LEDs

**Button:** D10 (emergency stop)

### EEG Montage

8 channels (10-20 system):

| Ch | Electrode | Position |
|----|-----------|----------|
| 1 | Pz | Midline parietal |
| 2 | P3 | Left parietal |
| 3 | P4 | Right parietal |
| 4 | PO3 | Left parietal-occipital |
| 5 | PO4 | Right parietal-occipital |
| 6 | O1 | Left occipital |
| 7 | Oz | Midline occipital |
| 8 | O2 | Right occipital |

**Reference:** SRB2 to earlobe/mastoid

## Troubleshooting

### Hardware Diagnostics

Run diagnostics before troubleshooting:

```bash
python troubleshoot.py [--cyton PORT] [--arduino PORT]
```

### Common Issues

**No EEG signal:**
- Check SRB2 reference electrode connection (most common)
- Verify electrode impedances <10 kΩ
- Run `troubleshoot.py` to check signal quality in real-time

**Low classification accuracy:**
- Run 3-5 calibration sessions (incremental learning)
- Increase baseline duration (30-60 seconds)
- Ensure proper fixation during calibration
- Try longer analysis windows: `python classify.py model.pkl --window-ms 500`

**Arduino not detected:**
- Check COM port in Device Manager (Windows) or `ls /dev/tty*` (Linux)
- Verify Arduino sketch uploaded: `arduino/ssvep_stimulator.ino`
- Try manual port: `--arduino COM3`

**LabRecorder not seeing streams:**
- Ensure `calibration.py` is running before starting LabRecorder
- Restart LabRecorder and click "Update"
- Check firewall (LSL uses UDP multicast)

## License

MIT License

## References

- Nakanishi, M., et al. (2018). TRCA for SSVEP-BCI. IEEE Trans. Biomed. Eng.
- Python TRCA: https://github.com/mnakanishi/trca
