# SSVEP BCI Fixes and Improvements

## ✅ COMPLETED FIXES

### 1. Calibration Beeping Buffer Fix
**Problem:** Beeps were buffering and overlapping during visual tasks, causing confusion.

**Solution:**
- Changed `winsound.Beep()` to non-blocking threaded beeps
- Added `suppress_beeps()` function to disable beeps during trials
- Beeps now play without interfering with data collection

**Files Modified:** `run_calibration.py`

### 2. OpenBCI Cyton Auto-Detection
**Problem:** Had to manually specify COM port for Cyton (unlike Arduino which auto-detects).

**Solution:**
- Added `auto_detect_cyton()` static method to `BrainFlowDriver`
- Searches for FTDI devices (Cyton uses FT231X chip)
- Automatically detects and connects if no port specified

**Files Modified:** `ssvep_bci/drivers/brainflow_driver.py`

**Usage:** Just run scripts without `--cyton` parameter and it will auto-detect!

### 3. Arduino Emergency Stop Button (D10) Detection
**Problem:** Button on Arduino was unused in Python scripts.

**Solution:**
- Added `button_pressed` flag to `ArduinoController`
- Arduino sends "STOPPED" when button pressed
- Python detects unexpected "STOPPED" messages as button press
- Added `check_button_pressed()` method for BCI scripts to monitor

**Files Modified:** `ssvep_bci/drivers/arduino_controller.py`

**Status:** Detection implemented, needs integration into BCI scripts (see TODO below)

---

## 🔧 CRITICAL ISSUES TO FIX

### ISSUE #1: Low Accuracy / Always Predicting 8.57 Hz

**Your Report:** "run_bci_calibrated almost always thinks I am looking at 8.57 Hz and still have some very low accuracy"

**Likely Causes:**
1. **Template quality issues** - Not enough calibration data or poor data quality during calibration
2. **Index mapping mismatch** - Possible bug where LED indices are swapped
3. **Preprocessing not being applied** - Templates might be created from filtered data but BCI uses raw data (or vice versa)
4. **Channel 1 (Pz) disabled** - You said you're not using Pz, but code might still expect 8 channels
5. **Threshold too strict** - confidence/margin thresholds might be rejecting good predictions

**Debugging Steps:**
1. Run `run_realtime_monitor.py` with your calibration to see LIVE correlations for all frequencies
2. Look at one LED at a time and verify the correct frequency shows highest correlation
3. Check if correlations are generally low (<0.4) or if wrong frequency is truly highest
4. Verify Channel 1 (Pz) is properly handled (should be ignored if unplugged)

**Fixes Needed:**
- Add diagnostic mode to show raw correlation values during classification
- Verify template extraction window is correct
- Check if preprocessing is consistent between calibration and BCI
- Add option to exclude Channel 1 from analysis

---

### ISSUE #2: No Interactive Validation / Online Learning

**Your Request:** "Is there any form of learning or training it can do where I confirm whether or not it's right?"

**Current Status:** Only offline learning (calibration, then use). No online learning yet.

**Solution Needed:**
Create interactive validation mode where:
1. BCI makes prediction
2. You confirm if correct (keypress: Y/N)
3. If correct, add to template library (online learning)
4. If wrong, ignore or create negative example
5. Templates update in real-time for improved accuracy

**Benefits:**
- Active learning: system improves while you use it
- Can correct mistakes immediately
- No need for perfect calibration up-front
- Build confidence in the system

---

## 📋 TODO LIST

### High Priority (Core Functionality)

- [ ] **Debug 8.57 Hz bias in predictions**
  - Add verbose logging to template_cca.py showing all correlations
  - Verify LED index mapping is correct
  - Check if preprocessing matches between calibration and BCI
  - Test with single-frequency calibration to isolate issue

- [ ] **Handle Channel 1 (Pz) being disabled**
  - Add config option to exclude channels
  - Update preprocessor to skip disabled channels
  - Update CCA decoder to work with variable channel count

- [ ] **Implement emergency stop monitoring in BCI scripts**
  - Add button checking to main loop of:
    - `run_bci_with_feedback.py`
    - `run_bci_calibrated.py`
    - `run_realtime_monitor.py`
  - Gracefully exit when button pressed

- [ ] **Create interactive validation/learning mode**
  - New script: `run_bci_interactive.py`
  - Show prediction, ask for confirmation
  - Update templates based on feedback
  - Save updated templates automatically

### Medium Priority (Optimization)

- [ ] **Optimize window size for accuracy**
  - Test different window sizes (252ms, 400ms, 500ms)
  - Find optimal tradeoff between latency and accuracy
  - Add as command-line parameter to all BCI scripts

- [ ] **Improve template quality checks**
  - Add SNR measurement during calibration
  - Reject poor quality trials
  - Show quality metrics after calibration

- [ ] **Add confidence scoring**
  - Show classification confidence in real-time
  - Allow threshold adjustment based on performance
  - Warn when confidence is consistently low

### Low Priority (UX Improvements)

- [ ] **Better visual feedback during calibration**
  - Progress bar for each trial
  - Visual countdown timer
  - Real-time signal quality indicator

- [ ] **Session analytics**
  - Plot accuracy over time
  - Show confusion matrix
  - ITR (Information Transfer Rate) calculation

---

## 🎯 RECOMMENDATIONS

### How Many Calibration Sessions?

**Short Answer:** 3-5 sessions minimum, more is better.

**Reasoning:**
- **Session 1**: Get initial templates, might be 60-70% accurate
- **Session 2-3**: System learns your patterns, accuracy improves to 80-85%
- **Session 4-5**: Fine-tuning, can reach 90%+ accuracy
- **Additional sessions**: Robustness across different days, mental states

**Best Practice:**
- Do 1 session per day over 5 days (more representative)
- Each session: 5 trials × 4 frequencies = 20 trials (about 2-3 minutes)
- Use incremental learning (combine all sessions) - already implemented!

### Why Current BCI May Not Be Working

**Hypothesis:**
1. **Not enough calibration data** - If you only did 1 session, templates are weak
2. **Different preprocessing** - Calibration data might be processed differently than BCI data
3. **Channel mismatch** - You disabled Pz but system still expects 8 channels
4. **Wrong LED mapping** - Physical LED position might not match code expectations

### Immediate Actions to Take

1. **Run real-time monitor** to diagnose:
   ```bash
   python run_realtime_monitor.py --subject Donovan_Santine --cyton COM3 --arduino COM4
   ```
   - Look at 8.57 Hz LED - does correlation for 8.57 show highest?
   - Look at other LEDs - are correlations scrambled?

2. **Verify LED positions match code:**
   - Far left LED should be 8.57 Hz
   - Look at it, verify correlation for 8.57 Hz is highest

3. **Check number of calibration sessions:**
   ```bash
   python run_bci_calibrated.py --list-calibrations
   ```
   - Should show all your sessions
   - If only 1 session, do 2-3 more

4. **Test with larger window for better accuracy:**
   ```bash
   python run_realtime_monitor.py --subject Donovan_Santine --cyton COM3 --window-ms 500
   ```

### Online vs Offline Learning

**Current:** Offline only
- Collect calibration data first
- Train templates
- Use templates (no updating)

**Needed:** Online learning (hybrid)
- Start with calibration templates
- User confirms predictions during use
- Update templates based on correct predictions
- Continuous improvement

**Implementation Plan:**
1. Create interactive BCI mode
2. After each prediction, show result and ask "Correct? (Y/N)"
3. If Yes: Add data to templates, incrementally update
4. If No: Use for negative training or discard
5. Periodically save updated templates

---

## 📊 EXPECTED PERFORMANCE

### Target Metrics
- **Accuracy:** >90% with good calibration and 400-500ms windows
- **ITR:** 20-30 bits/min (competitive with research BCIs)
- **Latency:** <600ms total (with 500ms window + processing)

### Current Status
- **Accuracy:** Low (needs diagnosis)
- **ITR:** Not measurable until accuracy improves
- **Latency:** Good (~300ms with default 252ms window)

### Path to 90% Accuracy
1. Fix any channel/preprocessing bugs (see debugging steps)
2. Collect 3-5 good calibration sessions
3. Use 400-500ms windows (sacrifice latency for accuracy)
4. Ensure good signal quality (check with troubleshooter)
5. Minimize eye movements during trials
6. Implement online learning for continuous improvement

---

## 🐛 KNOWN BUGS TO FIX

1. **Possible index mapping bug** - 8.57 Hz always predicted suggests first LED in array is always highest
2. **Channel count mismatch** - Pz disabled but code expects 8 channels
3. **Emergency stop not integrated** - Detection works but BCI scripts don't check for it yet

---

## 📝 NEXT STEPS

1. **IMMEDIATE:** Debug why 8.57 Hz is always predicted
   - Add verbose logging to see all correlation scores
   - Test real-time monitor to see if correlations are correct
   - Verify LED physical layout matches code

2. **SHORT TERM:** Implement interactive validation mode
   - Create `run_bci_interactive.py`
   - Add user confirmation after predictions
   - Implement online template updating

3. **MEDIUM TERM:** Optimize for 90%+ accuracy
   - Test different window sizes
   - Collect more calibration sessions
   - Fine-tune thresholds based on your data

---

**Ready to proceed with debugging the accuracy issue?**
