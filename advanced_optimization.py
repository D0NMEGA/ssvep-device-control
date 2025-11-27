#!/usr/bin/env python3
"""
Advanced Accuracy Optimization

Tests more aggressive strategies:
1. Exclude bad sessions entirely
2. Per-session models vs batch learning
3. No detrending (test if it's removing signal)
4. Best sessions only
5. Ensemble of per-session models
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

from ssvep_bci.config import SSVEPConfig
from ssvep_bci.preprocessor import extract_trials_from_xdf, OfflinePreprocessor, load_xdf
from ssvep_bci.trca import TRCA


def extract_trials_no_detrend(xdf_path: str, config):
    """Extract trials WITHOUT detrending to test if it's removing signal."""
    from ssvep_bci.preprocessor import load_xdf

    eeg_data, markers, metadata = load_xdf(xdf_path, config)
    fs = metadata['sampling_rate']
    timestamps = metadata['eeg_timestamps']

    stim_onsets = []
    trial_labels = []

    for marker in markers:
        mdata = marker['data']
        label = mdata.get('label', '')

        if label.startswith('STIM_ON'):
            stim_onsets.append(marker['timestamp'])
            target_freq = mdata.get('payload', {}).get('freq', -1)

            freq_idx = -1
            for i, freq in enumerate(config.target_frequencies):
                if abs(freq - target_freq) < 0.01:
                    freq_idx = i
                    break
            trial_labels.append(freq_idx)

    delay_sec = 0.13
    baseline_sec = 0.5
    trial_duration_sec = 4.0

    n_delay = int(delay_sec * fs)
    n_baseline = int(baseline_sec * fs)
    n_trial = int(trial_duration_sec * fs)
    n_total = n_baseline + n_trial

    trials = []
    valid_labels = []

    for ts, label in zip(stim_onsets, trial_labels):
        idx = np.argmin(np.abs(timestamps - ts))
        start_idx = idx + n_delay - n_baseline
        end_idx = start_idx + n_total

        if start_idx >= 0 and end_idx <= eeg_data.shape[1] and label >= 0:
            epoch = eeg_data[:, start_idx:end_idx]
            # NO DETRENDING - keep raw signal
            trials.append(epoch)
            valid_labels.append(label)

    return np.array(trials), np.array(valid_labels)


def evaluate_loocv_simple(X, y, config):
    """Simple LOOCV without filtering."""
    n_total = len(X)
    correct = 0

    for i in range(n_total):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[i] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[i:i+1], y[i]

        trca = TRCA(config, n_components=1)
        trca.fit(X_train, y_train)

        y_pred, _ = trca.predict_with_correlation(X_test)

        if y_pred == y_test:
            correct += 1

    return correct / n_total, correct, n_total


def compute_session_quality(X, y, config):
    """Compute average SNR for a session."""
    oz_idx = 6
    baseline_samples = 125
    snr_values = []

    for trial_idx in range(len(X)):
        trial = X[trial_idx]
        true_freq = config.target_frequencies[y[trial_idx]]

        oz_data = trial[oz_idx, baseline_samples:]
        freqs = np.fft.rfftfreq(len(oz_data), 1/config.fs)
        fft_mag = np.abs(np.fft.rfft(oz_data))

        freq_idx = np.argmin(np.abs(freqs - true_freq))
        target_power = fft_mag[freq_idx]

        noise_indices = [freq_idx-2, freq_idx-1, freq_idx+1, freq_idx+2]
        valid_noise_idx = [i for i in noise_indices if 0 <= i < len(fft_mag)]
        background = np.mean([fft_mag[i] for i in valid_noise_idx])

        snr = target_power / background if background > 0 else 0
        snr_values.append(snr)

    return np.mean(snr_values), np.std(snr_values)


def main():
    subject = "donny"
    config = SSVEPConfig()
    config.num_fbs = 2

    calib_dir = Path("calibration")
    subject_xdfs = sorted(calib_dir.glob(f"{subject}_*.xdf"))

    print("="*70)
    print("ADVANCED ACCURACY OPTIMIZATION")
    print("="*70)

    # ===================================================================
    # Strategy 1: Analyze session quality and identify bad sessions
    # ===================================================================
    print("\n[1] SESSION QUALITY ANALYSIS")
    print("="*70)

    session_quality = []
    for xdf_file in subject_xdfs:
        X, y = extract_trials_from_xdf(str(xdf_file), config,
                                       trial_duration_sec=4.0, baseline_sec=0.5)

        mean_snr, std_snr = compute_session_quality(X, y, config)

        preprocessor = OfflinePreprocessor(config)
        X_preprocessed = preprocessor.process(X)

        loocv_acc, loocv_correct, loocv_total = evaluate_loocv_simple(X_preprocessed, y, config)

        session_quality.append({
            'file': xdf_file.name,
            'n_trials': len(X),
            'mean_snr': mean_snr,
            'std_snr': std_snr,
            'loocv_acc': loocv_acc
        })

        print(f"  {xdf_file.name}")
        print(f"    Trials: {len(X)}, SNR: {mean_snr:.2f}±{std_snr:.2f}, LOOCV: {loocv_acc*100:.1f}%")

    # Sort by quality (SNR)
    session_quality.sort(key=lambda x: x['mean_snr'], reverse=True)

    print("\n  Sessions ranked by SNR:")
    for i, sess in enumerate(session_quality):
        print(f"    {i+1}. {sess['file']}: SNR={sess['mean_snr']:.2f}, LOOCV={sess['loocv_acc']*100:.1f}%")

    # ===================================================================
    # Strategy 2: Test without detrending
    # ===================================================================
    print("\n[2] TEST WITHOUT DETRENDING")
    print("="*70)

    all_trials_no_detrend = []
    all_labels_no_detrend = []

    for xdf_file in subject_xdfs:
        X, y = extract_trials_no_detrend(str(xdf_file), config)
        all_trials_no_detrend.append(X)
        all_labels_no_detrend.append(y)

    X_combined_no_detrend = np.concatenate(all_trials_no_detrend, axis=0)
    y_combined_no_detrend = np.concatenate(all_labels_no_detrend, axis=0)

    preprocessor_no_detrend = OfflinePreprocessor(config)
    X_preprocessed_no_detrend = preprocessor_no_detrend.process(X_combined_no_detrend)

    print(f"  Total trials (no detrend): {len(X_preprocessed_no_detrend)}")
    loocv_acc_no_detrend, _, _ = evaluate_loocv_simple(
        X_preprocessed_no_detrend, y_combined_no_detrend, config
    )
    print(f"  LOOCV accuracy: {loocv_acc_no_detrend*100:.1f}%")

    # ===================================================================
    # Strategy 3: Use only best sessions (top 50%)
    # ===================================================================
    print("\n[3] BEST SESSIONS ONLY (Top 50% by SNR)")
    print("="*70)

    n_best = max(3, len(session_quality) // 2)
    best_sessions = [sess['file'] for sess in session_quality[:n_best]]

    print(f"  Using {n_best} best sessions: {best_sessions}")

    all_trials_best = []
    all_labels_best = []

    for sess_info in session_quality[:n_best]:
        xdf_file = calib_dir / sess_info['file']
        X, y = extract_trials_from_xdf(str(xdf_file), config,
                                       trial_duration_sec=4.0, baseline_sec=0.5)

        preprocessor = OfflinePreprocessor(config)
        X_preprocessed = preprocessor.process(X)

        all_trials_best.append(X_preprocessed)
        all_labels_best.append(y)

    X_combined_best = np.concatenate(all_trials_best, axis=0)
    y_combined_best = np.concatenate(all_labels_best, axis=0)

    print(f"  Total trials from best sessions: {len(X_combined_best)}")
    loocv_acc_best, _, _ = evaluate_loocv_simple(X_combined_best, y_combined_best, config)
    print(f"  LOOCV accuracy: {loocv_acc_best*100:.1f}%")

    # ===================================================================
    # Strategy 4: Exclude worst session (Session 3 based on FFT analysis)
    # ===================================================================
    print("\n[4] EXCLUDE WORST SESSION")
    print("="*70)

    worst_session = session_quality[-1]['file']
    print(f"  Excluding worst session: {worst_session}")

    all_trials_excl = []
    all_labels_excl = []

    for sess_info in session_quality[:-1]:  # Exclude last (worst)
        xdf_file = calib_dir / sess_info['file']
        X, y = extract_trials_from_xdf(str(xdf_file), config,
                                       trial_duration_sec=4.0, baseline_sec=0.5)

        preprocessor = OfflinePreprocessor(config)
        X_preprocessed = preprocessor.process(X)

        all_trials_excl.append(X_preprocessed)
        all_labels_excl.append(y)

    X_combined_excl = np.concatenate(all_trials_excl, axis=0)
    y_combined_excl = np.concatenate(all_labels_excl, axis=0)

    print(f"  Total trials (excluding worst): {len(X_combined_excl)}")
    loocv_acc_excl, _, _ = evaluate_loocv_simple(X_combined_excl, y_combined_excl, config)
    print(f"  LOOCV accuracy: {loocv_acc_excl*100:.1f}%")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "="*70)
    print("SUMMARY OF ALL STRATEGIES")
    print("="*70)

    strategies = [
        ("Baseline (all sessions, with detrending)", 28.1),  # From previous run
        ("No detrending", loocv_acc_no_detrend * 100),
        (f"Best {n_best} sessions only", loocv_acc_best * 100),
        ("Exclude worst session", loocv_acc_excl * 100),
        ("Quality filtering SNR >= 1.5", 36.2),  # From previous run
    ]

    strategies.sort(key=lambda x: x[1], reverse=True)

    for i, (strategy, acc) in enumerate(strategies):
        print(f"  {i+1}. {strategy}: {acc:.1f}%")

    # ===================================================================
    # Save best model
    # ===================================================================
    best_strategy = strategies[0]
    print(f"\n{'='*70}")
    print(f"BEST STRATEGY: {best_strategy[0]}")
    print(f"  Accuracy: {best_strategy[1]:.1f}%")
    print(f"{'='*70}")

    # Train final model with best strategy
    if "best" in best_strategy[0].lower():
        X_final = X_combined_best
        y_final = y_combined_best
        description = f"best_{n_best}_sessions"
    elif "exclude" in best_strategy[0].lower():
        X_final = X_combined_excl
        y_final = y_combined_excl
        description = "exclude_worst"
    elif "no detrend" in best_strategy[0].lower():
        X_final = X_preprocessed_no_detrend
        y_final = y_combined_no_detrend
        description = "no_detrending"
    else:
        # Default to best sessions
        X_final = X_combined_best
        y_final = y_combined_best
        description = "default_best"

    print(f"\nTraining final model with {len(X_final)} trials...")
    trca_final = TRCA(config, n_components=1)
    trca_final.fit(X_final, y_final)

    # Test accuracy
    y_pred, _ = trca_final.predict_with_correlation(X_final)
    train_acc = np.mean(y_pred == y_final)

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{subject}_trca_advanced_{description}_{timestamp}.pkl"

    model_data = {
        'trca': trca_final,
        'config': config,
        'train_accuracy': train_acc,
        'loocv_accuracy': best_strategy[1] / 100,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'n_sessions': len(all_trials_best) if "best" in description else len(all_trials_excl),
        'n_total_trials': len(X_final),
        'strategy': best_strategy[0]
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved: {model_path}")
    print(f"  Training accuracy: {train_acc*100:.1f}%")
    print(f"  LOOCV accuracy: {best_strategy[1]:.1f}%")


if __name__ == "__main__":
    main()
