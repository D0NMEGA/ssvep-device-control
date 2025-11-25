#!/usr/bin/env python3
"""
TRCA Model Training from XDF Calibration Data

Processes XDF calibration files and trains TRCA model using:
- Zero-phase filtering (7-90 Hz bandpass + 60 Hz notch)
- Trial extraction from XDF markers
- TRCA spatial filtering and template creation
- Model evaluation and saving

Usage:
    python run_train_trca.py calibration/subject01_session1.xdf
    python run_train_trca.py calibration/*.xdf  # Train on multiple sessions
"""

import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Add ssvep_bci to path
sys.path.insert(0, str(Path(__file__).parent / "ssvep_bci"))

from utils.config import SSVEPConfig
from utils.io_xdf import load_xdf_calibration, extract_trials_from_xdf, extract_baseline_from_xdf
from models.offline_preprocessor import OfflinePreprocessor
from models.trca import TRCA


def process_xdf_file(xdf_path: str, config: SSVEPConfig) -> tuple:
    """Process a single XDF file.

    Args:
        xdf_path: Path to XDF file
        config: SSVEPConfig instance

    Returns:
        Tuple of (trial_data, labels, metadata)
        - trial_data: (n_trials, n_channels, n_samples)
        - labels: (n_trials,) frequency indices
        - metadata: dict with trial info
    """
    print(f"\nProcessing {xdf_path}...")

    # Load XDF
    eeg_data, markers, metadata = load_xdf_calibration(xdf_path, config)

    print(f"  Loaded {metadata['n_samples']} samples ({metadata['duration_sec']:.1f} sec)")
    print(f"  Found {len(markers)} markers")

    # Check for baseline
    baseline_data, baseline_meta = extract_baseline_from_xdf(xdf_path, config)
    if baseline_data is not None:
        print(f"  Baseline: {baseline_meta['duration_sec']:.1f} sec, {baseline_meta['n_samples']} samples")

    # Extract trials
    trial_data, labels, trial_meta = extract_trials_from_xdf(
        xdf_path,
        config,
        trial_duration_sec=4.0,
        baseline_duration_sec=0.5
    )

    print(f"  Extracted {trial_meta['n_trials']} trials")
    print(f"  Labels: {labels}")

    # Validate labels
    valid_mask = labels >= 0
    if not np.all(valid_mask):
        print(f"  Warning: {np.sum(~valid_mask)} trials with invalid labels, removing...")
        trial_data = trial_data[valid_mask]
        labels = labels[valid_mask]

    return trial_data, labels, trial_meta


def train_trca_model(
    xdf_files: list,
    output_path: str = None,
    n_components: int = 1,
    test_split: float = 0.2
):
    """Train TRCA model from XDF calibration files.

    Args:
        xdf_files: List of XDF file paths
        output_path: Path to save trained model
        n_components: Number of TRCA components
        test_split: Fraction of data to use for testing
    """
    print("=" * 60)
    print("TRCA Model Training from XDF Data")
    print("=" * 60)

    config = SSVEPConfig()
    preprocessor = OfflinePreprocessor(config)

    # Load and process all XDF files
    all_trials = []
    all_labels = []

    for xdf_path in xdf_files:
        trial_data, labels, metadata = process_xdf_file(xdf_path, config)
        all_trials.append(trial_data)
        all_labels.append(labels)

    # Concatenate all trials
    X = np.concatenate(all_trials, axis=0)  # Shape: (n_trials, n_channels, n_samples)
    y = np.concatenate(all_labels, axis=0)  # Shape: (n_trials,)

    print(f"\n{'='*60}")
    print(f"Total dataset:")
    print(f"  {X.shape[0]} trials")
    print(f"  {X.shape[1]} channels")
    print(f"  {X.shape[2]} samples per trial")
    print(f"{'='*60}")

    # Show class distribution
    print("\nClass distribution:")
    for class_idx in range(len(config.target_frequencies)):
        n_trials = np.sum(y == class_idx)
        freq = config.target_frequencies[class_idx]
        print(f"  Class {class_idx} ({freq:.2f} Hz): {n_trials} trials")

    # Apply offline preprocessing
    print("\nApplying offline preprocessing...")
    print("  - Common Average Reference (CAR)")
    print("  - Bandpass filter: 7-90 Hz (zero-phase)")
    print("  - Notch filter: 60 Hz (zero-phase)")

    X_preprocessed = preprocessor.process(X)

    print(f"  Preprocessed data range: [{X_preprocessed.min():.2f}, {X_preprocessed.max():.2f}]")

    # Split train/test
    n_trials = X_preprocessed.shape[0]
    n_test = int(n_trials * test_split)
    n_train = n_trials - n_test

    # Stratified split (ensure each class is represented in test)
    train_indices = []
    test_indices = []

    for class_idx in range(len(config.target_frequencies)):
        class_mask = (y == class_idx)
        class_indices = np.where(class_mask)[0]

        n_class = len(class_indices)
        n_class_test = max(1, int(n_class * test_split))

        # Shuffle
        np.random.shuffle(class_indices)

        test_indices.extend(class_indices[:n_class_test])
        train_indices.extend(class_indices[n_class_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    X_train = X_preprocessed[train_indices]
    y_train = y[train_indices]
    X_test = X_preprocessed[test_indices]
    y_test = y[test_indices]

    print(f"\nTrain/test split:")
    print(f"  Training: {len(train_indices)} trials ({(1-test_split)*100:.0f}%)")
    print(f"  Testing: {len(test_indices)} trials ({test_split*100:.0f}%)")

    # Train TRCA
    print(f"\nTraining TRCA (n_components={n_components})...")
    trca = TRCA(config, n_components=n_components)
    trca.fit(X_train, y_train)

    print(f"  Trained {len(trca.spatial_filters)} spatial filters")
    print(f"  Template shapes: {[t.shape for t in trca.templates]}")

    # Evaluate on training data
    print("\nEvaluating on training data...")
    y_pred_train, corr_train = trca.predict_with_correlation(X_train)
    accuracy_train = np.mean(y_pred_train == y_train)

    print(f"  Training accuracy: {accuracy_train*100:.1f}%")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    y_pred_test, corr_test = trca.predict_with_correlation(X_test)
    accuracy_test = np.mean(y_pred_test == y_test)

    print(f"  Test accuracy: {accuracy_test*100:.1f}%")

    # Confusion matrix
    print("\nConfusion matrix (test data):")
    n_classes = len(config.target_frequencies)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_test, y_pred_test):
        conf_matrix[true_label, pred_label] += 1

    # Print header
    print("       Predicted:")
    freq_labels = [f"{freq:5.2f}" for freq in config.target_frequencies]
    print("     ", " ".join(freq_labels))

    # Print rows
    for i, freq in enumerate(config.target_frequencies):
        row_str = " ".join([f"{conf_matrix[i,j]:5d}" for j in range(n_classes)])
        print(f"{freq:5.2f} ", row_str)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, freq in enumerate(config.target_frequencies):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_test[class_mask] == i)
            print(f"  {freq:.2f} Hz: {class_acc*100:.1f}%")

    # Save model
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"models/trca_model_{timestamp}.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'trca': trca,
        'config': config,
        'preprocessor': preprocessor,
        'training_accuracy': accuracy_train,
        'test_accuracy': accuracy_test,
        'n_training_trials': len(train_indices),
        'n_test_trials': len(test_indices),
        'timestamp': datetime.now().isoformat(),
        'xdf_files': [str(f) for f in xdf_files],
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {output_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal test accuracy: {accuracy_test*100:.1f}%")
    print(f"Model file: {output_path}")

    return model_data


def main():
    parser = argparse.ArgumentParser(
        description="Train TRCA model from XDF calibration data"
    )
    parser.add_argument(
        "xdf_files",
        nargs="+",
        help="XDF calibration files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for trained model (default: auto-generated)"
    )
    parser.add_argument(
        "--components",
        type=int,
        default=1,
        help="Number of TRCA components (default: 1)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )

    args = parser.parse_args()

    # Validate XDF files exist
    xdf_files = []
    for pattern in args.xdf_files:
        matches = list(Path(".").glob(pattern))
        if not matches:
            print(f"Warning: No files match pattern '{pattern}'")
        xdf_files.extend(matches)

    if not xdf_files:
        print("Error: No XDF files found")
        return 1

    print(f"Found {len(xdf_files)} XDF file(s):")
    for f in xdf_files:
        print(f"  - {f}")

    # Train model
    try:
        train_trca_model(
            xdf_files,
            output_path=args.output,
            n_components=args.components,
            test_split=args.test_split
        )
        return 0
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
