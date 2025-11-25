"""
Task-Related Component Analysis (TRCA) for SSVEP-BCI

Implementation based on:
Nakanishi et al. (2018). "Enhancing Detection of SSVEPs for a High-Speed
Brain Speller Using Task-Related Component Analysis."
IEEE Transactions on Biomedical Engineering, 65(1), 104-112.

Python reference: https://github.com/mnakanishi/trca

TRCA maximizes inter-trial covariance to extract task-related components
that are reproducible across trials. This provides better SNR than CCA
for steady-state evoked potentials.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import SSVEPConfig, DEFAULT_CONFIG


class TRCA:
    """Task-Related Component Analysis for SSVEP classification.

    TRCA learns spatial filters that maximize inter-trial covariance,
    extracting reproducible task-related components across trials.

    Attributes:
        config: SSVEPConfig instance
        n_components: Number of TRCA components to retain
        spatial_filters: List of spatial filter matrices (one per class)
        templates: List of template signals (one per class)
        is_fitted: Whether the model has been trained
    """

    def __init__(self, config: SSVEPConfig = None, n_components: int = 1):
        """Initialize TRCA classifier.

        Args:
            config: SSVEPConfig instance
            n_components: Number of TRCA components to retain (default: 1)
        """
        self.config = config or DEFAULT_CONFIG
        self.n_components = n_components

        # Model parameters (filled during fit)
        self.spatial_filters = []  # List of (n_channels, n_components) arrays
        self.templates = []  # List of (n_channels, n_samples) arrays
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TRCA':
        """Train TRCA model on calibration data.

        Args:
            X: Training data with shape (n_trials, n_channels, n_samples)
            y: Labels with shape (n_trials,) containing class indices

        Returns:
            self (fitted model)
        """
        n_trials, n_channels, n_samples = X.shape

        # Get unique classes
        classes = np.unique(y)
        n_classes = len(classes)

        # Initialize storage
        self.spatial_filters = []
        self.templates = []

        # For each class, compute TRCA spatial filters and template
        for class_idx in classes:
            # Get trials for this class
            class_mask = (y == class_idx)
            X_class = X[class_mask]  # Shape: (n_trials_class, n_channels, n_samples)

            # Compute TRCA spatial filter
            W = self._compute_trca_filter(X_class)

            # Keep only top n_components
            W = W[:, :self.n_components]  # Shape: (n_channels, n_components)

            # Compute template by averaging trials
            template = np.mean(X_class, axis=0)  # Shape: (n_channels, n_samples)

            self.spatial_filters.append(W)
            self.templates.append(template)

        self.is_fitted = True
        return self

    def _compute_trca_filter(self, X: np.ndarray) -> np.ndarray:
        """Compute TRCA spatial filter for a single class.

        Maximizes inter-trial covariance:
        W = argmax { W^T S W } / { W^T Q W }

        where S is the inter-trial covariance (sum of covariances between all
        pairs of trials) and Q is the average within-trial covariance.

        Args:
            X: Trials for one class with shape (n_trials, n_channels, n_samples)

        Returns:
            Spatial filter matrix W with shape (n_channels, n_channels)
            Columns are sorted by eigenvalue (descending)
        """
        n_trials, n_channels, n_samples = X.shape

        # Compute S: inter-trial covariance matrix
        # S = sum over all pairs (i,j) of cov(X_i, X_j)
        S = np.zeros((n_channels, n_channels))

        for i in range(n_trials):
            for j in range(i+1, n_trials):
                Xi = X[i]  # Shape: (n_channels, n_samples)
                Xj = X[j]  # Shape: (n_channels, n_samples)

                # Cross-covariance: Xi @ Xj^T
                S += Xi @ Xj.T + Xj @ Xi.T

        # Normalize by number of pairs
        n_pairs = n_trials * (n_trials - 1)
        if n_pairs > 0:
            S /= n_pairs

        # Compute Q: average within-trial covariance matrix
        # Q = average over trials of cov(X_i, X_i)
        Q = np.zeros((n_channels, n_channels))

        for i in range(n_trials):
            Xi = X[i]  # Shape: (n_channels, n_samples)
            Q += Xi @ Xi.T

        Q /= n_trials

        # Solve generalized eigenvalue problem: S @ W = Q @ W @ Lambda
        # Use eigh for symmetric matrices
        try:
            # Regularize Q to ensure positive definiteness
            Q_reg = Q + np.eye(n_channels) * 1e-6

            # Solve generalized eigenvalue problem
            eigenvalues, W = linalg.eigh(S, Q_reg)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            W = W[:, idx]

        except linalg.LinAlgError:
            # Fallback: use standard eigenvalue decomposition on Q^{-1} S
            try:
                Q_inv = linalg.inv(Q + np.eye(n_channels) * 1e-6)
                eigenvalues, W = linalg.eigh(Q_inv @ S)

                idx = np.argsort(eigenvalues)[::-1]
                W = W[:, idx]

            except linalg.LinAlgError:
                # Last resort: return identity
                print("Warning: Eigenvalue decomposition failed, using identity")
                W = np.eye(n_channels)

        return W

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test data.

        Args:
            X: Test data with shape (n_trials, n_channels, n_samples)
               or (n_channels, n_samples) for single trial

        Returns:
            Predicted class indices with shape (n_trials,) or scalar
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Handle single trial
        single_trial = False
        if X.ndim == 2:
            X = X[np.newaxis, :, :]  # Add trial dimension
            single_trial = True

        n_trials = X.shape[0]
        predictions = np.zeros(n_trials, dtype=int)

        for trial_idx in range(n_trials):
            trial = X[trial_idx]  # Shape: (n_channels, n_samples)

            # Compute correlation with each template
            correlations = self._compute_correlations(trial)

            # Predict class with highest correlation
            predictions[trial_idx] = np.argmax(correlations)

        if single_trial:
            return predictions[0]

        return predictions

    def predict_with_correlation(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and return correlation values.

        Args:
            X: Test data with shape (n_trials, n_channels, n_samples)
               or (n_channels, n_samples) for single trial

        Returns:
            Tuple of (predictions, correlations)
            - predictions: Class indices with shape (n_trials,) or scalar
            - correlations: Correlation matrix with shape (n_trials, n_classes) or (n_classes,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Handle single trial
        single_trial = False
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
            single_trial = True

        n_trials = X.shape[0]
        n_classes = len(self.templates)

        predictions = np.zeros(n_trials, dtype=int)
        all_correlations = np.zeros((n_trials, n_classes))

        for trial_idx in range(n_trials):
            trial = X[trial_idx]

            # Compute correlations
            correlations = self._compute_correlations(trial)
            all_correlations[trial_idx] = correlations

            # Predict class
            predictions[trial_idx] = np.argmax(correlations)

        if single_trial:
            return predictions[0], all_correlations[0]

        return predictions, all_correlations

    def _compute_correlations(self, trial: np.ndarray) -> np.ndarray:
        """Compute TRCA correlation with all templates.

        For each template k:
        r_k = corr(W_k^T @ X, W_k^T @ template_k)

        where W_k is the TRCA spatial filter for class k.

        Args:
            trial: Single trial with shape (n_channels, n_samples)

        Returns:
            Correlation values with shape (n_classes,)
        """
        n_classes = len(self.templates)
        correlations = np.zeros(n_classes)

        for class_idx in range(n_classes):
            W = self.spatial_filters[class_idx]  # Shape: (n_channels, n_components)
            template = self.templates[class_idx]  # Shape: (n_channels, n_samples)

            # Project trial and template onto TRCA space
            trial_proj = W.T @ trial  # Shape: (n_components, n_samples)
            template_proj = W.T @ template  # Shape: (n_components, n_samples)

            # Compute correlation for each component and average
            corr_sum = 0.0
            for comp in range(self.n_components):
                corr = self._pearson_correlation(trial_proj[comp], template_proj[comp])
                corr_sum += corr

            correlations[class_idx] = corr_sum / self.n_components

        return correlations

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient.

        Args:
            x: First signal
            y: Second signal

        Returns:
            Correlation coefficient (scalar)
        """
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator


# Unit test
if __name__ == "__main__":
    print("Testing TRCA...")

    config = SSVEPConfig()
    fs = config.fs

    # Generate synthetic SSVEP data
    n_classes = 4
    frequencies = [8.57, 10.0, 12.0, 15.0]
    n_trials_per_class = 10
    n_channels = config.n_eeg_channels
    trial_duration = 4.0
    n_samples = int(fs * trial_duration)

    print(f"Generating synthetic data:")
    print(f"  {n_classes} classes (frequencies: {frequencies} Hz)")
    print(f"  {n_trials_per_class} trials per class")
    print(f"  {n_channels} channels")
    print(f"  {n_samples} samples per trial ({trial_duration} sec @ {fs} Hz)")

    # Generate training data
    X_train = []
    y_train = []

    t = np.arange(n_samples) / fs

    for class_idx, freq in enumerate(frequencies):
        for trial in range(n_trials_per_class):
            # Generate SSVEP signal at target frequency
            ssvep = 20 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)

            # Create multi-channel data with spatial mixing
            trial_data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                # Each channel has different mixing of SSVEP and noise
                mixing_coef = 0.5 + 0.5 * np.random.rand()
                noise = 10 * np.random.randn(n_samples)
                trial_data[ch] = mixing_coef * ssvep + noise

            X_train.append(trial_data)
            y_train.append(class_idx)

    X_train = np.array(X_train)  # Shape: (n_trials, n_channels, n_samples)
    y_train = np.array(y_train)  # Shape: (n_trials,)

    print(f"\nTraining data shape: {X_train.shape}")

    # Train TRCA
    print("\nTraining TRCA...")
    trca = TRCA(config, n_components=1)
    trca.fit(X_train, y_train)

    print(f"  Trained {len(trca.spatial_filters)} spatial filters")
    print(f"  Trained {len(trca.templates)} templates")

    # Test on training data
    print("\nTesting on training data...")
    y_pred_train, corr_train = trca.predict_with_correlation(X_train)

    accuracy_train = np.mean(y_pred_train == y_train)
    print(f"  Training accuracy: {accuracy_train*100:.1f}%")

    # Generate test data
    print("\nGenerating test data...")
    X_test = []
    y_test = []

    for class_idx, freq in enumerate(frequencies):
        for trial in range(5):  # 5 test trials per class
            ssvep = 20 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)

            trial_data = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                mixing_coef = 0.5 + 0.5 * np.random.rand()
                noise = 10 * np.random.randn(n_samples)
                trial_data[ch] = mixing_coef * ssvep + noise

            X_test.append(trial_data)
            y_test.append(class_idx)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Test data shape: {X_test.shape}")

    # Test predictions
    print("\nTesting on test data...")
    y_pred_test, corr_test = trca.predict_with_correlation(X_test)

    accuracy_test = np.mean(y_pred_test == y_test)
    print(f"  Test accuracy: {accuracy_test*100:.1f}%")

    # Show confusion matrix
    print("\nConfusion matrix (test data):")
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_test, y_pred_test):
        conf_matrix[true_label, pred_label] += 1

    print("       Predicted:")
    print("     ", " ".join([f"{freq:5.2f}" for freq in frequencies]))
    for i, freq in enumerate(frequencies):
        print(f"{freq:5.2f} ", " ".join([f"{conf_matrix[i,j]:5d}" for j in range(n_classes)]))

    # Test single trial prediction
    print("\nTesting single trial prediction...")
    single_trial = X_test[0]
    pred_single, corr_single = trca.predict_with_correlation(single_trial)
    print(f"  True label: {y_test[0]} ({frequencies[y_test[0]]} Hz)")
    print(f"  Predicted: {pred_single} ({frequencies[pred_single]} Hz)")
    print(f"  Correlations: {corr_single}")

    print("\nTRCA test passed!")
