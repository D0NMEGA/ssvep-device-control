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
from .config import SSVEPConfig


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
            n_components: Number of TRCA components to retain (default: 1 = 2nd eigenvector only)
        """
        self.config = config or SSVEPConfig()
        self.n_components = n_components

        # Model parameters (filled during fit)
        self.spatial_filters = []  # List of (n_channels, n_components, n_filterbanks) arrays
        self.templates = []  # List of (n_channels, n_samples, n_filterbanks) arrays
        self.is_fitted = False

        # Filter bank coefficients (for weighted combination)
        self.fb_coefs = None  # Will be set during fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TRCA':
        """Train TRCA model on calibration data with filter banks.

        Args:
            X: Training data with shape (n_trials, n_channels, n_samples, n_filterbanks)
            y: Labels with shape (n_trials,) containing class indices

        Returns:
            self (fitted model)
        """
        if X.ndim == 4:
            n_trials, n_channels, n_samples, n_filterbanks = X.shape
        else:
            # Backward compatibility: no filter banks
            n_trials, n_channels, n_samples = X.shape
            n_filterbanks = 1
            X = X[..., np.newaxis]  # Add filterbank dimension

        # Compute filter bank coefficients
        from .filterbank import FilterBank
        fb = FilterBank(fs=self.config.fs, num_bands=n_filterbanks)
        self.fb_coefs = fb.get_coefficients()

        # Get unique classes
        classes = np.unique(y)
        n_classes = len(classes)

        # Initialize storage
        self.spatial_filters = []
        self.templates = []

        # For each class and filter bank, compute TRCA spatial filters and template
        for class_idx in classes:
            # Get trials for this class
            class_mask = (y == class_idx)
            X_class = X[class_mask]  # Shape: (n_trials_class, n_channels, n_samples, n_filterbanks)

            # Initialize storage for this class
            W_all_fb = []
            template_all_fb = []

            # Process each filter bank independently
            for fb_i in range(n_filterbanks):
                X_fb = X_class[:, :, :, fb_i]  # Shape: (n_trials, n_channels, n_samples)

                # Compute TRCA spatial filter
                W = self._compute_trca_filter(X_fb)

                # Use 2nd eigenvector (index 1) as per reference
                W = W[:, 1:2]  # Shape: (n_channels, 1)

                # Compute template by averaging trials
                template = np.mean(X_fb, axis=0)  # Shape: (n_channels, n_samples)

                W_all_fb.append(W)
                template_all_fb.append(template)

            # Stack across filter banks
            W_class = np.stack(W_all_fb, axis=2)  # Shape: (n_channels, 1, n_filterbanks)
            template_class = np.stack(template_all_fb, axis=2)  # Shape: (n_channels, n_samples, n_filterbanks)

            self.spatial_filters.append(W_class)
            self.templates.append(template_class)

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

        # CRITICAL: Center each trial (remove mean) as per reference implementation
        X_centered = np.zeros_like(X)
        for trial_i in range(n_trials):
            X_centered[trial_i] = X[trial_i] - X[trial_i].mean(axis=1, keepdims=True)

        # Compute S: inter-trial covariance matrix
        # S = sum over all pairs (i,j) of cov(X_i, X_j)
        S = np.zeros((n_channels, n_channels))

        for i in range(n_trials):
            for j in range(i+1, n_trials):
                Xi = X_centered[i]  # Shape: (n_channels, n_samples)
                Xj = X_centered[j]  # Shape: (n_channels, n_samples)

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
            Xi = X_centered[i]  # Shape: (n_channels, n_samples)
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
            X: Test data with shape:
               - (n_trials, n_channels, n_samples, n_filterbanks) for multiple trials
               - (n_channels, n_samples, n_filterbanks) for single trial with filterbanks
               - (n_channels, n_samples) for single trial without filterbanks (backward compat)

        Returns:
            Predicted class indices with shape (n_trials,) or scalar
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Handle single trial (2D or 3D with filterbanks)
        single_trial = False
        if X.ndim == 2:
            # (n_channels, n_samples) - backward compatibility
            X = X[np.newaxis, :, :]
            single_trial = True
        elif X.ndim == 3:
            # (n_channels, n_samples, n_filterbanks) - single trial with filterbanks
            X = X[np.newaxis, :, :, :]
            single_trial = True

        n_trials = X.shape[0]
        predictions = np.zeros(n_trials, dtype=int)

        for trial_idx in range(n_trials):
            trial = X[trial_idx]

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
            X: Test data with shape:
               - (n_trials, n_channels, n_samples, n_filterbanks) for multiple trials
               - (n_channels, n_samples, n_filterbanks) for single trial with filterbanks
               - (n_channels, n_samples) for single trial without filterbanks (backward compat)

        Returns:
            Tuple of (predictions, correlations)
            - predictions: Class indices with shape (n_trials,) or scalar
            - correlations: Correlation matrix with shape (n_trials, n_classes) or (n_classes,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Handle single trial (2D or 3D with filterbanks)
        single_trial = False
        if X.ndim == 2:
            # (n_channels, n_samples) - backward compatibility
            X = X[np.newaxis, :, :]
            single_trial = True
        elif X.ndim == 3:
            # (n_channels, n_samples, n_filterbanks) - single trial with filterbanks
            X = X[np.newaxis, :, :, :]
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
        """Compute TRCA correlation with all templates using filter bank weighting.

        For each template k and filter bank fb:
        r_k,fb = corr(W_k,fb^T @ X_fb, W_k,fb^T @ template_k,fb)

        Then combine across filter banks using weighted sum:
        r_k = sum_fb (coef_fb * r_k,fb)

        Args:
            trial: Single trial with shape (n_channels, n_samples, n_filterbanks)
                   or (n_channels, n_samples) for backward compatibility

        Returns:
            Correlation values with shape (n_classes,)
        """
        # Handle backward compatibility (no filter banks)
        if trial.ndim == 2:
            trial = trial[..., np.newaxis]

        n_channels, n_samples, n_filterbanks = trial.shape
        n_classes = len(self.templates)

        correlations = np.zeros(n_classes)

        for class_idx in range(n_classes):
            W = self.spatial_filters[class_idx]  # Shape: (n_channels, 1, n_filterbanks)
            template = self.templates[class_idx]  # Shape: (n_channels, n_samples, n_filterbanks)

            # Compute correlation for each filter bank
            fb_correlations = np.zeros(n_filterbanks)

            for fb_i in range(n_filterbanks):
                W_fb = W[:, :, fb_i]  # Shape: (n_channels, 1)
                template_fb = template[:, :, fb_i]  # Shape: (n_channels, n_samples)
                trial_fb = trial[:, :, fb_i]  # Shape: (n_channels, n_samples)

                # Project trial and template onto TRCA space
                trial_proj = W_fb.T @ trial_fb  # Shape: (1, n_samples)
                template_proj = W_fb.T @ template_fb  # Shape: (1, n_samples)

                # Compute Pearson correlation
                corr = self._pearson_correlation(trial_proj[0], template_proj[0])
                fb_correlations[fb_i] = corr

            # Weighted combination across filter banks (key reference feature!)
            correlations[class_idx] = np.dot(self.fb_coefs[:n_filterbanks], fb_correlations)

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
