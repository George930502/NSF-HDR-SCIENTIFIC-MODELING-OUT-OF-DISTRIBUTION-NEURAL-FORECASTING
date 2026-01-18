"""
Data Analysis Script for Neural Forecasting Challenge
Analyzes the properties of the μECoG neural signal dataset.
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Set paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

def load_all_datasets():
    """Load all training and test datasets."""
    datasets = {}

    # Training data
    train_files = [
        ("affi_train", "train_data_affi.npz"),
        ("beignet_train", "train_data_beignet.npz"),
        ("affi_train_private", "train_data_affi_2024-03-20_private.npz"),
        ("beignet_train_private_1", "train_data_beignet_2022-06-01_private.npz"),
        ("beignet_train_private_2", "train_data_beignet_2022-06-02_private.npz"),
    ]

    # Test data (masked)
    test_files = [
        ("affi_test", "test_data_affi_masked.npz"),
        ("beignet_test", "test_data_beignet_masked.npz"),
        ("affi_test_private", "test_data_affi_2024-03-20_private_masked.npz"),
        ("beignet_test_private_1", "test_data_beignet_2022-06-01_private_masked.npz"),
        ("beignet_test_private_2", "test_data_beignet_2022-06-02_private_masked.npz"),
    ]

    print("Loading training datasets...")
    for name, filename in train_files:
        filepath = TRAIN_DIR / filename
        if filepath.exists():
            data = np.load(filepath)['arr_0']
            datasets[name] = data
            print(f"  {name}: shape = {data.shape}")
        else:
            print(f"  {name}: NOT FOUND")

    print("\nLoading test datasets...")
    for name, filename in test_files:
        filepath = TEST_DIR / filename
        if filepath.exists():
            data = np.load(filepath)['arr_0']
            datasets[name] = data
            print(f"  {name}: shape = {data.shape}")
        else:
            print(f"  {name}: NOT FOUND")

    return datasets


def analyze_dataset(name, data):
    """Analyze a single dataset."""
    print(f"\n{'='*60}")
    print(f"Analysis of {name}")
    print(f"{'='*60}")

    n_samples, n_timesteps, n_channels, n_features = data.shape
    print(f"Shape: {data.shape}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Timesteps: {n_timesteps}")
    print(f"  - Channels: {n_channels}")
    print(f"  - Features: {n_features}")

    # Analyze the target feature (feature 0)
    target = data[:, :, :, 0]
    print(f"\nTarget feature (index 0) statistics:")
    print(f"  Mean: {target.mean():.6f}")
    print(f"  Std: {target.std():.6f}")
    print(f"  Min: {target.min():.6f}")
    print(f"  Max: {target.max():.6f}")

    # Analyze per-channel statistics
    channel_means = target.mean(axis=(0, 1))
    channel_stds = target.std(axis=(0, 1))
    print(f"\nPer-channel target statistics:")
    print(f"  Channel means range: [{channel_means.min():.6f}, {channel_means.max():.6f}]")
    print(f"  Channel stds range: [{channel_stds.min():.6f}, {channel_stds.max():.6f}]")

    # Analyze other features (frequency bands)
    print(f"\nAll features statistics:")
    for f in range(n_features):
        feat_data = data[:, :, :, f]
        print(f"  Feature {f}: mean={feat_data.mean():.6f}, std={feat_data.std():.6f}, "
              f"min={feat_data.min():.6f}, max={feat_data.max():.6f}")

    # Analyze temporal structure (input vs target timesteps)
    input_data = data[:, :10, :, 0]  # First 10 timesteps
    output_data = data[:, 10:, :, 0]  # Last 10 timesteps

    print(f"\nTemporal analysis (target feature):")
    print(f"  Input (t=0:10): mean={input_data.mean():.6f}, std={input_data.std():.6f}")
    print(f"  Output (t=10:20): mean={output_data.mean():.6f}, std={output_data.std():.6f}")

    # Check for NaN or Inf values
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"\nData quality:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    return {
        'shape': data.shape,
        'target_mean': target.mean(),
        'target_std': target.std(),
        'target_min': target.min(),
        'target_max': target.max(),
        'channel_means': channel_means,
        'channel_stds': channel_stds,
    }


def analyze_cross_session_drift(datasets):
    """Analyze distribution differences across sessions."""
    print(f"\n{'='*60}")
    print("Cross-Session Analysis (Distribution Drift)")
    print(f"{'='*60}")

    # Affi datasets
    affi_keys = [k for k in datasets.keys() if 'affi' in k and 'train' in k]
    if len(affi_keys) > 1:
        print("\nAffi (Monkey A) cross-session comparison:")
        for key in affi_keys:
            data = datasets[key][:, :, :, 0]  # Target feature
            print(f"  {key}: mean={data.mean():.6f}, std={data.std():.6f}")

    # Beignet datasets
    beignet_keys = [k for k in datasets.keys() if 'beignet' in k and 'train' in k]
    if len(beignet_keys) > 1:
        print("\nBeignet (Monkey B) cross-session comparison:")
        for key in beignet_keys:
            data = datasets[key][:, :, :, 0]  # Target feature
            print(f"  {key}: mean={data.mean():.6f}, std={data.std():.6f}")


def analyze_frequency_bands(datasets):
    """Analyze correlation between frequency bands."""
    print(f"\n{'='*60}")
    print("Frequency Band Analysis")
    print(f"{'='*60}")

    for name, data in datasets.items():
        if 'train' in name and 'private' not in name:
            print(f"\n{name}:")
            n_features = data.shape[-1]

            # Flatten spatial and temporal dims
            flat_data = data.reshape(-1, n_features)

            # Compute correlation matrix
            corr_matrix = np.corrcoef(flat_data.T)

            print("  Feature correlation with target (feature 0):")
            for f in range(1, n_features):
                print(f"    Feature {f}: r={corr_matrix[0, f]:.4f}")


def check_masked_test_data(datasets):
    """Check how test data is masked."""
    print(f"\n{'='*60}")
    print("Test Data Masking Analysis")
    print(f"{'='*60}")

    for name, data in datasets.items():
        if 'test' in name:
            print(f"\n{name}:")

            # Check if last 10 timesteps are masked
            input_data = data[:, :10, :, 0]
            masked_data = data[:, 10:, :, 0]

            # Check if masked data repeats the 10th timestep
            timestep_10 = data[:, 9:10, :, 0]  # The 10th timestep (index 9)
            timestep_11 = data[:, 10:11, :, 0]  # The 11th timestep (index 10)

            diff = np.abs(timestep_10 - timestep_11).mean()
            print(f"  Difference between t=9 and t=10: {diff:.6f}")

            # Check if all masked timesteps are the same
            all_same = True
            for t in range(10, 20):
                diff_t = np.abs(data[:, t, :, 0] - data[:, 9, :, 0]).mean()
                if diff_t > 1e-6:
                    all_same = False
                    break
            print(f"  All masked timesteps same as t=9: {all_same}")


def main():
    """Main analysis function."""
    print("Neural Forecasting Dataset Analysis")
    print("=" * 60)

    # Load all datasets
    datasets = load_all_datasets()

    # Analyze each dataset
    results = {}
    for name, data in datasets.items():
        results[name] = analyze_dataset(name, data)

    # Cross-session analysis
    analyze_cross_session_drift(datasets)

    # Frequency band analysis
    analyze_frequency_bands(datasets)

    # Check masked test data
    check_masked_test_data(datasets)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("\nKey observations for model design:")
    print("1. Data shape: N x 20 x C x 9 (samples, timesteps, channels, features)")
    print("2. Target is feature 0; features 1-8 are frequency band decompositions")
    print("3. Input: first 10 timesteps, Output: last 10 timesteps")
    print("4. Monkey A (affi): 239 channels, Monkey B (beignet): 87 channels")
    print("5. Need to handle session drift between training and test data")


if __name__ == "__main__":
    main()
