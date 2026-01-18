"""
Visualization Script for Neural Forecasting Dataset Analysis

This script generates visualizations to complement the statistical analysis.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# Paths
TRAIN_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\dataset\train"
TEST_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\dataset\test"
OUTPUT_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_npz_data(filepath):
    """Load data from npz file."""
    data = np.load(filepath)
    keys = list(data.keys())
    return data[keys[0]]

def plot_value_distributions(datasets, output_path):
    """Plot value distributions for all datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(datasets.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        flat_data = data.flatten()

        # Clip for visualization
        clipped = np.clip(flat_data, np.percentile(flat_data, 1), np.percentile(flat_data, 99))

        ax.hist(clipped, bins=100, alpha=0.7, density=True)
        ax.set_title(f'{name}\nmean={np.mean(flat_data):.1f}, std={np.std(flat_data):.1f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')

    plt.suptitle('Value Distributions (1st-99th percentile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_distributions(data, name, output_path):
    """Plot distributions for each feature."""
    n_features = data.shape[-1]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    feature_names = ['Target (LFP)', 'Band 1', 'Band 2', 'Band 3', 'Band 4',
                     'Band 5', 'Band 6', 'Band 7', 'Band 8']

    for f in range(n_features):
        ax = axes[f]
        feat_data = data[:, :, :, f].flatten()

        # Clip for visualization
        clipped = np.clip(feat_data, np.percentile(feat_data, 1), np.percentile(feat_data, 99))

        ax.hist(clipped, bins=80, alpha=0.7, color=f'C{f}')
        ax.set_title(f'{feature_names[f]}\nmean={np.mean(feat_data):.1f}, std={np.std(feat_data):.1f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    plt.suptitle(f'Feature Distributions - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_temporal_patterns(data, name, output_path):
    """Plot temporal patterns in the data."""
    n_timesteps = data.shape[1]
    n_features = data.shape[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean value across timesteps for each feature
    ax = axes[0, 0]
    timestep_means = np.mean(data, axis=(0, 2))  # (timesteps, features)
    for f in range(n_features):
        ax.plot(range(n_timesteps), timestep_means[:, f], label=f'Feature {f}', linewidth=2)
    ax.axvline(x=9.5, color='red', linestyle='--', label='Prediction boundary')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Value Across Timesteps')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Target feature temporal evolution (sample trajectories)
    ax = axes[0, 1]
    target = data[:, :, 0, 0]  # First channel, target feature
    # Plot a few random sample trajectories
    np.random.seed(42)
    sample_indices = np.random.choice(len(data), min(20, len(data)), replace=False)
    for i, idx in enumerate(sample_indices):
        alpha = 0.3 if i > 0 else 1.0
        ax.plot(range(n_timesteps), target[idx], alpha=alpha, linewidth=1)
    ax.axvline(x=9.5, color='red', linestyle='--', label='Prediction boundary')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Target Value (Channel 0)')
    ax.set_title('Sample Trajectories (Channel 0, Feature 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Autocorrelation of target feature
    ax = axes[1, 0]
    target_flat = data[:, :, :, 0]  # (samples, timesteps, channels)
    autocorrs = []
    for lag in range(1, n_timesteps):
        t_data = target_flat[:, :-lag, :].flatten()
        t_lag_data = target_flat[:, lag:, :].flatten()
        corr = np.corrcoef(t_data, t_lag_data)[0, 1]
        autocorrs.append(corr)
    ax.bar(range(1, n_timesteps), autocorrs, color='steelblue')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation of Target Feature')
    ax.grid(True, alpha=0.3)

    # 4. Timestep-to-timestep differences
    ax = axes[1, 1]
    diffs = data[:, 1:, :, 0] - data[:, :-1, :, 0]  # Target feature differences
    diff_means = np.mean(np.abs(diffs), axis=(0, 2))  # Mean abs diff per timestep
    ax.bar(range(len(diff_means)), diff_means, color='coral')
    ax.axvline(x=8.5, color='red', linestyle='--', label='Last known diff')
    ax.set_xlabel('Timestep Transition (t -> t+1)')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Temporal Volatility (Target Feature)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Temporal Patterns - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_session_drift(data1, data2, name1, name2, output_path):
    """Plot session drift comparison."""
    n_features = data1.shape[-1]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    feature_names = ['Target (LFP)', 'Band 1', 'Band 2', 'Band 3', 'Band 4',
                     'Band 5', 'Band 6', 'Band 7', 'Band 8']

    for f in range(n_features):
        ax = axes[f]

        feat1 = data1[:, :, :, f].flatten()
        feat2 = data2[:, :, :, f].flatten()

        # Use same bins for comparison
        min_val = min(np.percentile(feat1, 1), np.percentile(feat2, 1))
        max_val = max(np.percentile(feat1, 99), np.percentile(feat2, 99))
        bins = np.linspace(min_val, max_val, 60)

        ax.hist(feat1, bins=bins, alpha=0.5, label=name1, density=True)
        ax.hist(feat2, bins=bins, alpha=0.5, label=name2, density=True)
        ax.set_title(f'{feature_names[f]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.suptitle(f'Distribution Shift: {name1} vs {name2}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_correlations(data, name, output_path):
    """Plot feature correlation heatmap."""
    n_features = data.shape[-1]

    # Flatten for correlation
    flat_data = data.reshape(-1, n_features)
    corr_matrix = np.corrcoef(flat_data.T)

    feature_names = ['Target', 'Band 1', 'Band 2', 'Band 3', 'Band 4',
                     'Band 5', 'Band 6', 'Band 7', 'Band 8']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add labels
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)

    # Add correlation values
    for i in range(n_features):
        for j in range(n_features):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    plt.colorbar(im, label='Correlation')
    ax.set_title(f'Feature Correlation Matrix - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_channel_variability(data, name, output_path):
    """Plot channel variability analysis."""
    n_channels = data.shape[2]

    # Calculate variance for each channel
    channel_vars = []
    channel_means = []
    for c in range(n_channels):
        ch_data = data[:, :, c, :].flatten()
        channel_vars.append(np.var(ch_data))
        channel_means.append(np.mean(ch_data))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Channel variance bar plot
    ax = axes[0, 0]
    sorted_indices = np.argsort(channel_vars)[::-1]
    ax.bar(range(n_channels), [channel_vars[i] for i in sorted_indices], color='steelblue')
    ax.set_xlabel('Channel (sorted by variance)')
    ax.set_ylabel('Variance')
    ax.set_title('Channel Variance Distribution')

    # 2. Channel mean vs variance scatter
    ax = axes[0, 1]
    ax.scatter(channel_means, channel_vars, alpha=0.6)
    ax.set_xlabel('Channel Mean')
    ax.set_ylabel('Channel Variance')
    ax.set_title('Mean vs Variance by Channel')

    # 3. Channel correlation matrix sample (first 20 channels)
    ax = axes[1, 0]
    n_show = min(20, n_channels)
    target_data = data[:, :, :n_show, 0]  # (samples, timesteps, channels)
    reshaped = target_data.reshape(-1, n_show)
    ch_corr = np.corrcoef(reshaped.T)
    im = ax.imshow(ch_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    ax.set_title(f'Channel Correlation (first {n_show} channels, Target Feature)')
    plt.colorbar(im, ax=ax)

    # 4. Histogram of all channel correlations
    ax = axes[1, 1]
    # Get upper triangle of correlation matrix (excluding diagonal)
    full_target = data[:, :, :, 0].reshape(-1, n_channels)
    full_corr = np.corrcoef(full_target.T)
    upper_triangle = full_corr[np.triu_indices(n_channels, k=1)]
    ax.hist(upper_triangle, bins=50, color='coral', alpha=0.7)
    ax.axvline(np.mean(upper_triangle), color='red', linestyle='--',
              label=f'Mean={np.mean(upper_triangle):.2f}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Inter-Channel Correlations')
    ax.legend()

    plt.suptitle(f'Channel Analysis - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_prediction_difficulty(data, name, output_path):
    """Plot factors affecting prediction difficulty."""
    n_timesteps = data.shape[1]
    init_steps = 10

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Known vs Future distributions
    ax = axes[0, 0]
    known_target = data[:, :init_steps, :, 0].flatten()
    future_target = data[:, init_steps:, :, 0].flatten()

    bins = np.linspace(min(np.percentile(known_target, 1), np.percentile(future_target, 1)),
                       max(np.percentile(known_target, 99), np.percentile(future_target, 99)), 60)
    ax.hist(known_target, bins=bins, alpha=0.5, label=f'Known (t=0-9)\nmean={np.mean(known_target):.1f}', density=True)
    ax.hist(future_target, bins=bins, alpha=0.5, label=f'Future (t=10-19)\nmean={np.mean(future_target):.1f}', density=True)
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Future Target Distribution')
    ax.legend()

    # 2. Predictability decay
    ax = axes[0, 1]
    last_known = data[:, init_steps-1, :, 0]  # (samples, channels)
    correlations = []
    for t in range(init_steps, n_timesteps):
        future_t = data[:, t, :, 0]
        corr = np.corrcoef(last_known.flatten(), future_t.flatten())[0, 1]
        correlations.append(corr)
    ax.plot(range(init_steps, n_timesteps), correlations, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Future Timestep')
    ax.set_ylabel('Correlation with Last Known (t=9)')
    ax.set_title('Predictability Decay Over Horizon')
    ax.grid(True, alpha=0.3)

    # 3. Error accumulation simulation (naive baseline: repeat last value)
    ax = axes[1, 0]
    last_known_values = data[:, init_steps-1:init_steps, :, 0]  # (samples, 1, channels)
    predictions = np.repeat(last_known_values, n_timesteps - init_steps, axis=1)
    actual = data[:, init_steps:, :, 0]
    errors = np.abs(predictions - actual)
    mean_errors_per_step = np.mean(errors, axis=(0, 2))  # (timesteps,)
    ax.bar(range(init_steps, n_timesteps), mean_errors_per_step, color='coral')
    ax.set_xlabel('Future Timestep')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Naive Baseline Error (Repeat Last Value)')
    ax.grid(True, alpha=0.3)

    # 4. Per-timestep variance
    ax = axes[1, 1]
    timestep_vars = np.var(data[:, :, :, 0], axis=(0, 2))  # Variance per timestep
    ax.bar(range(n_timesteps), timestep_vars, color='steelblue')
    ax.axvline(x=9.5, color='red', linestyle='--', label='Prediction boundary')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Variance')
    ax.set_title('Target Variance per Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Prediction Difficulty Analysis - {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}

    train_files = [
        ("affi", "train_data_affi.npz"),
        ("affi_private", "train_data_affi_2024-03-20_private.npz"),
        ("beignet", "train_data_beignet.npz"),
        ("beignet_private_1", "train_data_beignet_2022-06-01_private.npz"),
        ("beignet_private_2", "train_data_beignet_2022-06-02_private.npz"),
    ]

    for name, filename in train_files:
        filepath = os.path.join(TRAIN_DIR, filename)
        if os.path.exists(filepath):
            datasets[name] = load_npz_data(filepath)
            print(f"  Loaded {name}: shape = {datasets[name].shape}")

    # Generate plots
    print("\nGenerating visualizations...")

    # 1. Overall value distributions
    plot_value_distributions(datasets, os.path.join(OUTPUT_DIR, "01_value_distributions.png"))

    # 2. Feature distributions for each dataset
    for name in ["affi", "beignet"]:
        if name in datasets:
            plot_feature_distributions(datasets[name], name,
                                      os.path.join(OUTPUT_DIR, f"02_feature_dist_{name}.png"))

    # 3. Temporal patterns
    for name in ["affi", "beignet"]:
        if name in datasets:
            plot_temporal_patterns(datasets[name], name,
                                  os.path.join(OUTPUT_DIR, f"03_temporal_{name}.png"))

    # 4. Session drift comparisons
    if "affi" in datasets and "affi_private" in datasets:
        plot_session_drift(datasets["affi"], datasets["affi_private"],
                          "affi", "affi_private",
                          os.path.join(OUTPUT_DIR, "04_drift_affi.png"))

    if "beignet" in datasets and "beignet_private_1" in datasets:
        plot_session_drift(datasets["beignet"], datasets["beignet_private_1"],
                          "beignet", "beignet_private_1",
                          os.path.join(OUTPUT_DIR, "04_drift_beignet.png"))

    # 5. Feature correlations
    for name in ["affi", "beignet"]:
        if name in datasets:
            plot_feature_correlations(datasets[name], name,
                                     os.path.join(OUTPUT_DIR, f"05_feat_corr_{name}.png"))

    # 6. Channel analysis
    for name in ["affi", "beignet"]:
        if name in datasets:
            plot_channel_variability(datasets[name], name,
                                    os.path.join(OUTPUT_DIR, f"06_channel_{name}.png"))

    # 7. Prediction difficulty
    for name in ["affi", "beignet"]:
        if name in datasets:
            plot_prediction_difficulty(datasets[name], name,
                                      os.path.join(OUTPUT_DIR, f"07_pred_diff_{name}.png"))

    print("\n" + "="*60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
