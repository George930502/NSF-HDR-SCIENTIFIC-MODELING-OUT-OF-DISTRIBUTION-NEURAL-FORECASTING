"""
Comprehensive Statistical Analysis of Neural Forecasting Dataset

This script performs a detailed analysis of neural forecasting data to understand
why models may have high MSE and what preprocessing might help.
"""

import numpy as np
import os
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

# Paths
TRAIN_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\dataset\train"
TEST_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\dataset\test"
OUTPUT_DIR = r"C:\Users\george\Desktop\dev\001-NSF-HDR-Hackathon\Nerual-Forecasting\analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_npz_data(filepath):
    """Load data from npz file."""
    data = np.load(filepath)
    # npz files store arrays with keys, get the first one
    keys = list(data.keys())
    return data[keys[0]]

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def analyze_distribution(data, name):
    """Analyze distribution statistics for a dataset."""
    print(f"\n--- Distribution Analysis for {name} ---")
    print(f"Shape: {data.shape}")
    print(f"  (samples, timesteps, channels, features)")

    # Flatten for overall statistics
    flat_data = data.flatten()

    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean:     {np.mean(flat_data):.6f}")
    print(f"  Std:      {np.std(flat_data):.6f}")
    print(f"  Min:      {np.min(flat_data):.6f}")
    print(f"  Max:      {np.max(flat_data):.6f}")
    print(f"  Median:   {np.median(flat_data):.6f}")

    # Skewness and kurtosis
    skew = stats.skew(flat_data)
    kurt = stats.kurtosis(flat_data)
    print(f"  Skewness: {skew:.6f}")
    print(f"  Kurtosis: {kurt:.6f}")

    # Outliers (beyond 3 sigma)
    mean_val = np.mean(flat_data)
    std_val = np.std(flat_data)
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val
    outliers = np.sum((flat_data < lower_bound) | (flat_data > upper_bound))
    outlier_pct = 100 * outliers / len(flat_data)
    print(f"\nOutliers (>3 sigma):")
    print(f"  Count:      {outliers:,}")
    print(f"  Percentage: {outlier_pct:.4f}%")
    print(f"  Bounds:     [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Per-feature statistics
    n_features = data.shape[-1]
    print(f"\nPer-Feature Statistics (feature 0 is target):")
    print(f"{'Feature':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Skew':>10}")
    print("-" * 70)

    feature_stats = []
    for f in range(n_features):
        feat_data = data[:, :, :, f].flatten()
        feat_mean = np.mean(feat_data)
        feat_std = np.std(feat_data)
        feat_min = np.min(feat_data)
        feat_max = np.max(feat_data)
        feat_skew = stats.skew(feat_data)
        feature_stats.append({
            'mean': feat_mean, 'std': feat_std,
            'min': feat_min, 'max': feat_max, 'skew': feat_skew
        })
        print(f"{f:<10} {feat_mean:>12.4f} {feat_std:>12.4f} {feat_min:>12.4f} {feat_max:>12.4f} {feat_skew:>10.4f}")

    return {
        'mean': np.mean(flat_data),
        'std': np.std(flat_data),
        'min': np.min(flat_data),
        'max': np.max(flat_data),
        'skew': skew,
        'kurtosis': kurt,
        'outlier_pct': outlier_pct,
        'feature_stats': feature_stats
    }

def analyze_temporal_properties(data, name):
    """Analyze temporal properties of the data."""
    print(f"\n--- Temporal Analysis for {name} ---")

    n_samples, n_timesteps, n_channels, n_features = data.shape

    # Calculate autocorrelation across timesteps for target feature (feature 0)
    print("\nAutocorrelation Analysis (Target Feature 0):")

    # Average autocorrelation across samples and channels
    autocorr_lags = []
    for lag in range(1, min(n_timesteps, 10)):
        # Get feature 0 data
        target = data[:, :, :, 0]  # (samples, timesteps, channels)

        # Calculate correlation between t and t+lag
        t_data = target[:, :-lag, :].flatten()
        t_lag_data = target[:, lag:, :].flatten()

        if len(t_data) > 0 and np.std(t_data) > 0 and np.std(t_lag_data) > 0:
            corr = np.corrcoef(t_data, t_lag_data)[0, 1]
            autocorr_lags.append((lag, corr))
            print(f"  Lag {lag}: {corr:.4f}")

    # Difference analysis (t vs t+1)
    print("\nTimestep Difference Analysis (t to t+1):")
    diffs = data[:, 1:, :, :] - data[:, :-1, :, :]
    diff_stats = {
        'mean': np.mean(diffs),
        'std': np.std(diffs),
        'abs_mean': np.mean(np.abs(diffs))
    }
    print(f"  Mean Difference: {diff_stats['mean']:.6f}")
    print(f"  Std of Differences: {diff_stats['std']:.6f}")
    print(f"  Mean Absolute Difference: {diff_stats['abs_mean']:.6f}")

    # Per-feature difference
    print("\nPer-Feature Timestep Differences:")
    for f in range(n_features):
        feat_diffs = diffs[:, :, :, f].flatten()
        print(f"  Feature {f}: mean_abs_diff = {np.mean(np.abs(feat_diffs)):.6f}")

    # Trend within samples
    print("\nTrend Analysis (mean across timesteps within each sample):")
    # Calculate mean value at each timestep, averaged across samples and channels
    timestep_means = np.mean(data, axis=(0, 2))  # (timesteps, features)
    print(f"  Timestep means for Feature 0 (target):")
    for t in range(n_timesteps):
        print(f"    t={t}: {timestep_means[t, 0]:.6f}")

    # Trend slope
    for f in range(n_features):
        x = np.arange(n_timesteps)
        y = timestep_means[:, f]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        if f == 0:
            print(f"\n  Linear trend for Feature 0 (target):")
            print(f"    Slope: {slope:.6f}, R-squared: {r_value**2:.4f}")

    return {
        'autocorr': autocorr_lags,
        'diff_stats': diff_stats
    }

def analyze_cross_session_drift(data1, data2, name1, name2):
    """Compare statistics between two datasets to measure drift."""
    print(f"\n--- Cross-Session Drift Analysis: {name1} vs {name2} ---")

    n_features = data1.shape[-1]
    n_channels = data1.shape[2]

    # Overall distribution comparison
    flat1 = data1.flatten()
    flat2 = data2.flatten()

    print("\nOverall Distribution Shift:")
    print(f"  {name1}: mean={np.mean(flat1):.4f}, std={np.std(flat1):.4f}")
    print(f"  {name2}: mean={np.mean(flat2):.4f}, std={np.std(flat2):.4f}")
    print(f"  Mean shift: {np.mean(flat2) - np.mean(flat1):.4f}")
    print(f"  Std ratio: {np.std(flat2) / np.std(flat1):.4f}")

    # KL Divergence approximation using histogram
    print("\nJensen-Shannon Divergence (per feature):")
    js_divergences = []
    for f in range(n_features):
        feat1 = data1[:, :, :, f].flatten()
        feat2 = data2[:, :, :, f].flatten()

        # Create histograms for JS divergence
        min_val = min(feat1.min(), feat2.min())
        max_val = max(feat1.max(), feat2.max())
        bins = np.linspace(min_val, max_val, 100)

        hist1, _ = np.histogram(feat1, bins=bins, density=True)
        hist2, _ = np.histogram(feat2, bins=bins, density=True)

        # Add small epsilon to avoid division by zero
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        js_div = jensenshannon(hist1, hist2)
        js_divergences.append(js_div)
        print(f"  Feature {f}: JS Divergence = {js_div:.6f}")

    # Per-channel drift (for common channels)
    min_channels = min(data1.shape[2], data2.shape[2])
    print(f"\nPer-Channel Mean Shift (first {min(10, min_channels)} channels):")
    channel_drifts = []
    for c in range(min(10, min_channels)):
        ch1_mean = np.mean(data1[:, :, c, :])
        ch2_mean = np.mean(data2[:, :, c, :])
        drift = ch2_mean - ch1_mean
        channel_drifts.append(drift)
        print(f"  Channel {c}: {ch1_mean:.4f} -> {ch2_mean:.4f} (shift: {drift:.4f})")

    # Find channels with most drift
    all_channel_drifts = []
    for c in range(min_channels):
        ch1_mean = np.mean(data1[:, :, c, :])
        ch2_mean = np.mean(data2[:, :, c, :])
        all_channel_drifts.append((c, abs(ch2_mean - ch1_mean)))

    sorted_drifts = sorted(all_channel_drifts, key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 Channels with Most Drift:")
    for c, drift in sorted_drifts[:5]:
        print(f"  Channel {c}: absolute drift = {drift:.6f}")

    return {
        'js_divergences': js_divergences,
        'mean_shift': np.mean(flat2) - np.mean(flat1)
    }

def analyze_feature_correlations(data, name):
    """Analyze correlations between target (feature 0) and other features."""
    print(f"\n--- Feature Correlation Analysis for {name} ---")

    n_features = data.shape[-1]

    # Flatten spatial dimensions for correlation
    flat_data = data.reshape(-1, n_features)  # (samples*timesteps*channels, features)

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(flat_data.T)

    print("\nCorrelation with Target (Feature 0):")
    correlations = []
    for f in range(1, n_features):
        corr = corr_matrix[0, f]
        correlations.append((f, corr))
        print(f"  Feature {f}: r = {corr:.4f}")

    # Sort by absolute correlation
    sorted_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
    print("\nFeatures Ranked by Predictive Power (correlation with target):")
    for f, corr in sorted_corr:
        print(f"  Feature {f}: |r| = {abs(corr):.4f}")

    # Full correlation matrix
    print("\nFull Feature Correlation Matrix:")
    print("    " + "".join([f"{f:>8}" for f in range(n_features)]))
    for i in range(n_features):
        row_str = f"{i:<3} "
        for j in range(n_features):
            row_str += f"{corr_matrix[i,j]:>8.3f}"
        print(row_str)

    return {
        'target_correlations': correlations,
        'correlation_matrix': corr_matrix
    }

def analyze_channels(data, name):
    """Analyze per-channel statistics."""
    print(f"\n--- Channel Analysis for {name} ---")

    n_samples, n_timesteps, n_channels, n_features = data.shape

    print(f"Number of channels: {n_channels}")

    # Per-channel statistics
    channel_stats = []
    for c in range(n_channels):
        ch_data = data[:, :, c, :].flatten()
        ch_mean = np.mean(ch_data)
        ch_std = np.std(ch_data)
        ch_var = np.var(ch_data)
        channel_stats.append({
            'channel': c,
            'mean': ch_mean,
            'std': ch_std,
            'variance': ch_var
        })

    # Sort by variance (most variable channels)
    sorted_by_var = sorted(channel_stats, key=lambda x: x['variance'], reverse=True)

    print("\nTop 10 Most Variable Channels:")
    print(f"{'Channel':<10} {'Mean':>12} {'Std':>12} {'Variance':>12}")
    print("-" * 50)
    for cs in sorted_by_var[:10]:
        print(f"{cs['channel']:<10} {cs['mean']:>12.4f} {cs['std']:>12.4f} {cs['variance']:>12.4f}")

    print("\nLeast 10 Variable Channels:")
    for cs in sorted_by_var[-10:]:
        print(f"{cs['channel']:<10} {cs['mean']:>12.4f} {cs['std']:>12.4f} {cs['variance']:>12.4f}")

    # Channel-channel correlations (for target feature)
    print("\nChannel Correlation Matrix Sample (first 10 channels, feature 0):")
    target_data = data[:, :, :min(10, n_channels), 0]  # (samples, timesteps, channels)
    reshaped = target_data.reshape(-1, min(10, n_channels))  # (samples*timesteps, channels)
    ch_corr = np.corrcoef(reshaped.T)

    print("    " + "".join([f"{c:>7}" for c in range(min(10, n_channels))]))
    for i in range(min(10, n_channels)):
        row_str = f"{i:<3} "
        for j in range(min(10, n_channels)):
            row_str += f"{ch_corr[i,j]:>7.3f}"
        print(row_str)

    return channel_stats

def analyze_prediction_difficulty(data, name):
    """Analyze factors that make prediction difficult."""
    print(f"\n--- Prediction Difficulty Analysis for {name} ---")

    n_samples, n_timesteps, n_channels, n_features = data.shape
    init_steps = 10  # First 10 steps are known

    # Compare known vs future timesteps
    known_data = data[:, :init_steps, :, :]
    future_data = data[:, init_steps:, :, :]

    print("\nKnown vs Future Timesteps (Feature 0 - Target):")
    known_target = known_data[:, :, :, 0]
    future_target = future_data[:, :, :, 0]

    print(f"  Known (t=0-9):  mean={np.mean(known_target):.4f}, std={np.std(known_target):.4f}")
    print(f"  Future (t=10-19): mean={np.mean(future_target):.4f}, std={np.std(future_target):.4f}")

    # Predictability: correlation between last known step and future steps
    print("\nPredictability Analysis (correlation of last known step with future):")
    last_known = data[:, init_steps-1, :, 0]  # (samples, channels)

    for t in range(init_steps, n_timesteps):
        future_t = data[:, t, :, 0]  # (samples, channels)
        # Flatten for correlation
        corr = np.corrcoef(last_known.flatten(), future_t.flatten())[0, 1]
        print(f"  t={init_steps-1} vs t={t}: r = {corr:.4f}")

    # Signal-to-noise ratio estimate
    print("\nSignal vs Noise Analysis:")
    # Consider temporal trend as signal, deviations as noise
    for f in range(n_features):
        feat_data = data[:, :, :, f]  # (samples, timesteps, channels)

        # Calculate per-sample temporal mean as "signal"
        sample_temporal_means = np.mean(feat_data, axis=1)  # (samples, channels)

        # Deviations from mean as "noise"
        expanded_means = np.expand_dims(sample_temporal_means, axis=1)  # (samples, 1, channels)
        deviations = feat_data - expanded_means

        signal_power = np.var(sample_temporal_means)
        noise_power = np.var(deviations)
        snr = signal_power / (noise_power + 1e-10)

        if f < 5:
            print(f"  Feature {f}: SNR = {snr:.4f} (signal_var={signal_power:.4f}, noise_var={noise_power:.4f})")

def main():
    print_section("NEURAL FORECASTING DATASET STATISTICAL ANALYSIS")

    # Load all datasets
    print("\nLoading datasets...")

    datasets = {}

    # Training datasets
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
        else:
            print(f"  WARNING: {filepath} not found")

    # Test datasets (masked)
    test_files = [
        ("affi_test", "test_data_affi_masked.npz"),
        ("affi_private_test", "test_data_affi_2024-03-20_private_masked.npz"),
        ("beignet_test", "test_data_beignet_masked.npz"),
        ("beignet_private_1_test", "test_data_beignet_2022-06-01_private_masked.npz"),
        ("beignet_private_2_test", "test_data_beignet_2022-06-02_private_masked.npz"),
    ]

    for name, filename in test_files:
        filepath = os.path.join(TEST_DIR, filename)
        if os.path.exists(filepath):
            datasets[name] = load_npz_data(filepath)
            print(f"  Loaded {name}: shape = {datasets[name].shape}")
        else:
            print(f"  WARNING: {filepath} not found")

    # =========================================================================
    # 1. DISTRIBUTION ANALYSIS
    # =========================================================================
    print_section("1. DISTRIBUTION ANALYSIS")

    distribution_results = {}
    for name in ["affi", "affi_private", "beignet", "beignet_private_1", "beignet_private_2"]:
        if name in datasets:
            distribution_results[name] = analyze_distribution(datasets[name], name)

    # =========================================================================
    # 2. TEMPORAL PROPERTIES
    # =========================================================================
    print_section("2. TEMPORAL PROPERTIES")

    temporal_results = {}
    for name in ["affi", "beignet"]:
        if name in datasets:
            temporal_results[name] = analyze_temporal_properties(datasets[name], name)

    # =========================================================================
    # 3. CROSS-SESSION DRIFT
    # =========================================================================
    print_section("3. CROSS-SESSION DRIFT ANALYSIS")

    drift_results = {}

    # Compare affi main vs private
    if "affi" in datasets and "affi_private" in datasets:
        drift_results["affi_drift"] = analyze_cross_session_drift(
            datasets["affi"], datasets["affi_private"],
            "affi (main)", "affi_private"
        )

    # Compare beignet main vs private
    if "beignet" in datasets and "beignet_private_1" in datasets:
        drift_results["beignet_drift_1"] = analyze_cross_session_drift(
            datasets["beignet"], datasets["beignet_private_1"],
            "beignet (main)", "beignet_private_1"
        )

    if "beignet" in datasets and "beignet_private_2" in datasets:
        drift_results["beignet_drift_2"] = analyze_cross_session_drift(
            datasets["beignet"], datasets["beignet_private_2"],
            "beignet (main)", "beignet_private_2"
        )

    # Compare the two private beignet datasets
    if "beignet_private_1" in datasets and "beignet_private_2" in datasets:
        drift_results["beignet_private_comparison"] = analyze_cross_session_drift(
            datasets["beignet_private_1"], datasets["beignet_private_2"],
            "beignet_private_1", "beignet_private_2"
        )

    # =========================================================================
    # 4. FEATURE CORRELATIONS
    # =========================================================================
    print_section("4. FEATURE CORRELATIONS")

    correlation_results = {}
    for name in ["affi", "beignet"]:
        if name in datasets:
            correlation_results[name] = analyze_feature_correlations(datasets[name], name)

    # =========================================================================
    # 5. CHANNEL ANALYSIS
    # =========================================================================
    print_section("5. CHANNEL ANALYSIS")

    channel_results = {}
    for name in ["affi", "beignet"]:
        if name in datasets:
            channel_results[name] = analyze_channels(datasets[name], name)

    # =========================================================================
    # 6. PREDICTION DIFFICULTY ANALYSIS
    # =========================================================================
    print_section("6. PREDICTION DIFFICULTY ANALYSIS")

    for name in ["affi", "beignet"]:
        if name in datasets:
            analyze_prediction_difficulty(datasets[name], name)

    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print_section("SUMMARY AND RECOMMENDATIONS")

    print("\n=== KEY FINDINGS ===\n")

    # Data range issues
    print("1. DATA SCALE AND DISTRIBUTION:")
    for name in ["affi", "beignet"]:
        if name in distribution_results:
            r = distribution_results[name]
            print(f"   {name}:")
            print(f"     - Value range: [{r['min']:.2f}, {r['max']:.2f}]")
            print(f"     - Mean: {r['mean']:.4f}, Std: {r['std']:.4f}")
            print(f"     - Skewness: {r['skew']:.4f}, Kurtosis: {r['kurtosis']:.4f}")
            print(f"     - Outliers (>3sigma): {r['outlier_pct']:.2f}%")

    # Drift analysis summary
    print("\n2. SESSION DRIFT:")
    if "affi_drift" in drift_results:
        print(f"   affi main vs private: mean shift = {drift_results['affi_drift']['mean_shift']:.4f}")
        max_js = max(drift_results['affi_drift']['js_divergences'])
        print(f"     Max JS divergence across features: {max_js:.4f}")

    if "beignet_drift_1" in drift_results:
        print(f"   beignet main vs private_1: mean shift = {drift_results['beignet_drift_1']['mean_shift']:.4f}")
        max_js = max(drift_results['beignet_drift_1']['js_divergences'])
        print(f"     Max JS divergence across features: {max_js:.4f}")

    # Feature importance
    print("\n3. FEATURE IMPORTANCE (correlation with target):")
    for name in ["affi", "beignet"]:
        if name in correlation_results:
            corrs = correlation_results[name]['target_correlations']
            sorted_corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
            print(f"   {name} - Top 3 predictive features:")
            for f, c in sorted_corrs[:3]:
                print(f"     Feature {f}: r = {c:.4f}")

    # Recommendations
    print("\n=== RECOMMENDATIONS FOR REDUCING MSE ===\n")

    print("1. NORMALIZATION:")
    print("   - Current normalization uses mean +/- 4*std range")
    print("   - Consider per-channel normalization (z-score) to handle channel variability")
    print("   - Consider robust normalization (median/IQR) given outliers")

    print("\n2. HANDLING SESSION DRIFT:")
    print("   - Significant distribution shift between main and private datasets")
    print("   - Consider domain adaptation techniques")
    print("   - Add batch normalization layers that can adapt to new sessions")
    print("   - Train on augmented data with artificial drift")

    print("\n3. FEATURE ENGINEERING:")
    print("   - Features have varying correlations with target")
    print("   - Consider attention mechanism to weight feature importance")
    print("   - Low-pass filtering may help if high-frequency noise is present")

    print("\n4. MODEL ARCHITECTURE:")
    print("   - Strong autocorrelation suggests temporal models (LSTM/GRU) are appropriate")
    print("   - Consider Transformer architecture for capturing long-range dependencies")
    print("   - Multi-scale modeling may help capture both fast and slow dynamics")

    print("\n5. TRAINING STRATEGY:")
    print("   - Include private datasets in training to expose model to drift")
    print("   - Use curriculum learning: start with easy samples, progress to harder")
    print("   - Consider multi-task learning with auxiliary prediction targets")

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
