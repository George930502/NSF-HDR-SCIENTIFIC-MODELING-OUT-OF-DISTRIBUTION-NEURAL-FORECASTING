"""
Ridge Regression Tuning V3 - Hybrid approach
- Use Ridge for affi (beats AR(1))
- Test AR(1) vs Ridge for beignet
- Try very low alphas for beignet
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(monkey: str):
    """Load training data."""
    if monkey == "affi":
        files = [
            ("train_data_affi.npz", "affi"),
            ("train_data_affi_2024-03-20_private.npz", "affi_private"),
        ]
    else:
        files = [
            ("train_data_beignet.npz", "beignet"),
            ("train_data_beignet_2022-06-01_private.npz", "beignet_private_1"),
            ("train_data_beignet_2022-06-02_private.npz", "beignet_private_2"),
        ]

    datasets = {}
    all_data = []
    for f, name in files:
        path = os.path.join(TRAIN_DIR, f)
        data = np.load(path)['arr_0']
        datasets[name] = data
        all_data.append(data)

    return np.concatenate(all_data, axis=0), datasets


def prepare_features(X):
    """Prepare features."""
    X_input = X[:, :10, :, 0]
    N, T, C = X_input.shape

    mean = X_input.mean(axis=(1, 2), keepdims=True)
    std = X_input.std(axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    X_norm = (X_input - mean) / std

    last_val = X_norm[:, -1, :]
    window_mean = X_norm.mean(axis=1)
    velocity = X_norm[:, -1, :] - X_norm[:, -2, :]
    X_flat = X_norm.reshape(N, T * C)

    features = np.concatenate([X_flat, last_val, window_mean, velocity], axis=1)

    return features, mean, std


def prepare_target(X):
    """Prepare target."""
    X_input = X[:, :10, :, 0]
    target = X[:, 10:, :, 0]

    N, T, C = X_input.shape

    mean = X_input.mean(axis=(1, 2), keepdims=True)
    std = X_input.std(axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    target_norm = (target - mean) / std

    return target_norm.reshape(N, -1), mean, std


def ar1_predict(X, decay=0.7):
    """AR(1) prediction."""
    X_input = X[:, :10, :, 0]
    N, T, C = X_input.shape

    last_val = X_input[:, -1, :]
    channel_mean = X_input.mean(axis=1)

    predictions = np.zeros((N, 10, C))
    current = last_val.copy()
    for t in range(10):
        predictions[:, t, :] = current
        current = decay * current + (1 - decay) * channel_mean

    return predictions


def cross_validate_beignet_detailed():
    """Detailed CV for beignet with very low alphas."""
    print("\n" + "="*60)
    print("Detailed beignet tuning")
    print("="*60)

    X_all, datasets = load_data("beignet")
    n_channels = X_all.shape[2]

    # Test very low alphas
    alphas = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Test different decay values for AR(1)
    decays = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_results = {alpha: [] for alpha in alphas}
    ar1_results = {decay: [] for decay in decays}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        X_train, X_val = X_all[train_idx], X_all[val_idx]

        feat_train, _, _ = prepare_features(X_train)
        target_train, _, _ = prepare_target(X_train)
        feat_val, mean_val, std_val = prepare_features(X_val)

        target_val_orig = X_val[:, 10:, :, 0]

        # Ridge with different alphas
        for alpha in alphas:
            model = Ridge(alpha=alpha, solver='auto')
            model.fit(feat_train, target_train)

            pred_flat = model.predict(feat_val)
            pred_norm = pred_flat.reshape(-1, 10, n_channels)
            pred = pred_norm * std_val + mean_val

            mse = ((pred - target_val_orig) ** 2).mean()
            ridge_results[alpha].append(mse)

        # AR(1) with different decays
        for decay in decays:
            pred = ar1_predict(X_val, decay=decay)
            mse = ((pred - target_val_orig) ** 2).mean()
            ar1_results[decay].append(mse)

    print("\nRidge results:")
    best_ridge_alpha = None
    best_ridge_mse = float('inf')
    for alpha in alphas:
        mse = np.mean(ridge_results[alpha])
        std = np.std(ridge_results[alpha])
        marker = ""
        if mse < best_ridge_mse:
            best_ridge_mse = mse
            best_ridge_alpha = alpha
            marker = " <-- best"
        print(f"  alpha={alpha:>5}: MSE={mse:>10.2f} ± {std:>8.2f}{marker}")

    print("\nAR(1) results:")
    best_ar1_decay = None
    best_ar1_mse = float('inf')
    for decay in decays:
        mse = np.mean(ar1_results[decay])
        std = np.std(ar1_results[decay])
        marker = ""
        if mse < best_ar1_mse:
            best_ar1_mse = mse
            best_ar1_decay = decay
            marker = " <-- best"
        print(f"  decay={decay:.2f}: MSE={mse:>10.2f} ± {std:>8.2f}{marker}")

    print(f"\nBest Ridge: alpha={best_ridge_alpha}, MSE={best_ridge_mse:.2f}")
    print(f"Best AR(1): decay={best_ar1_decay}, MSE={best_ar1_mse:.2f}")

    if best_ar1_mse < best_ridge_mse:
        print(f"\n>>> AR(1) is better by {((best_ridge_mse - best_ar1_mse) / best_ridge_mse * 100):.1f}%")
        return "ar1", best_ar1_decay, best_ar1_mse
    else:
        print(f"\n>>> Ridge is better by {((best_ar1_mse - best_ridge_mse) / best_ar1_mse * 100):.1f}%")
        return "ridge", best_ridge_alpha, best_ridge_mse


def cross_validate_affi_detailed():
    """Detailed CV for affi."""
    print("\n" + "="*60)
    print("Detailed affi tuning")
    print("="*60)

    X_all, datasets = load_data("affi")
    n_channels = X_all.shape[2]

    alphas = [100, 500, 1000, 2000, 5000]
    decays = [0.6, 0.7, 0.8, 0.9]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_results = {alpha: [] for alpha in alphas}
    ar1_results = {decay: [] for decay in decays}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        X_train, X_val = X_all[train_idx], X_all[val_idx]

        feat_train, _, _ = prepare_features(X_train)
        target_train, _, _ = prepare_target(X_train)
        feat_val, mean_val, std_val = prepare_features(X_val)

        target_val_orig = X_val[:, 10:, :, 0]

        for alpha in alphas:
            model = Ridge(alpha=alpha, solver='auto')
            model.fit(feat_train, target_train)

            pred_flat = model.predict(feat_val)
            pred_norm = pred_flat.reshape(-1, 10, n_channels)
            pred = pred_norm * std_val + mean_val

            mse = ((pred - target_val_orig) ** 2).mean()
            ridge_results[alpha].append(mse)

        for decay in decays:
            pred = ar1_predict(X_val, decay=decay)
            mse = ((pred - target_val_orig) ** 2).mean()
            ar1_results[decay].append(mse)

    print("\nRidge results:")
    best_ridge_alpha = None
    best_ridge_mse = float('inf')
    for alpha in alphas:
        mse = np.mean(ridge_results[alpha])
        std = np.std(ridge_results[alpha])
        marker = ""
        if mse < best_ridge_mse:
            best_ridge_mse = mse
            best_ridge_alpha = alpha
            marker = " <-- best"
        print(f"  alpha={alpha:>5}: MSE={mse:>10.2f} ± {std:>8.2f}{marker}")

    print("\nAR(1) results:")
    best_ar1_decay = None
    best_ar1_mse = float('inf')
    for decay in decays:
        mse = np.mean(ar1_results[decay])
        std = np.std(ar1_results[decay])
        marker = ""
        if mse < best_ar1_mse:
            best_ar1_mse = mse
            best_ar1_decay = decay
            marker = " <-- best"
        print(f"  decay={decay:.2f}: MSE={mse:>10.2f} ± {std:>8.2f}{marker}")

    print(f"\nBest Ridge: alpha={best_ridge_alpha}, MSE={best_ridge_mse:.2f}")
    print(f"Best AR(1): decay={best_ar1_decay}, MSE={best_ar1_mse:.2f}")

    if best_ar1_mse < best_ridge_mse:
        print(f"\n>>> AR(1) is better by {((best_ridge_mse - best_ar1_mse) / best_ridge_mse * 100):.1f}%")
        return "ar1", best_ar1_decay, best_ar1_mse
    else:
        print(f"\n>>> Ridge is better by {((best_ar1_mse - best_ridge_mse) / best_ar1_mse * 100):.1f}%")
        return "ridge", best_ridge_alpha, best_ridge_mse


def main():
    # Tune both monkeys
    affi_result = cross_validate_affi_detailed()
    beignet_result = cross_validate_beignet_detailed()

    print("\n" + "="*60)
    print("FINAL CONFIGURATION")
    print("="*60)

    configs = {
        'affi': affi_result,
        'beignet': beignet_result,
    }

    for monkey, (method, param, mse) in configs.items():
        print(f"\n{monkey}:")
        print(f"  Method: {method}")
        print(f"  Parameter: {param}")
        print(f"  CV MSE: {mse:.2f}")

    # Estimate total MSR
    total_msr = np.mean([configs['affi'][2], configs['beignet'][2]])
    print(f"\nEstimated Total MSR: {total_msr:.2f}")
    print(f"Top team: 40857.78")
    print(f"vs Top team: {((total_msr - 40857.78) / 40857.78 * 100):+.1f}%")

    # Save final models
    print("\n" + "="*60)
    print("Training and saving final models")
    print("="*60)

    for monkey in ['affi', 'beignet']:
        method, param, mse = configs[monkey]
        X_all, _ = load_data(monkey)

        if method == "ridge":
            features, _, _ = prepare_features(X_all)
            target, _, _ = prepare_target(X_all)

            model = Ridge(alpha=param, solver='auto')
            model.fit(features, target)

            model_path = os.path.join(OUTPUT_DIR, f"ridge_{monkey}_final.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'method': 'ridge',
                    'model': model,
                    'n_channels': X_all.shape[2],
                    'alpha': param,
                }, f)
        else:
            model_path = os.path.join(OUTPUT_DIR, f"ridge_{monkey}_final.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'method': 'ar1',
                    'decay': param,
                    'n_channels': X_all.shape[2],
                }, f)

        print(f"  {monkey}: saved to {model_path}")


if __name__ == "__main__":
    main()
