"""
Local evaluation script for Neural Forecasting Model.

Simulates the Codabench evaluation pipeline locally.
"""

import sys
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def calculate_mse(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Calculate MSE between two arrays (only on timesteps 10:20).

    This matches the scoring.py evaluation logic.
    """
    if array1.shape != array2.shape:
        raise ValueError(f"Shapes don't match: {array1.shape} vs {array2.shape}")

    # Only evaluate on future timesteps (10:20)
    array1 = array1[:, 10:]
    array2 = array2[:, 10:]

    mse = np.mean((array1 - array2) ** 2)
    return float(mse)


def load_test_data(test_dir: Path, train_dir: Path):
    """
    Load test data and ground truth.

    For local evaluation, we use the unmasked training data as ground truth.
    Note: The actual test data on Codabench has the last 10 timesteps masked.
    """
    datasets = {}

    # For local testing, we'll use the training data
    # split into test-like format
    files = {
        'affi': [
            ('affi_train', 'train_data_affi.npz'),
        ],
        'beignet': [
            ('beignet_train', 'train_data_beignet.npz'),
        ],
    }

    # Also check for private test files if available
    test_files = {
        'affi': 'test_data_affi_masked.npz',
        'beignet': 'test_data_beignet_masked.npz',
    }

    for monkey, file_list in files.items():
        for name, filename in file_list:
            filepath = train_dir / filename
            if filepath.exists():
                data = np.load(filepath)['arr_0']
                datasets[name] = data
                print(f"Loaded {name}: shape = {data.shape}")

    return datasets


def evaluate_model(
    submission_dir: Path,
    data_dir: Path,
    output_dir: Path,
):
    """
    Run local evaluation.

    Args:
        submission_dir: Path to submission directory containing model.py
        data_dir: Path to dataset directory
        output_dir: Path to save evaluation results
    """
    print("=" * 60)
    print("Local Evaluation for Neural Forecasting")
    print("=" * 60)

    # Add submission directory to path
    sys.path.insert(0, str(submission_dir))

    # Import the Model class
    from model import Model

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Load datasets for evaluation
    # Using training data for local validation (split off some samples)
    datasets = load_test_data(test_dir, train_dir)

    results = {}

    for monkey in ['affi', 'beignet']:
        print(f"\n{'='*40}")
        print(f"Evaluating {monkey}")
        print(f"{'='*40}")

        # Load model
        model = Model(monkey)
        model.load()

        # Get test data
        key = f'{monkey}_train'
        if key not in datasets:
            print(f"No test data for {monkey}")
            continue

        # Use last 20% of training data for local evaluation
        full_data = datasets[key]
        n_test = max(10, int(len(full_data) * 0.2))
        test_data = full_data[-n_test:]

        print(f"Test samples: {len(test_data)}")

        # Create masked version (like Codabench)
        masked_data = test_data.copy()
        # Mask last 10 timesteps by copying timestep 9
        masked_data[:, 10:, :, :] = masked_data[:, 9:10, :, :]

        # Ground truth (only feature 0)
        ground_truth = test_data[:, :, :, 0]  # (N, 20, C)

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(masked_data)

        # Calculate MSE
        mse = calculate_mse(ground_truth, predictions)

        print(f"MSE for {monkey}: {mse:.6f}")
        results[f'MSE_{monkey}'] = mse

    # Calculate total MSE
    if len(results) > 0:
        total_mse = sum(results.values()) / len(results)
        results['total_MSE'] = total_mse
        print(f"\n{'='*40}")
        print(f"Total MSE: {total_mse:.6f}")
        print(f"{'='*40}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "scores.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def evaluate_with_test_data(
    submission_dir: Path,
    data_dir: Path,
    output_dir: Path,
):
    """
    Evaluate using actual test data (for final submission validation).
    """
    print("=" * 60)
    print("Evaluation with Test Data")
    print("=" * 60)

    sys.path.insert(0, str(submission_dir))
    from model import Model

    test_dir = data_dir / "test"
    train_dir = data_dir / "train"

    # Test files mapping
    test_files = {
        'affi': {
            'test': 'test_data_affi_masked.npz',
            'ref': 'train_data_affi.npz',  # Use training data as reference
        },
        'affi_private': {
            'test': 'test_data_affi_2024-03-20_private_masked.npz',
            'ref': 'train_data_affi_2024-03-20_private.npz',
        },
        'beignet': {
            'test': 'test_data_beignet_masked.npz',
            'ref': 'train_data_beignet.npz',
        },
        'beignet_private_1': {
            'test': 'test_data_beignet_2022-06-01_private_masked.npz',
            'ref': 'train_data_beignet_2022-06-01_private.npz',
        },
        'beignet_private_2': {
            'test': 'test_data_beignet_2022-06-02_private_masked.npz',
            'ref': 'train_data_beignet_2022-06-02_private.npz',
        },
    }

    results = {}

    for dataset_name, files in test_files.items():
        monkey = 'affi' if 'affi' in dataset_name else 'beignet'

        test_path = test_dir / files['test']
        ref_path = train_dir / files['ref']

        if not test_path.exists():
            print(f"Test file not found: {test_path}")
            continue

        # Note: We can't truly evaluate without ground truth
        # This just runs predictions on test data
        print(f"\nProcessing {dataset_name}...")

        model = Model(monkey)
        model.load()

        test_data = np.load(test_path)['arr_0']
        print(f"Test data shape: {test_data.shape}")

        predictions = model.predict(test_data)
        print(f"Predictions shape: {predictions.shape}")

        # Save predictions
        pred_path = output_dir / f"test_{dataset_name}.predictions.npz"
        np.savez(pred_path, predictions)
        print(f"Saved predictions to {pred_path}")

        # If we have reference data (for local validation only)
        if ref_path.exists():
            ref_data = np.load(ref_path)['arr_0']

            # Can only compare if shapes match
            if ref_data.shape[0] == test_data.shape[0]:
                ground_truth = ref_data[:, :, :, 0]
                mse = calculate_mse(ground_truth, predictions)
                results[f'MSE_{dataset_name}'] = mse
                print(f"MSE for {dataset_name}: {mse:.6f}")

    if results:
        total_mse = sum(results.values()) / len(results)
        results['total_MSE'] = total_mse
        print(f"\nTotal MSE: {total_mse:.6f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    if results:
        results_path = output_dir / "scores.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Local Evaluation")
    parser.add_argument("--submission_dir", type=str, default="./submission",
                       help="Path to submission directory")
    parser.add_argument("--data_dir", type=str, default="../dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                       help="Path to save evaluation results")
    parser.add_argument("--use_test", action="store_true",
                       help="Use actual test data (for final validation)")

    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.use_test:
        evaluate_with_test_data(submission_dir, data_dir, output_dir)
    else:
        evaluate_model(submission_dir, data_dir, output_dir)


if __name__ == "__main__":
    main()
