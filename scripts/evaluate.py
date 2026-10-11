"""Evaluation script for RHYME."""

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

from rhyme.model import RHYME
from data.loaders import AudioDataLoader


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_deepfake_preds = []
    all_deepfake_targets = []
    all_generator_preds = []
    all_generator_targets = []

    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)

            outputs = model(audio)

            deepfake_preds = outputs['deepfake'].argmax(dim=1)
            generator_preds = outputs['generator'].argmax(dim=1)

            if 'is_deepfake' in batch:
                all_deepfake_preds.extend(deepfake_preds.cpu().numpy())
                all_deepfake_targets.extend(batch['is_deepfake'].numpy())

            if 'generator' in batch:
                all_generator_preds.extend(generator_preds.cpu().numpy())
                all_generator_targets.extend(batch['generator'].numpy())

    results = {
        'deepfake': {
            'predictions': np.array(all_deepfake_preds),
            'targets': np.array(all_deepfake_targets),
        },
        'generator': {
            'predictions': np.array(all_generator_preds),
            'targets': np.array(all_generator_targets),
        },
    }

    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for task_name, task_results in results.items():
        if len(task_results['predictions']) == 0:
            continue

        print(f"\n{task_name.upper()}")
        print("-"*80)

        report = classification_report(
            task_results['targets'],
            task_results['predictions'],
            digits=4,
        )
        print(report)

    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RHYME model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test data directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}")
    model = RHYME.from_pretrained(args.checkpoint)
    model = model.to(device)

    print(f"Loading test data from {args.test_dir}")
    test_loader = AudioDataLoader(
        args.test_dir,
        batch_size=32,
        shuffle=False,
    )

    print("Evaluating...")
    results = evaluate(model, test_loader, device)
    print_results(results)


if __name__ == "__main__":
    main()
