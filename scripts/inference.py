"""Inference script for RHYME."""

import argparse
import torch
from pathlib import Path

from rhyme.model import RHYME
from data.preprocessor import AudioPreprocessor


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference with RHYME")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RHYME.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load and preprocess audio
    preprocessor = AudioPreprocessor()
    audio = preprocessor.load_audio(args.audio)

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(audio_tensor)

    # Extract predictions
    deepfake_logits = outputs['deepfake'][0]
    generator_logits = outputs['generator'][0]

    deepfake_pred = deepfake_logits.argmax().item()
    generator_pred = generator_logits.argmax().item()

    deepfake_conf = torch.softmax(deepfake_logits, dim=0)[deepfake_pred].item()

    # Print results
    print(f"\n{'='*60}")
    print(f"Inference Results: {Path(args.audio).name}")
    print(f"{'='*60}")

    if deepfake_pred == 0:
        print(f"Result: ✓ AUTHENTIC AUDIO")
    else:
        print(f"Result: ⚠️  DEEPFAKE DETECTED")

    print(f"Confidence: {deepfake_conf:.4f}")
    print(f"Generator Type: {generator_pred}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
