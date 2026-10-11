# RHYME: Curved Worlds, Clear Boundaries - Generalizing Speech Deepfake Detection

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-IJCNLP%202025-blue)](https://aclanthology.org/2025.ijcnlp-long.104/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A generalized speech deepfake detection framework using hyperbolic and spherical geometry spaces

</div>

## Overview

**RHYME** is a state-of-the-art framework for detecting synthetic audio and deepfakes across diverse speech generation methods. The key innovation is leveraging **non-Euclidean geometric projections** (hyperbolic and spherical spaces) to create synthesis-invariant embeddings that generalize across:

- Traditional TTS systems (Tacotron, Glow-TTS, FastSpeech)
- Modern neural vocoding systems (WaveGlow, HiFi-GAN)
- Advanced generative models (diffusion, flow-matching)

### Key Insight

Unlike prior methods that overfit to specific synthesis artifacts, RHYME uses:
- **Hyperbolic geometry** to capture hierarchical relationships among generator types
- **Spherical geometry** to identify vocoder-specific patterns
- **Riemannian averaging** to compute synthesis-invariant representations

## Paper

**Curved Worlds, Clear Boundaries: Generalizing Speech Deepfake Detection using Hyperbolic and Spherical Geometry Spaces**

- **Authors**: Farhan Sheth, Girish, Mohd Mujtaba Akhtar, Muskaan Singh
- **Conference**: IJCNLP-AACL 2025
- **Paper Link**: [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.104/)

## Key Features

✨ **Cross-Paradigm Generalization**: Detects deepfakes across TTS, vocoding, and generative models

🌍 **Non-Euclidean Geometry**: Leverages hyperbolic and spherical spaces for better embeddings

📊 **Multi-Encoder Fusion**: Fuses representations from multiple pretrained speech encoders

🎯 **Synthesis-Invariant**: Creates embeddings that are invariant to generation methods

⚡ **Riemannian Geometry**: Uses geodesic distances and Riemannian mean for robust averaging

🔊 **State-of-the-Art**: Achieves SOTA results where previous approaches fail

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/farhan-sheth/RHYME-IJCNLP-AACL.git
cd RHYME-IJCNLP-AACL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from rhyme.model import RHYME
from rhyme.utils import load_audio

# Load the pretrained model
model = RHYME.from_pretrained('path/to/checkpoint')

# Process audio
audio_path = "path/to/audio.wav"
audio = load_audio(audio_path)

# Get prediction
prediction = model(audio)
print(f"Is Deepfake: {prediction['is_deepfake']}")
print(f"Confidence: {prediction['confidence']:.4f}")
print(f"Generator Type: {prediction['generator_type']}")
```

## Repository Structure

```
RHYME-IJCNLP-AACL/
├── rhyme/                      # Main package
│   ├── __init__.py
│   ├── model.py               # Core RHYME model
│   ├── geometry.py            # Hyperbolic/Spherical projections
│   ├── encoders.py            # Multi-encoder fusion
│   ├── utils.py               # Utility functions
│   └── metrics.py             # Evaluation metrics
├── data/                       # Dataset scripts and utilities
│   ├── __init__.py
│   ├── loaders.py             # Data loading utilities
│   ├── preprocessor.py        # Audio preprocessing
│   └── datasets.py            # Dataset classes
├── models/                     # Pretrained model checkpoints
│   └── rhyme_checkpoint.pt
├── configs/                    # Configuration files
│   ├── config_base.yaml
│   ├── config_tts.yaml
│   ├── config_vocoder.yaml
│   └── config_generative.yaml
├── scripts/                    # Training and evaluation scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Inference script
│   └── benchmark.py          # Cross-paradigm benchmarking
├── results/                    # Results and logs
│   ├── metrics/               # Performance metrics
│   └── logs/                  # Training logs
├── tests/                      # Unit tests
│   ├── test_model.py
│   ├── test_geometry.py
│   └── test_encoders.py
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # Detailed architecture
│   ├── TRAINING.md            # Training guide
│   └── API.md                 # API reference
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Model Architecture

### Non-Euclidean Geometric Projection

RHYME's core innovation uses three complementary geometric spaces:

```
Input Audio
    ↓
┌─────────────────────────────────────┐
│   Multi-Encoder Feature Extraction  │
├─────────────────────────────────────┤
│ WavLM, UniSpeech, HuBERT, XLS-R    │
│ Features: (D1, D2, D3, D4)          │
└──────┬──────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│   Non-Euclidean Projection           │
├──────────────────────────────────────┤
│ Hyperbolic Space  │  Euclidean Space │
│ (Hierarchical)    │  (Linear)        │
│ + Spherical Space (Directional)      │
└──────┬──────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│   Riemannian Averaging               │
│   (Synthesis-Invariant Embedding)    │
└──────┬──────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│   Classification Head                │
│   (Real/Synthetic + Generator Type)  │
└──────────────────────────────────────┘
```

### Geometric Components

#### Hyperbolic Projection
- Models hierarchical relationships among generator types
- Better captures the tree-like structure of synthesis methods
- Uses Poincaré ball model

#### Spherical Projection
- Identifies vocoder-specific patterns
- Captures directional relationships
- Operates on normalized embeddings

#### Riemannian Mean
- Computes geodesic averages of embeddings
- Creates synthesis-invariant representations
- Uses exponential/logarithmic maps

## Results

### Cross-Paradigm Generalization Benchmark

| Evaluation Scenario | Accuracy | F1-Score | Generalization Gap |
|-------------------|----------|----------|-------------------|
| Train/Test Same   | 98.2%    | 98.0%    | -                  |
| Cross-Paradigm    | 94.5%    | 93.8%    | 3.7%               |
| Unknown Generator | 91.3%    | 90.2%    | 6.9%               |

*Results demonstrate strong generalization across synthesis methods*

## Supported Synthesis Methods

### TTS Systems
- Tacotron 2
- Glow-TTS
- FastSpeech 2
- Transformer TTS

### Neural Vocoders
- WaveGlow
- HiFi-GAN
- WaveNet

### Generative Models
- Diffusion-based TTS
- Flow-matching systems
- VAE-based systems

## Getting Started

### Training

```bash
python scripts/train.py --config configs/config_base.yaml
```

For specific synthesis families:
```bash
python scripts/train.py --config configs/config_tts.yaml  # TTS systems
python scripts/train.py --config configs/config_vocoder.yaml  # Vocoders
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/rhyme_checkpoint.pt --test-data path/to/test
```

### Cross-Paradigm Benchmarking

```bash
python scripts/benchmark.py --checkpoint models/rhyme_checkpoint.pt
```

### Inference

```bash
python scripts/inference.py --audio path/to/audio.wav --checkpoint models/rhyme_checkpoint.pt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchaudio 0.9+
- librosa 0.9+
- NumPy, SciPy
- PyYAML
- scikit-learn

See `requirements.txt` for complete dependencies.

## Installation from Source

```bash
git clone https://github.com/farhan-sheth/RHYME-IJCNLP-AACL.git
cd RHYME-IJCNLP-AACL
pip install -e .
```

## Usage Examples

### Example 1: Detect Deepfake in Audio

```python
from rhyme.model import RHYME
from rhyme.utils import load_audio

model = RHYME.from_pretrained('models/rhyme_checkpoint.pt')
model.eval()

audio = load_audio('suspect_audio.wav')
result = model(audio)

if result['is_deepfake']:
    print(f"⚠️ Deepfake detected!")
    print(f"Generator: {result['generator_type']}")
    print(f"Confidence: {result['confidence']:.4f}")
else:
    print(f"✓ Authentic audio")
```

### Example 2: Batch Processing

```python
from rhyme.model import RHYME
from data.loaders import AudioDataLoader

model = RHYME.from_pretrained('models/rhyme_checkpoint.pt')
loader = AudioDataLoader('path/to/audio/directory')

for batch in loader:
    predictions = model(batch['audio'])
    # Process predictions
```

### Example 3: Custom Configuration

```python
from rhyme.model import RHYME
import yaml

with open('configs/custom_config.yaml') as f:
    config = yaml.safe_load(f)

model = RHYME(config)
# Train or use the model
```

## Citation

If you use RHYME in your research, please cite the following paper:

```bibtex
@inproceedings{sheth2025rhyme,
  title={Curved Worlds, Clear Boundaries: Generalizing Speech Deepfake Detection using Hyperbolic and Spherical Geometry Spaces},
  author={Sheth, Farhan and Girish and Akhtar, Mohd Mujtaba and Singh, Muskaan},
  booktitle={Proceedings of IJCNLP-AACL 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)

## Acknowledgments

- The speech deepfake detection research community
- Developers of WavLM, UniSpeech, HuBERT, XLS-R models
- Riemannian geometry implementations
- IJCNLP-AACL 2025 organizers

## Questions & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/farhan-sheth/RHYME-IJCNLP-AACL/issues)
- **Email**: girish281003@gmail.com

## Related Work

- [WavLM: Pre-trained Speech Representations](https://github.com/microsoft/unilm/tree/master/wavlm)
- [Deepfake Detection Surveys](https://github.com/search?q=deepfake+detection+speech)
- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- [Riemannian Geometry in ML](https://arxiv.org/abs/2003.02236)

---

<div align="center">

**Built with ❤️ for detecting synthetic speech across all generation paradigms**

[⬆ back to top](#rhyme-curved-worlds-clear-boundaries---generalizing-speech-deepfake-detection)

</div>
