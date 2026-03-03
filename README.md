# MindReader: EEG Motor Imagery Classification for Brain-Computer Interfaces

A comprehensive benchmark comparing neural network architectures for classifying imagined movements from EEG signals. This project investigates the critical challenge of **cross-subject generalization** in brain-computer interfaces.

## 🧠 Overview

Can we build AI models that decode brain signals across different people? We compare 6 architectures—from compact CNNs to Transformers—on their ability to classify 4 motor imagery tasks (left hand, right hand, both feet, tongue) from multi-channel EEG data.

**Key Research Questions:**
- Which architectures best handle cross-subject generalization?
- Do Transformers learn meaningful spatial attention over motor cortex electrodes?
- How do different frequency bands (mu, beta) affect each architecture?
- Can data augmentation overcome limited training data?

## 📊 Dataset

**Primary:** BCI Competition IV Dataset 2a
- 9 subjects × 288 trials × 4 classes
- 22 EEG channels + 3 EOG channels @ 250 Hz
- 4-second trial windows

**Secondary:** PhysioNet EEG Motor Movement/Imagery Dataset (validation)

## 🏗️ Architectures

| Model | Owner | Parameters | Key Feature |
|-------|-------|------------|-------------|
| **EEGNet** | Person D | ~2K | Compact CNN baseline |
| **LSTM** | Person B | TBD | Temporal modeling |
| **CNN+LSTM** | Person A | TBD | Hybrid spatial-temporal |
| **CNN+GRU** | Person C | TBD | Alternative recurrent hybrid |
| **TCN** | Person A | TBD | Dilated convolutions |
| **Transformer** | Person B | TBD | Spatial-temporal attention |

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/mindreader-eeg-classification.git
cd mindreader-eeg-classification

# Install dependencies
pip install -r requirements.txt

# Download BCI Competition IV Dataset 2a
# Place in data/raw/

# Preprocess data
python src/preprocessing.py

# Train a model (example: EEGNet)
python src/train.py --model eegnet --subject 1 --eval_mode subject_dependent

# Evaluate with LOSO
python src/evaluate.py --model eegnet --eval_mode loso
```

## 📁 Repository Structure
```
├── data/                   # Datasets (gitignored)
├── src/
│   ├── models/            # Architecture implementations
│   ├── preprocessing.py   # Data pipeline
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation framework
├── experiments/
│   ├── configs/          # Experiment configurations
│   └── results/          # Experimental results
├── figures/              # Generated figures for paper
└── paper/                # Research paper LaTeX files
```

## 📚 References

- BCI Competition IV Dataset 2a: [Link](http://www.bbci.de/competition/iv/)
- EEGNet Paper: Lawhern et al. (2018)
- PhysioNet Dataset: Schalk et al. (2004)

## 📄 License

[Choose appropriate license - MIT, Apache 2.0, etc.]

---

**Status:** 🚧 Active Development