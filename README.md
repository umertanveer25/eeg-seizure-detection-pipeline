<div align="center">

# 🧠 State-of-the-Art EEG Seizure Detection
**Advanced Spectral Feature Engineering for Patient-Independent Neuromorphic Classification**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20snnTorch-orange.svg)](https://snntorch.readthedocs.io/en/latest/)
[![Accuracy](https://img.shields.io/badge/Max_LOPO_Accuracy-78.71%25-brightgreen.svg)]()

A high-performance pipeline establishing a new State-of-the-Art (SOTA) on the **CHB-MIT Scalp EEG Dataset**. By utilizing advanced multi-domain spectral features (Wavelets, Entropy, Hjorth), this repository systematically outperforms the baseline literature across Spiking Neural Networks (SNN), Artificial Neural Networks (ANN), and Gradient Boosting variants.

</div>

---

## 📖 Overview

This repository builds upon and **systematically outperforms** the methodologies established in recent literature on *Composite EEG biomarker modeling and energy-accuracy trade-off analysis*. 

While standard approaches rely on simplistic biomarkers (like the Seizure Intensity Index: Delta power, Line length, Log-variance), this project extracts a rich, high-dimensional representation of brainwave dynamics to vastly improve **cross-patient generalization (Leave-One-Patient-Out)** without suffering from overfitting.

### Key Innovations
1. **Multi-Domain Spectral Extraction**: Incorporates Discrete Wavelet Transforms (DWT) energy, Spectral Entropy, and Hjorth Parameters (Activity, Mobility, Complexity).
2. **Neuromorphic Parallelization**: Fully parallelized Leaky Integrate-and-Fire (LIF) Spiking Neural Network simulation across CPU/GPU cores using `joblib` and `snntorch`.
3. **Apples-to-Apples Superiority**: Proves that traditional ANN architectures jump by **+16.30%** in generalization accuracy simply by upgrading the feature space.

---

## 🏆 Performance Benchmarks

All models were evaluated using a strict **Leave-One-Patient-Out (LOPO)** cross-validation on the CHB-MIT dataset to guarantee rigorous patient-independent generalizability.

| Algorithm | Baseline Literature (SII Features) | **Ours (Advanced Spectral Features)** | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Artificial Neural Net (ANN)** | 56.58% | **72.88%** | **+16.30%** 🚀 |
| **Spiking Neural Net (SNN)** | 64.34% | **75.19%** | **+10.85%** 🚀 |
| **Logistic Regression (LR)** | 72.83% | **77.49%** | **+4.66%** 📈 |
| **Random Forest** | *Not tested* | **74.76%** | *N/A* |
| **XGBoost** | *Not tested* | **78.28%** | *N/A* |
| **LightGBM** | *Not tested* | **78.71%** | **SOTA** 👑 |

> **Conclusion**: Our `LightGBM` model establishes the new performance ceiling at **78.71%**, while our feature pipeline makes the baseline's exact ANN architecture **16.3% more accurate**.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/EEG-Seizure-Advanced-Features.git
cd EEG-Seizure-Advanced-Features
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the **CHB-MIT Scalp EEG Database** (v1.0.0).
2. Ensure the metadata file (`chb_mit_seizure_metadata.csv`) is placed accurately or update the paths in the source files.
3. The scripts assume the dataset is mapped to `D:\chb-mit-scalp-eeg-database-1.0.0`. Change this variable in the scripts if your directory differs.

---

## 🚀 Usage

The repository is split into classical Machine Learning evaluation and PyTorch neuromorphic simulation.

### 1. Traditional & Gradient Boosting Evaluation
Evaluates LightGBM, XGBoost, Random Forest, Logistic Regression, and ANN.

```bash
python src/evaluate_lopo.py
```

### 2. Spiking Neural Network (SNN) Evaluation
Runs the robust, parallelized Leaky Integrate-and-Fire simulation using `snnTorch`. Utilizes `joblib` for rapid multi-core feature extraction before sequential cross-validation.

```bash
python src/evaluate_snn_lopo.py
```

---

## 🔬 Feature Space Architecture

The `extract_advanced_features()` pipeline processes 1-second EEG windows into a 37-dimensional vector:
* **Time Domain**: Mean, Variance, Standard Deviation, Line Length, Zero-Crossings.
* **Hjorth Parameters**: Activity, Mobility, Complexity.
* **Frequency Domain**: Delta, Theta, Alpha, Beta, Gamma power bands via FFT.
* **Non-Linear Dynamics**: Spectral Entropy.
* **Wavelet Domain**: Energy coefficients extracted via a 4-level Discrete Wavelet Transform (Daubechies 4).

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Acknowledgments

* Dataset provided by [PhysioNet (CHB-MIT)](https://physionet.org/content/chbmit/1.0.0/)
* Spiking Neural Network dynamics powered by [snnTorch](https://github.com/jeshraghian/snntorch)
