# 🚗 CANomaly-LSTM
### LSTM Autoencoder–Based Anomaly Detection for automotive CAN-Bus

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)]()

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca972794-0f3c-4624-b53e-7d77cb0cbfbe" alt="Confusion Matrix" width="600" />
</p>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Data Generation](#1-data-generation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation](#3-evaluation)
- [Results](#-results)
- [Contact](#-contact)

---

## ✨ Overview

**CANomaly-LSTM** is a specialized, end-to-end anomaly detection pipeline designed for **Controller Area Network (CAN)** security. As modern vehicles become increasingly connected, they face growing threats from cyberattacks. This project provides a robust solution using Deep Learning to identify malicious activities on the CAN bus.

The system utilizes an **LSTM (Long Short-Term Memory) Autoencoder** architecture to learn the temporal patterns of normal CAN traffic. By analyzing reconstruction errors, it can effectively detect anomalies such as spoofing, replay attacks, and DoS attempts without requiring labeled attack data for training.

### Key Capabilities
- **Synthetic Traffic Generation**: Create realistic CAN data with customizable normal patterns and attack scenarios.
- **Unsupervised Learning**: Trains only on normal data, making it capable of detecting zero-day attacks.
- **Automated Thresholding**: Dynamically selects the optimal reconstruction error threshold to maximize F1-score.

---

## 💡 Features

### 🛡️ Comprehensive Attack Simulation
The built-in generator supports 4 distinct attack types to test system robustness:
- **Spoofing**: Injecting fake messages with legitimate IDs.
- **Replay**: Re-transmitting valid captured messages to deceive ECUs.
- **Unauthorized ID**: Broadcasting messages with IDs not defined in the system DBC.
- **Payload Corruption**: Randomizing data bytes to simulate fuzzing or sensor malfunctions.

### 🧠 Advanced Model Architecture
- **Input Features**: One-hot encoded CAN IDs + Normalized Payload (8 bytes) + Inter-Arrival Time (IAT).
- **Sliding Window**: Processes data in sequences (window size: 50, stride: 5) to capture temporal context.
- **Autoencoder**: Compresses input into a latent representation and reconstructs it; high error indicates anomaly.

---

## 📁 Project Structure

```bash
CANomaly-LSTM/
├── data/                    # Data storage
│   ├── can_data.csv         # Generated synthetic dataset
│   └── recon_errors.csv     # Model outputs (errors & labels)
│
├── outputs/                 # Results & Visualizations
│   ├── confusion_matrix.png # Visual performance metric
│   └── confusion_report.txt # Detailed classification metrics
│
├── src/                     # Source Code
│   ├── generate_can_dataset.py  # Data generation with attack injection
│   ├── train_lstm_ae.py         # LSTM Autoencoder training loop
│   └── plot_confusion.py        # Evaluation & plotting scripts
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt         # Python dependencies
└── SECURITY.md
```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yigtwxx/CANomaly-LSTM.git
   cd CANomaly-LSTM
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Follow these steps to run the complete pipeline:

### 1. Data Generation
Generate a new synthetic dataset containing both normal traffic and injected attacks.
```bash
python src/generate_can_dataset.py
```
*Output: `data/can_data.csv`*

### 2. Model Training
Train the LSTM Autoencoder on the normal subset of the data.
```bash
python src/train_lstm_ae.py
```
*Output: Trained model (in memory) & Reconstruction errors saved to `data/recon_errors.csv`*

### 3. Evaluation
Calculate metrics, find the optimal threshold, and generate the confusion matrix.
```bash
python src/plot_confusion.py
```
*Output: `outputs/confusion_matrix.png` & `outputs/confusion_report.txt`*

---

## 📊 Results

The model achieves high performance in distinguishing between normal operation and various attack vectors.

### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/ca972794-0f3c-4624-b53e-7d77cb0cbfbe)

### Performance Metrics
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **98.91%** | Overall correctness of the model. |
| **Precision** | **95.52%** | High reliability in anomaly alerts (low false positives). |
| **Recall** | **60.95%** | Ability to detect the majority of attack instances. |
| **F1-Score** | **0.7442** | Balanced harmonic mean of Precision and Recall. |

*(Metrics based on the optimal threshold of 0.665126)*

---

## 💬 Contact

**Yiğit Erdoğan**  
- 📧 Email: [yigiterdogan6@icloud.com](mailto:yigiterdogan6@icloud.com)
- 🌐 GitHub: [@Yigtwxx](https://github.com/Yigtwxx)

<br>

> **Note**: This project is for educational and research purposes. Always test security tools in controlled environments.
