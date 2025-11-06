# 🚗 CANomaly-LSTM  
### LSTM Autoencoder–Based Anomaly Detection for CAN-Bus Traffic

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Stars](https://img.shields.io/github/stars/Yigtwxx/CANomaly-LSTM)
<br><br>
<img width="500" height="656" alt="image" src="https://github.com/user-attachments/assets/ca972794-0f3c-4624-b53e-7d77cb0cbfbe" />


<br>

**Confusion Matrix — Threshold: 0.665126**  
📄 Metrics summary can be found in `outputs/confusion_report.txt`

</div>

---

## ✅ Overview

CANomaly-LSTM is a compact, end-to-end anomaly detection pipeline for **automotive CAN-Bus networks**.  
It generates synthetic CAN traffic, injects realistic attack patterns, trains an LSTM Autoencoder on normal sequences, and detects anomalies using **reconstruction error + optimized thresholding**.

Bu proje; CAN güvenliği, zaman serisi analizi ve derin öğrenmeyi basit bir yapıda birleştirir.

---

## ✅ Features

- Synthetic CAN dataset generation (timestamped frames, payload bytes, labels)  
- 4 attack types:
  - Spoofing  
  - Replay  
  - Unauthorized ID  
  - Payload Corruption  
- One-hot CAN ID encoding  
- Payload (b0…b7) + Inter-Arrival Time (IAT)  
- Sliding windows (size 50, stride 5)  
- LSTM Autoencoder (Encoder → Latent → Decoder)  
- Automatic threshold selection (best F1-score)  
- Confusion matrix, classification report, and error CSV export  

---
```bash 
## ✅ Project Structure
CANomaly-LSTM/
├── src/
│   ├── generate_can_dataset.py      # Synthetic CAN data + attack injection
│   ├── train_lstm_ae.py             # LSTM Autoencoder training pipeline
│   └── plot_confusion.py            # Evaluation + confusion matrix generation
│
├── data/
│   ├── can_data.csv                 # Generated CAN-Bus dataset
│   └── recon_errors.csv             # Reconstruction errors + window labels
│
├── outputs/
│   ├── confusion_matrix.png         # Seaborn heatmap
│   └── confusion_report.txt         # Precision/Recall/F1 metrics
│
├── requirements.txt
├── LICENSE
└── README.md
``` 
---

## ✅ How It Works (Short)

1. **Dataset Generator**  
   Creates 20k+ normal frames and injects 4 types of anomalies  
   → stored in `data/can_data.csv`  
   (script: `generate_can_dataset.py`)

2. **Training (LSTM Autoencoder)**  
   Model learns **normal-only** sequences  
   → reconstruction error = anomaly score  
   → results saved to `data/recon_errors.csv`  
   (script: `train_lstm_ae.py`)

3. **Evaluation**  
   - Finds best F1 threshold automatically  
   - Generates confusion matrix  
   - Saves metrics report  
   (script: `plot_confusion.py`)

---

## ✅ Installation

```bash
pip install -r requirements.txt

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib==3.8.3
seaborn==0.13.2
torch==2.2.0

✅ Usage
1. Generate Dataset
python src/generate_can_dataset.py

2. Train the Autoencoder
python src/train_lstm_ae.py

3. Create Confusion Matrix & Report
python src/plot_confusion.py

Classification Report Summary
(Generated automatically in outputs/confusion_report.txt)

Accuracy: 0.9891
Precision (Anomaly): 0.9552
Recall (Anomaly): 0.6095
F1-Score: 0.7442

```
✅ Contact

📧 Email: yigiterdogan6@icloud.com

🌐 GitHub: https://github.com/Yigtwxx


🧠 Focus Areas: Deep Learning • Computer Vision • Data Science
⭐ If you find this project useful, feel free to star the repository.

 ```
