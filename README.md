# CANomaly-LSTM  
**LSTM Autoencoder–based Anomaly Detection System for Automotive CAN-Bus Networks**

This project implements an end-to-end anomaly detection pipeline for automotive CAN-Bus traffic using an LSTM Autoencoder. The system generates fully synthetic CAN data, injects multiple attack scenarios, extracts sequential features, trains on normal-only traffic, and detects anomalies through reconstruction-error scoring.  
It demonstrates a practical Intrusion Detection System (IDS) design combining time-series modeling, feature engineering, and deep learning.

---

## 🚀 Key Features

- **Fully synthetic CAN-Bus dataset generation**
  - 20,000+ normal messages
  - Realistic timestamp progression and byte-level payload structure
- **Four attack scenarios**
  - **Spoofing** (fake messages with abnormal byte patterns)  
  - **Replay** (high-frequency re-transmission of previous messages)  
  - **Unauthorized ID** (messages from unknown CAN identifiers)  
  - **Payload Corruption** (bit-level inversion & data tampering)
- **Feature engineering**
  - One-hot encoded CAN IDs  
  - 8-byte payload vectors  
  - Inter-Arrival Time (IAT)
- **Sliding-window sequence generation**  
  - Window size: 50  
  - Stride: 5  
- **Deep learning model**
  - LSTM Encoder → Latent Vector → LSTM Decoder  
  - Trained only on **normal** traffic  
  - Reconstruction MSE used as anomaly score
- **Automatic threshold selection**
  - Scans 200 candidate thresholds  
  - Picks the one that maximizes **F1-score**
- **Evaluation & Visualization**
  - Confusion Matrix (Seaborn heatmap)
  - Classification Report (precision, recall, F1)
  - Reconstruction error CSV output

---

## 📁 Project Structure

CANomaly-LSTM/
├── data/
│ ├── can_data.csv # Synthetic CAN-Bus dataset
│ └── recon_errors.csv # Reconstruction errors + window labels
│
├── src/
│ ├── generate_can_dataset.py # Synthetic traffic & attack generator
│ ├── train_lstm_ae.py # LSTM Autoencoder training + error export
│ └── plot_confusion.py # Confusion Matrix + Classification Report
│
├── outputs/
│ ├── confusion_matrix.png # Heatmap visualization
│ └── confusion_report.txt # Detailed model performance report
│
├── requirements.txt
├── LICENSE
└── README.md

---

## ✅ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/CANomaly-LSTM
cd CANomaly-LSTM

pip install -r requirements.txt

📊 Usage
1. Generate Synthetic CAN-Bus Data
python src/generate_can_dataset.py


Outputs:
data/can_data.csv

2. Train the LSTM Autoencoder
python src/train_lstm_ae.py


Outputs:

data/recon_errors.csv

console summary of selected threshold + F1-score

3. Produce Confusion Matrix & Report
python src/plot_confusion.py


Outputs:

outputs/confusion_matrix.png

outputs/confusion_report.txt

📈 Example Results

Confusion Matrix (example):

	      Pred 0	Pred 1
True 0	3931	  3
True 1	41	    64

Accuracy: 98.9%

Normal detection: extremely high (low false positives)

Anomaly recall: moderate, typical for AE-based IDS

Automatic threshold: selects the best value for F1 optimization

The system achieves near-perfect normal traffic reconstruction and detects injected attacks with strong performance.

🧠 Model Architecture
Input Window (50 × F)
     ↓
LSTM Encoder
     ↓
Latent Vector (bottleneck)
     ↓
LSTM Decoder
     ↓
Reconstructed Window
     ↓
MSE Reconstruction Error → Anomaly Score


The Autoencoder is trained only on normal windows, enabling it to detect deviations in unseen attack sequences.

📌 Why This Project Matters

Modern vehicles heavily depend on CAN-Bus, yet it lacks built-in security.

Attackers can inject, replay, or manipulate messages with minimal effort.

Deep-learning-based IDS systems are emerging as the next-generation defense layer.

This project demonstrates a practical, reproducible, fully synthetic yet realistic IDS pipeline suitable for:

Research

Education

Automotive cybersecurity demonstrations

Portfolio / hiring showcase

📜 License

MIT License — free for personal and commercial use.

⭐ Contributing

Pull requests are welcome. Feel free to open issues for feature suggestions or improvements.

💬 Contact

For questions or collaboration: <yigiterdogan6@icloud.com>

If you like the project, consider starring ⭐ the repository!
