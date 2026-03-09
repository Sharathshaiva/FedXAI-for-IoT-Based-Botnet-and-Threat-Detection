# FedXAI: Federated Explainable AI for IoT Botnet Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![FL Accuracy](https://img.shields.io/badge/FL_Accuracy-99.94%25-2E7D32?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9999-2E7D32?style=flat-square)
![False Negatives](https://img.shields.io/badge/False_Negatives-0-2E7D32?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

**M.Tech Project · Ramaiah Institute of Technology, Bengaluru**  
*Sharath V · Dept. of CSE(AI&ML) · 2025–2026*

[Overview](#overview) · [Results](#results) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Dashboard](#dashboard) · [Citation](#citation)

</div>

---

## Overview

**FedXAI** is a unified framework for privacy-preserving, interpretable IoT botnet detection combining three complementary technologies:

| Component | Technology | What it solves |
|---|---|---|
| **Federated Learning** | FedAvg (McMahan et al.) | Raw data never leaves IoT devices |
| **Explainable AI** | SHAP + LIME | Every detection is explained |
| **Differential Privacy** | Gaussian Mechanism | Formal, quantifiable privacy guarantee |

Evaluated on the **N-BaIoT** benchmark — real network traffic from **9 heterogeneous IoT devices** infected with **Mirai** and **Gafgyt/BASHLITE** botnets across **10 attack variants** and **115 statistical features**.

---

## Results

### Global Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Centralized MLP | 99.96% | 99.96% | 100.0% | 99.98% | 1.0000 |
| Centralized CNN | 99.98% | 99.98% | 100.0% | 99.99% | 0.9999 |
| **FedXAI (FedAvg)** | **99.94%** | **99.92%** | **100.0%** | **99.96%** | **0.9999** |
| FedXAI + Weak DP (ε≈50) | 99.74% | 99.72% | 99.74% | 99.72% | 0.9977 |
| FedXAI + Mild DP (ε≈10) | 99.13% | 99.10% | 99.15% | 99.10% | 0.9916 |
| FedXAI + Strong DP (ε≈5) | 98.46% | 98.43% | 98.46% | 98.45% | 0.9849 |
| FedXAI + Very Strong (ε≈1) | 92.37% | 92.34% | 92.39% | 92.35% | 0.9239 |

> FedXAI achieves only **0.028% accuracy gap** vs centralized MLP — with **zero false negatives** and raw data fully private.

### Per-Device Accuracy

| Device | Accuracy | AUC-ROC |
|---|---|---|
| Danmini Doorbell | 99.97% | 1.0000 |
| Ecobee Thermostat | 100.00% | 1.0000 |
| Ennio Doorbell | 99.92% | 0.9992 |
| Philips Baby Monitor | 99.90% | 0.9998 |
| Provision PT737E Cam | 99.89% | 1.0000 |
| Provision PT838 Cam | 99.80% | 1.0000 |
| Samsung SNH Webcam | 99.98% | 1.0000 |
| SimpleHome Cam 1002 | 99.90% | 0.9996 |
| SimpleHome Cam 1003 | 99.97% | 0.9998 |

### Top SHAP Features

| Rank | Feature | Mean \|SHAP\| | Physical Meaning |
|---|---|---|---|
| 1 | H_L0.01_weight | 0.0340 | Host traffic weight, 10ms window |
| 2 | MI_dir_L0.01_weight | 0.0336 | Directional traffic weight, 10ms |
| 3 | MI_dir_L0.1_weight | 0.0292 | Directional weight, 100ms window |
| 4 | H_L0.01_variance | 0.0243 | Host traffic variance, 10ms |
| 5 | MI_dir_L0.01_variance | 0.0232 | Directional variance, 10ms |

Botnet malware (Mirai/Gafgyt) generates abnormally high-volume, directional traffic bursts in very short time windows — creating distinctive statistical signatures in the 10ms–100ms features.

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Google Colab (recommended) or local GPU
- Kaggle account (for dataset download)

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/FedXAI.git
cd FedXAI
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Get the Dataset

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d mkashifn/nbaiot-dataset --unzip -p data/nbaiot/
```

Or download manually from [Kaggle N-BaIoT](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset).

### 4. Run Notebooks (in order)

| Step | Notebook | Output |
|---|---|---|
| 1 | `notebooks/Phase1B_Phase2_Baseline.ipynb` | Preprocessed data + centralized models |
| 2 | `notebooks/Phase3_FederatedLearning.ipynb` | `federated_global_model.keras` |
| 3 | `notebooks/Phase4_SHAP_LIME.ipynb` | SHAP values + plots |
| 4 | `notebooks/Phase5_DifferentialPrivacy.ipynb` | DP results + comparison |

### 5. Launch Dashboard

```bash
# Copy model files to dashboard/ first
cp path/to/federated_global_model.keras dashboard/
cp path/to/client_data.pkl dashboard/
cp path/to/feature_names.pkl dashboard/

streamlit run dashboard/app.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     IoT DEVICE LAYER (9 devices)                │
│  Doorbell  Thermostat  Camera × 6  Baby Monitor                 │
│  Each device trains locally on private N-BaIoT traffic          │
└────────────────────────┬────────────────────────────────────────┘
                         │  DP Noise (optional, σ=0.05)
                         │  Clip + N(0, σ²C²I)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEDAVG AGGREGATION SERVER                     │
│  w_global = Σ (nₖ / N) × wₖ    │    20 communication rounds    │
└────────────────────────┬────────────────────────────────────────┘
                         │  Global model broadcast
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   XAI + EVALUATION LAYER                        │
│  SHAP DeepExplainer  │  LIME Tabular  │  Acc / F1 / AUC-ROC    │
│  Per-device heatmap  │  Waterfall     │  DP budget accounting   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
            Detection + Confidence + Feature Attribution
```

### MLP Architecture

```
Input (115 features)
    → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    → Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
    → Dense(32,  ReLU)
    → Dense(1,   Sigmoid)          → P(Botnet Attack)

Parameters : ~108,481
Optimizer  : Adam (lr = 1e-3)
Loss       : Binary Cross-Entropy
Stopping   : Early stopping, patience = 5 on val AUC
```

### FL Configuration

```
Clients          : 9 (one per IoT device)
Rounds           : 20
Local epochs     : 3 per round
Batch size       : 256
Aggregation      : FedAvg (weighted by dataset size)
Fraction fit     : 1.0 (all clients every round)
```

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) has five interactive pages:

| Page | Feature |
|---|---|
| **Overview** | Master comparison table, per-device bar chart, FL architecture |
| **Attack Simulator** | Real `federated_global_model.keras` inference on actual N-BaIoT samples + live SHAP waterfall |
| **FL Monitor** | Accuracy/loss convergence over 20 rounds, per-device breakdown |
| **XAI Explorer** | SHAP global importance, beeswarm, waterfall per sample, per-device heatmap |
| **Privacy Analysis** | DP tradeoff curves, epsilon comparison, Gaussian mechanism explainer |

The **Attack Simulator** uses real model inference — not simulated results — and generates a live SHAP explanation showing which features triggered the alert.

---

## Repository Structure

```
FedXAI/
├── notebooks/
│   ├── Phase1B_Phase2_Baseline.ipynb       # EDA + MLP/CNN centralized baseline
│   ├── Phase3_FederatedLearning.ipynb      # Manual FedAvg, 9 clients, 20 rounds
│   ├── Phase4_SHAP_LIME.ipynb              # SHAP DeepExplainer + LIME Tabular
│   └── Phase5_DifferentialPrivacy.ipynb    # Gaussian DP, 5 epsilon configs
├── dashboard/
│   ├── app.py                              # Full 5-page Streamlit dashboard
│   └── attack_simulator.py                # Standalone real-inference simulator
├── figures/
│   ├── fig1_1_botnet_lifecycle.png
│   ├── fig1_2_centralized_vs_fl.png
│   ├── fig3_1_fedxai_architecture.png
│   ├── fig3_2_fedavg_flowchart.png
│   └── fig3_3_mlp_architecture.png
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `data/` and `models/` are excluded from the repo (too large).  
> Store on Google Drive and mount in Colab, or download from Kaggle directly.

---

## Dataset: N-BaIoT

| Property | Value |
|---|---|
| Devices | 9 real IoT devices |
| Features | 115 statistical network traffic features |
| Attack families | Mirai (5 variants) + Gafgyt/BASHLITE (5 variants) |
| Total variants | 10 attack types |
| Benign samples | ~550,000 |
| Attack samples | ~1,900,000 |
| Source | [Kaggle](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) / [UCI ML Repository](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) |

Feature categories:
- **H** — host-level statistics (packet counts, sizes)
- **MI_dir** — directional statistics (src→dst)
- **HH** — host-to-host statistics
- **HH_jit** — host-to-host jitter (timing irregularity)
- Time windows: **L0.01** (10ms), **L0.1** (100ms), **L1.5** (1.5s), **L10** (10s), **L60** (60s)

---

## Privacy Guarantee

FedXAI implements **(ε, δ)-Differential Privacy** via the Gaussian Mechanism:

```
Formal guarantee:
  Pr[M(D) ∈ S]  ≤  e^ε × Pr[M(D') ∈ S]  +  δ

For any two datasets D, D' differing by one device's data.
```

| Config | Noise σ | Epsilon ε | Accuracy | Recommended? |
|---|---|---|---|---|
| No DP | 0.00 | ∞ | 99.94% | Dev/research |
| Weak DP | 0.01 | ≈50 | 99.74% | Low-sensitivity |
| **Mild DP** | **0.05** | **≈10** | **99.13%** | **Production** |
| Strong DP | 0.10 | ≈5 | 98.46% | High-security |
| Maximum DP | 0.50 | ≈1 | 92.37% | Max privacy |

---

## Environment

Tested on:
- Google Colab (Tesla T4 GPU, 16GB VRAM)
- Python 3.12, TensorFlow 2.19.0, NumPy 1.26.4

---

## Citation

```bibtex
@mastersthesis{sharath2026fedxai,
  title      = {FedXAI: Federated Explainable AI for
                Privacy-Preserving IoT Botnet Detection},
  author     = {Sharath, V},
  school     = {Ramaiah Institute of Technology, Bengaluru},
  year       = {2026},
  type       = {M.Tech Thesis},
  department = {Computer Science and Engineering}
}
```

---

## License

MIT © 2026 Sharath V — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [N-BaIoT Dataset](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) — Yair Meidan et al., Ben-Gurion University
- [SHAP](https://github.com/shap/shap) — Scott Lundberg & Su-In Lee
- [TensorFlow](https://tensorflow.org) — Google Brain Team
- [Streamlit](https://streamlit.io) — Streamlit Inc.
- Supervisor: Prof. [Name], Dept. of CSE, Ramaiah Institute of Technology

---

<div align="center">
Built as part of M.Tech thesis at Ramaiah Institute of Technology, Bengaluru
</div>
