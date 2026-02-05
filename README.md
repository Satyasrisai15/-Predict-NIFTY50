# NIFTY50 Stock Predictor

A professional-grade Hybrid AI framework (CNN-LSTM-Transformer) designed to predict 3-month, 1-year, and 3-year returns for all NIFTY 50 stocks.

## 📊 Project Overview

This project implements a sophisticated deep learning architecture to analyze historical stock data and forecast future trends. It is structured as a production-ready Python package.

**Key Features:**
*   **Hybrid Architecture**: Combines CNN (feature extraction), Bi-LSTM (temporal sequence learning), and Transformers (long-range dependencies).
*   **Multi-Horizon Prediction**: Simultaneously predicts returns for Short (3M), Medium (1Y), and Long-term (3Y).
*   **Comparison Framework**: Includes tools to benchmark predictions directly against actual market performance (simulated or backtested).

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   Virtual Environment (recommended)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/Nifty50-predictor.git
    cd Nifty50-predictor
    ```

2.  **Install dependencies**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Initialize Data**
    The raw data and models are excluded from the repository to keep it lightweight. You can reproduce them easily:
    ```bash
    # Download latest data from Yahoo Finance
    python -m src.data.download

    # Process features (Engineering)
    python -m src.data.feature_engineering
    ```

## 🧠 Training & inference

### Training Models
To train models for all 50 stocks (or specific ones), use the training script. 
*Note: Pre-trained models are ignored by git to save bandwidth (1.3GB+). Train locally or use the Colab notebook.*

```bash
# Train a specific stock locally for testing
python scripts/train_local.py
```

### Generating Predictions
Once models are trained (or if you have downloaded pre-trained weights), generate the full inference report:

```bash
python scripts/predict_all.py
```
Results will be saved to `evaluation_results/final_report_YYYYMMDD.csv`.

## 📁 Project Structure

```text
Nifty50-predictor/
├── configs/            # Configuration files
├── docs/               # Documentation & design docs
├── evaluation_results/ # Generated prediction CSVs (Proof of Performance)
├── notebooks/          # Experimental notebooks (Colab/Jupyter)
├── scripts/            # Executable entry points (train, predict)
├── src/                # Core Source Code
│   ├── data/           # Data loading & processing
│   ├── models/         # PyTorch Model Definitions
│   ├── training/       # Training loops
│   └── utils/          # Visualization & metrics
└── requirements.txt    # Dependencies
```

## 📈 Proof of Performance

While the binary model files (`.pt`) are not included in this repo due to size constraints (>1GB), the **Evaluation Results** are committed to verify the model's performance.

*   **Latest Report**: `evaluation_results/final_report_YYYYMMDD.csv`
*   **Visualizations**: See `plots/` for HTML interactive graphs (if generated).

## 🛠 Technology Stack
*   **Core**: PyTorch, NumPy, Pandas
*   **Data**: Yahoo Finance (`yfinance`)
*   **Visualization**: Plotly
