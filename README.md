# Thesis-STONK: STONK: Stock Trend Optimization using News Knowledge

This repository contains the code, data, and results for my thesis, which explores the application of combining sentiment analysis with financial data to forecast stock movement. It combines structured numerical data from the S&P 500 with textual sentiment signals from multiple news sources, and evaluates both classical machine learning models and large language models (LLMs).

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training & Evaluation](#model-training--evaluation)
  - [Statistical Analysis](#statistical-analysis)
  - [S&P 500 Multimodal Models](#sp500-multimodal-models)
  - [Daily News Sentiment Models](#daily-news-sentiment-models)
  - [LLM Inference](#llm-inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

```
thesis-dsaa/
├── dataset_final/             # Raw and processed datasets
│   ├── Financial_Phrasebank/  # Sentiment-labeled phrasebank data
│   ├── FinSen_S&P500/         # Curated financial sentiment data aligned to S&P 500 prices
│   ├── Daily_Financial_News/  # Company-specific daily news text and annotations
│   └── data_creation.ipynb    # Notebook to generate and merge datasets
│
├── code_final/                # All modeling code and notebooks
│   ├── statistical_analysis/   # EDA and causality analysis notebooks
│   ├── S&P500_main/           # Scripts for numerical + text multimodal models
│   │   ├── fine-tuning_main/   # Training pipelines
│   │   ├── S&P500_models_main/ # Model definitions (Concatenation, etc.)
│   │   └── S&P500_ablations/   # Ablation study scripts
│   ├── daily_news_main/       # Sentiment models on daily news per ticker
│   ├── llm_inference/         # QWEN zero/few/one-shot sentiment evaluation scripts
│   └── ablations/             # Additional ablation experiments
│
├── results_final/             # Generated figures, tables, and report artifacts
│   ├── 21_day_rolling_volatility.png
│   ├── Sentiment_vs_Volatility.png
│   ├── PACF.png
│   ├── S&P500_FinSen/         # Detailed result CSVs and plots
│   ├── MRK/                   # Company-specific result folders
│   └── graph.twb              # Tableau workbook for interactive dashboards
│
└── README.md                  # Project overview and instructions
```

## Prerequisites

- Python 3.8 or higher
- Git
- [Tableau Desktop](https://www.tableau.com/) (for opening the `.twb` workbook)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sarthak-12/thesis-dsaa.git
   cd thesis-dsaa
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\\Scripts\\activate  # Windows PowerShell
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install pandas numpy scipy scikit-learn torch transformers jupyter matplotlib statsmodels seaborn
   ```

> *Feel free to adjust versions or add any missing packages to your environment.*

## Data Preparation

Run the `data_creation.ipynb` notebook in `dataset_final/` to:

1. Download or load all raw data sources.
2. Clean and preprocess text and price data.
3. Merge sentiment annotations with S&P 500 price series.
4. Export final CSVs in the same folder for modeling.

```bash
jupyter notebook dataset_final/data_creation.ipynb
```

## Model Training & Evaluation

### Statistical Analysis

- Located in `code_final/statistical_analysis/FinSen_S&P500.ipynb` and `FinSen_S&P500-causality.ipynb`.
- Performs EDA, rolling volatility calculations, partial autocorrelation (PACF), and Granger causality tests.

### S&P 500 Multimodal Models

- Scripts in `code_final/S&P500_main/`.
- Experiment with concatenated numerical and text embeddings.
- Supports fine-tuning and ablation studies on pre-trained transformers.

### Daily News Sentiment Models

- Per-ticker sentiment classification in `code_final/daily_news_main/`.
- Uses annotated news text for MS, MU, MRK, QQQ, NVDA.
- Compare classifier performance across companies.

### LLM Inference

- Scripts in `code_final/llm_inference/`.
- Evaluate QWEN models in zero-, one-, and few-shot settings.
- Generates sentiment labels and compares against annotations.

## Results

Final figures and tables are available under `results_final/`. Key highlights:

- **21-Day Rolling Volatility** vs. **Daily Sentiment Scores**
- Partial autocorrelation plots (PACF)
- Performance metrics for multimodal and text-only models
- LLM sentiment evaluation summaries

Open `results_final/graph.twb` in Tableau for an interactive dashboard.

## Contact

Sarthak Khanna
- GitHub: [sarthak-12](https://github.com/sarthak-12)

