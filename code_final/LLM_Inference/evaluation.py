import json
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from math import sqrt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------
# Read JSONL file into DataFrame
# -------------------------------
def read_predictions(jsonl_path):
    records = []
    with open(jsonl_path, "r") as infile:
        for line in infile:
            record = json.loads(line)
            records.append(record)
    df = pd.DataFrame(records)
    return df

# -------------------------------
# Calculate Classification Metrics for a fold
# -------------------------------
def calculate_classification_metrics(y_true, y_pred):
    # Using binary predictions (0/1) as probabilities for ROC-AUC is not ideal,
    # but we use it here as an approximation.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        roc_auc = float('nan')
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, mcc

# -------------------------------
# Calculate Financial Metrics for a fold
# -------------------------------
def calculate_financial_metrics(returns, predictions):
    # Strategy returns: if prediction=1, then take the daily return,
    # if prediction=0, assume no trade (return 0)
    strategy_returns = np.array(predictions) * np.array(returns)
    total_days = len(strategy_returns)
    
    # Directional win rate: percentage of days with positive strategy return
    directional_win_rate = np.sum(strategy_returns > 0) / total_days if total_days > 0 else float('nan')
    
    winning_sum = np.sum(strategy_returns[strategy_returns > 0])
    losing_sum = np.sum(strategy_returns[strategy_returns < 0])
    profit_factor = winning_sum / abs(losing_sum) if abs(losing_sum) > 1e-10 else float('nan')
    
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    sharpe_ratio = (mean_return / std_return) * sqrt(252) if std_return > 1e-10 else 0.0
    
    return directional_win_rate, profit_factor, sharpe_ratio

# -------------------------------
# Main Evaluation Script
# -------------------------------
def main():
    jsonl_file = "fold_wise_predictions_few_shot.jsonl"
    df = read_predictions(jsonl_file)
    logging.info(f"Loaded {len(df)} records from {jsonl_file}")
    
    # Ensure that Ground_Truth and Model_Prediction are numeric
    df['Ground_Truth'] = pd.to_numeric(df['Ground_Truth'], errors='coerce')
    df['Model_Prediction'] = pd.to_numeric(df['Model_Prediction'], errors='coerce')
    
    # Initialize lists to store metrics per fold
    fold_class_metrics = []
    fold_fin_metrics = []
    
    # Process fold-wise metrics (assuming the "Fold" field indicates the fold number)
    unique_folds = df['Fold'].unique()
    for fold in sorted(unique_folds):
        fold_df = df[df['Fold'] == fold]
        y_true = fold_df['Ground_Truth'].values
        y_pred = fold_df['Model_Prediction'].values
        accuracy, precision, recall, f1, roc_auc, mcc = calculate_classification_metrics(y_true, y_pred)
        fold_class_metrics.append({
            "Fold": fold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "MCC": mcc
        })
        
        # Financial metrics: use the Daily_Return field to compute strategy return
        returns = fold_df['Daily_Return'].values.astype(float)
        directional_win_rate, profit_factor, sharpe_ratio = calculate_financial_metrics(returns, y_pred)
        fold_fin_metrics.append({
            "Fold": fold,
            "Directional_Win_Rate": directional_win_rate,
            "Profit_Factor": profit_factor,
            "Sharpe_Ratio": sharpe_ratio
        })
    
    # Convert metrics to DataFrames
    df_class_metrics = pd.DataFrame(fold_class_metrics)
    df_fin_metrics = pd.DataFrame(fold_fin_metrics)
    
    # Print fold-wise classification metrics
    print("Fold-wise Classification Metrics:")
    print(df_class_metrics.to_string(index=False))
    
    # Print fold-wise financial metrics
    print("\nFold-wise Financial Metrics:")
    print(df_fin_metrics.to_string(index=False))
    
    # Overall metrics across all folds - for classification we concatenate all predictions
    all_y_true = df['Ground_Truth'].values
    all_y_pred = df['Model_Prediction'].values
    overall_acc, overall_prec, overall_rec, overall_f1, overall_roc_auc, overall_mcc = calculate_classification_metrics(all_y_true, all_y_pred)
    
    # For financial metrics, compute overall strategy returns across all records
    all_returns = df['Daily_Return'].values.astype(float)
    overall_directional_win_rate, overall_profit_factor, overall_sharpe = calculate_financial_metrics(all_returns, all_y_pred)
    
    print("\nOverall Classification Metrics:")
    print(f"Accuracy: {overall_acc:.2f}")
    print(f"Precision: {overall_prec:.2f}")
    print(f"Recall: {overall_rec:.2f}")
    print(f"F1-Score: {overall_f1:.2f}")
    print(f"ROC-AUC: {overall_roc_auc:.2f}")
    print(f"MCC: {overall_mcc:.2f}")
    
    print("\nOverall Financial Metrics:")
    print(f"Directional Win Rate: {overall_directional_win_rate:.2f}")
    print(f"Profit Factor: {overall_profit_factor:.2f}")
    print(f"Sharpe Ratio: {overall_sharpe:.2f}")

if __name__ == "__main__":
    main()
