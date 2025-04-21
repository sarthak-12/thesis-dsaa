import os
import json
import logging
import warnings
import re

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import TimeSeriesSplit
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")

#################################
# 1. SETUP LOGGING
#################################
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("llm_inference_debug.log", mode="w"),
        logging.StreamHandler()
    ]
)

#################################
# 2. DATA LOADING & PREPROCESSING
#################################
def load_and_prepare_data():
    logging.info("Loading data...")
    filepath_news = 'multimodal_S&P500_all.csv'
    filepath_num = 'merged_S&P500.csv'
    
    data = pd.read_csv(filepath_news)
    data_num = pd.read_csv(filepath_num)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data_num['Date'] = pd.to_datetime(data_num['Date'])
    
    merged_data = pd.merge(
        data, 
        data_num[['Date', 'Daily_Return']], 
        on='Date', 
        how='inner'
    )
    if 'Unnamed: 0' in merged_data.columns:
        merged_data.drop(columns=['Unnamed: 0'], inplace=True)
    
    rolling_features = [
        'garch_cond_variance_lag1', 'garch_cond_volatility_lag1',
        'garch_residuals_lag1', 'rolling_cond_volatility_3_lag1',
        'rolling_cond_volatility_5_lag1'
    ]
    for col in rolling_features:
        if col in merged_data.columns:
            merged_data.drop(columns=[col], inplace=True)
    
    merged_data.sort_values(by="Date", inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
    
    filtered_data = merged_data[
        (merged_data['cleaned_content'].notnull()) &
        (merged_data['cleaned_content'] != "")
    ].reset_index(drop=True)
    
    logging.info(f"Full data shape after merge and filter: {filtered_data.shape}")
    logging.debug(f"Sample data:\n{filtered_data.head(3)}")
    return filtered_data

#################################
# 3. ATTACH PREVIOUS DAY NEWS
#################################
def attach_prev_day_news(df):
    logging.info("Attaching previous day news...")
    unique_dates = np.sort(df["Date"].unique())
    prev_news_map = {}
    
    for current_date in unique_dates:
        prev_rows = df[(df["Date"] < current_date) & (df["cleaned_content"] != "")]
        if prev_rows.empty:
            prev_news_map[current_date] = "No previous news available"
        else:
            last_date = prev_rows["Date"].max()
            articles = prev_rows[prev_rows["Date"] == last_date]["cleaned_content"]
            combined_articles = " ".join(articles.astype(str).tolist())
            prev_news_map[current_date] = combined_articles
    
    df["prev_day_news"] = df["Date"].map(prev_news_map)
    
    logging.debug("Finished attaching previous day news.")
    return df

#################################
# 4. TIME SERIES SPLIT
#################################
def get_time_series_folds(df, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    fold_num = 1
    for train_idx, test_idx in tss.split(df):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        logging.info(f"Fold {fold_num}: Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        logging.debug(f"Fold {fold_num} - Test sample:\n{test_df.head(3)}")
        folds.append((train_df, test_df))
        fold_num += 1
    return folds

#################################
# 5. LLM LOADING & PROMPT LOGIC
#################################
def load_llm_qwen(use_gpu=True):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    logging.info(f"Loading Qwen model: {model_name} on GPU={use_gpu}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto" if use_gpu else None,
        torch_dtype=torch.float16 if use_gpu else torch.float32
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        max_new_tokens=50
    )
    logging.info("LLM pipeline ready on GPU.")
    return text_gen

def build_prompt(row, few_shot_examples=None):
    """
    Builds a prompt using the following format:
    
    Based on the numerical data and news article from yesterday, predict whether the stock will move 'up' or 'down' today. 
    Use the following format to frame your answer. Do not echo the prompt.
    
    Format:
    <Numerical Data>: "Open: 4500; sentiment_volatility_lag1: 0; aggregate_sentiment_score_lag1: 2; Close_lag1: 4480; High_lag1: 4510; Volume_lag1: 345000000; Daily_Return_lag1: 0; Volatility_lag1: 0"
    <News Article>: Federal Reserve announces interest rate cut to boost economic growth
    </Answer/>:
    """
    # Few-shot examples are commented out for now.
    # examples_str = ""
    # if few_shot_examples:
    #     examples_list = []
    #     for ex in few_shot_examples:
    #         examples_list.append(
    #             f"News Article:\n{ex['news']}\nNumerical Data: {ex.get('numeric', 'N/A')}\nFinal_Answer: {ex['answer']}\n"
    #         )
    #     examples_str = "\n".join(examples_list)
    #     examples_str = "Few-Shot Examples:\n" + examples_str + "\n"
    
    # Use only the row's values
    # Remove decimals by converting values to integers.
    numerical_data = "; ".join([
        f"{col}: {int(float(row[col]))}" for col in [
            "Open", "sentiment_volatility_lag1", "aggregate_sentiment_score_lag1",
            "Close_lag1", "High_lag1", "Volume_lag1", "Daily_Return_lag1", "Volatility_lag1"
        ] if col in row and pd.notnull(row[col])
    ])
    news_article = str(row.get("prev_day_news", "No previous news available"))
    
    prompt = (
        "Based on the numerical data and the news article from yesterday, determine if today’s stock will move 'up' or 'down'.\n"
        "Respond with exactly one word: either 'up' or 'down'. Do not include any additional text or echo any part of this prompt.\n\n"
        f"<Numerical Data>: \"{numerical_data}\"\n"
        f"<News Article>: {news_article}\n\n"
        "</Answer/>:"
    )

    return prompt

def extract_answer(model_output):
    """
    Extracts 'up' or 'down' from the model's generated text using regex.
    Returns 'up', 'down', or None if no match is found.
    """
    # Define the regex pattern
    pattern = r"</Answer/>\s*:\s*(up|down)"
    
    # Search for the pattern in the model output
    match = re.search(pattern, model_output, re.IGNORECASE)
    
    # Return the matched word if found
    if match:
        return match.group(1).lower()  # Ensure lowercase ('up' or 'down')
    else:
        # Fallback: Try splitting on whitespace and check the first word
        first_word = model_output.strip().split()[-1].lower()
        if first_word in ["up", "down"]:
            return first_word
        return None  # No valid answer found
    
def predict_movement(llm_pipeline, row, few_shot_examples=None):
    """
    Generates a prediction ('up' or 'down') using the built prompt,
    then extracts the answer using regex.
    """
    prompt = build_prompt(row, few_shot_examples=few_shot_examples)
    logging.debug(f"Generated prompt:\n{prompt}")
    
    try:
        # Generate output from the model
        output = llm_pipeline(prompt)
        generated_text = output[0]["generated_text"].strip()
        logging.debug(f"Model raw output: {generated_text}")
        
        # Extract the answer using the regex function
        answer = extract_answer(generated_text)
        if answer == "up":
            return 1, "up"
        elif answer == "down":
            return 0, "down"
        else:
            logging.warning("Failed to extract answer. Defaulting to 'up'.")
            return 1, "error"
    except Exception as e:
        logging.error(f"Error during LLM inference: {e}")
        return 1, "error"

#################################
# 6. FOLD-WISE INFERENCE & REAL-TIME FILE WRITING
#################################
def main():
    logging.info("Starting fold-wise inference for unique dates in each test set...")
    df = load_and_prepare_data()
    df = attach_prev_day_news(df)
    folds = get_time_series_folds(df, n_splits=5)
    
    # few_shot_examples = []  # Commented out since we're not using few-shot examples.
    
    output_file = "fold_wise_predictions_improved.jsonl"
    # Load the LLM pipeline only once outside the fold loop.
    llm_pipeline = load_llm_qwen(use_gpu=True)
    
    with open(output_file, "w") as outfile:
        for fold_num, (train_df, test_df) in enumerate(folds, start=1):
            logging.debug(f"Processing Fold {fold_num}")
            # Group test rows by unique Date to ensure one prediction per date.
            test_df = test_df.groupby("Date").first().reset_index()
            
            for idx, row in test_df.iterrows():
                pred_binary, pred_word = predict_movement(llm_pipeline, row)
                record = {
                    "Date": row["Date"].isoformat(),
                    "Fold": fold_num,
                    "Daily_Return": row.get("Daily_Return", None),
                    "Daily_Return_lag1": row.get("Daily_Return_lag1", None),
                    "Ground_Truth": row.get("Movement", None),
                    "Model_Prediction": pred_binary
                }
                outfile.write(json.dumps(record) + "\n")
                outfile.flush()
                logging.debug(f"Fold {fold_num}, Date {row['Date'].strftime('%Y-%m-%d')}: Prediction={pred_word} (binary={pred_binary})")
    
    logging.info(f"Done! Wrote predictions for each fold to '{output_file}' in real time.")

if __name__ == "__main__":
    main()
