import os
import json
import logging
import warnings
import re

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime

warnings.filterwarnings("ignore")

#################################
# 1. Setup Logging
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
# 2. Data Loading & Processing
#################################
def load_and_prepare_data(subset_size=300):
    """
    Loads and prepares data by:
      - Reading news and numeric CSV files,
      - Merging them on the "Date" column,
      - Dropping unwanted columns,
      - Converting the Date column to datetime and sorting data,
      - Setting up the Movement column as binary (1 for up, 0 for down),
      - Filtering rows with non-empty 'cleaned_content'.
    For debugging, prints a subset (first subset_size rows).
    """
    logging.info("Loading data...")
    # Adjust file paths accordingly
    filepath_news = 'multimodal_S&P500_all.csv'
    filepath_num = 'merged_S&P500.csv'
    
    data = pd.read_csv(filepath_news)
    data_num = pd.read_csv(filepath_num)

    data['Date'] = pd.to_datetime(data['Date'])
    data_num['Date'] = pd.to_datetime(data_num['Date'])

    merged_data = pd.merge(data, data_num[['Date', 'Daily_Return']], on='Date', how='inner')
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

    # Process Movement column
    if 'Movement' in merged_data.columns:
        merged_data['Movement'] = merged_data['Movement'].apply(
            lambda x: 1 if str(x).strip().lower() == 'up' else 0
        )
    else:
        merged_data['Movement'] = (merged_data['Daily_Return'] > 0).astype(int)

    filtered_data = merged_data[merged_data['cleaned_content'].notnull() & (merged_data['cleaned_content'] != "")]
    filtered_data.reset_index(drop=True, inplace=True)

    logging.info(f"Full data shape after merge and filter: {filtered_data.shape}")

    subset_data = filtered_data.head(subset_size)
    logging.info(f"Subset data shape (for debugging): {subset_data.shape}")
    logging.debug(f"Subset sample:\n{subset_data.head(3)}")

    return filtered_data

#################################
# 3. LLM Loading & Inference (Few-Shot)
#################################
def load_llm_qwen(model_name="Qwen/Qwen2.5-7B", use_gpu=True):
    """
    Loads the Qwen model and tokenizer (with trust_remote_code=True) and creates a text-generation pipeline.
    """
    logging.info(f"Loading Qwen model: {model_name} ...")
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
        temperature=0.2,      # Lower temperature for deterministic results
        max_new_tokens=50     # Increase tokens if necessary to allow complete response
        # Optional: If available, you can add a stop sequence parameter.
    )
    logging.info("LLM pipeline initialized.")
    return text_gen

def build_prompt_news_few_shot(row, few_shot_examples=None):
    """
    Builds a few-shot prompt using a reinforced delimiter ("Final_Answer:") and the current news article.
    """
    examples_str = ""
    if few_shot_examples:
        examples_list = []
        for ex in few_shot_examples:
            examples_list.append(
                f"News Article:\n{ex['news']}\nFinal_Answer: {ex['answer']}\n"
            )
        examples_str = "\n".join(examples_list)
    
    current_article = str(row.get('cleaned_content', "No news content available"))
    prompt = (
        f"{examples_str}\n"
        f"News Article:\n{current_article}\n\n"
        "Based on the article above, predict if today's stock movement will be 'up' or 'down'.\n"
        "Answer with ONLY one word and prefix it with 'Final_Answer:' (e.g., Final_Answer: up).\n"
    )
    return prompt

def predict_movement_few_shot(llm_pipeline, row, few_shot_examples=None):
    """
    Uses few-shot inference to generate a prediction from the LLM and extracts the answer using regex.
    Returns 1 for 'up' and 0 for 'down'.
    """
    prompt = build_prompt_news_few_shot(row, few_shot_examples=few_shot_examples)
    logging.debug(f"Generated few-shot prompt:\n{prompt}")
    try:
        output = llm_pipeline(prompt)
        generated_text = output[0]["generated_text"].strip().lower()
        logging.debug(f"Model raw output: {generated_text}")
        # Use regex to extract word after 'final_answer:'
        pattern = r"final_answer:\s*(up|down)"
        match = re.search(pattern, generated_text)
        if match:
            answer = match.group(1)
            logging.debug(f"Extracted answer using regex: {answer}")
            return (1 if answer == "up" else 0), answer
        else:
            # Fallback: take the first token.
            first_word = generated_text.split()[0]
            logging.debug(f"Fallback first word extraction: {first_word}")
            if first_word in ["up", "down"]:
                return (1 if first_word == "up" else 0), first_word
            else:
                return 1, first_word  # default fallback.
    except Exception as e:
        logging.error(f"Error during LLM inference: {e}")
        return 1, "error"

#################################
# Few-Shot Examples
#################################
few_shot_examples = [
    {
        "news": ("Federal Reserve signals no drastic rate hike after softer inflation data and lower-than-expected CPI readings; "
                 "market sentiment suggests caution with an expectation of lower market movement."),
        "answer": "down"
    },
    {
        "news": ("Strong economic data and robust earnings reports have boosted investor confidence; "
                 "analysts are optimistic as companies outperform forecasts, resulting in bullish market sentiment."),
        "answer": "up"
    }
]

#################################
# 4. Main Execution - Write Predictions to JSONL
#################################
def main():
    logging.info("Starting few-shot inference for stock movement prediction...")
    
    # Load and prepare the data (using a subset for debugging; remove subset_size for full dataset)
    df = load_and_prepare_data(subset_size=300)
    df.dropna(subset=["Daily_Return"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Load the LLM pipeline.
    llm_pipeline = load_llm_qwen(model_name="Qwen/Qwen2.5-7B", use_gpu=True)
    
    output_filename = "predictions.jsonl"
    with open(output_filename, "w") as outfile:
        for idx, row in df.iterrows():
            pred_binary, pred_word = predict_movement_few_shot(llm_pipeline, row, few_shot_examples=few_shot_examples)
            record = {
                "Date": str(row['Date']),
                "news_article": row.get("cleaned_content", ""),
                "prediction_word": pred_word,
                "prediction_binary": pred_binary
            }
            outfile.write(json.dumps(record) + "\n")
            logging.debug(f"Written record {idx}: {record}")
    
    logging.info(f"Predictions written to {output_filename}")

if __name__ == "__main__":
    main()
