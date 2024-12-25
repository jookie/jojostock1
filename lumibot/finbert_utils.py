# 24-11-19 16:00:00,505: INFO: [mlstrat] The market is not currently open, skipping this trading iteration
# 2024-11-19 16:00:01,046: INFO: [mlstrat] Executing the after_market_closes lifecycle method

from __future__ import annotations
import sys ; sys.path.append("~/lib/rl")
import warnings ; warnings.filterwarnings("ignore")
import pandas as pd


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
import streamlit as st

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

# added by dov to prevent TOKENIZERS_PARALLELISM the warning:
# TOKENIZERS_PARALLELISM=False

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
         
        return probability, sentiment
    else:
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(
        ['markets responded negatively to the news!',
         'traders were displeased!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())
    st.write(tensor, sentiment)