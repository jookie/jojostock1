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
# Step 1: Sentiment Analysis with BERT
# We will use the transformers library from Hugging Face to load a pre-trained BERT model for sentiment analysis. This model will classify financial news or tweets as positive, neutral, or negative.

def get_sentiment_score(text):
    # DOV
    from transformers import pipeline
    # Load pre-trained BERT model for sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    return sentiment, score

# Step 2: Data Collection
# For this example, we will assume that you have a dataset of financial news or tweets. You can use APIs like Twitter API or financial news APIs to collect real-time data.
def collect_data():
    import pandas as pd
    # Example dataset
    data = {
        'date': ['2023-10-01', '2023-10-02', '2023-10-03'],
        'text': ['Great earnings report!', 'Market crash expected.', 'Stable growth predicted.']
    }
    df = pd.DataFrame(data)
    # Apply sentiment analysis
    df['sentiment'], df['score'] = zip(*df['text'].apply(get_sentiment_score))
    print(df)
    
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