#pip install torch
#streamlit python 3.9.12
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import pandas as pd
import preprocess
import numpy as np

# Read in labeled stock tweet sentiment data
data = pd.read_csv('train/stock_data.csv')

# Process the Tweets for NLP
data = preprocess.Preprocess_Tweets(data)
display(data)