from __future__ import annotations
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv ; 
load_dotenv(); 
api_secret = os.getenv("API_SECRET" )
api_key    = os.getenv("API_KEY")
base_url   = os.getenv("BASE_URL")
paper = True 
import subprocess
if st.button("Run Backtest"):
    # Start Lumibot as a subprocess
    result = subprocess.run(
        ["python", "t1.py"], 
        capture_output=True, text=True
    )
    st.write(f"SUBOROCESS: {result} ")
    st.write("Output:", result.stdout)
    st.error(result.stderr)
