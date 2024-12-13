
from __future__ import annotations
# import s2
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")

import subprocess
if st.button("Run Backtest"):
    # Start Lumibot as a subprocess
    result = subprocess.run(
        ["python", "s2.py"], 
        capture_output=True, text=True
    )
    if result.returncode == 0:
        st.success(f"Order placed successfully: {result.stdout}")
    else:
        st.error(f"Error placing order: {result.stderr}")



    