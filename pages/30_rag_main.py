

# my_project is in jojostock1 and I am running it with streamlit on my minimac as follows:

# (venv) (base) dovpeles@Dovs-Mac-mini jojostock1 : streamlit run  ml_stock_sentiment.py

# in jojostock1 I have:
# jojostock1/pages/30_rag_main.py
# jojostock1/pages/FinRag/main.py 
# I want in pages a script to activate  /pages/FinRag/main.py which has the following script : 


import os
import subprocess
import time
import streamlit as st

# Function to start FastAPI server if it's not running
def start_fastapi():
    try:
        # Check if the FastAPI server is already running
        import requests
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            st.success("FastAPI server is already running.")
            return
    except:
        pass  # If request fails, start the server

    st.info("Starting FastAPI server...")
    
    # Run FastAPI server in a separate process
    subprocess.Popen(["python3", "pages/FinRag/main.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Give the server some time to start
    time.sleep(3)

# Start FastAPI automatically
start_fastapi()
