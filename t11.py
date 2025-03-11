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
    subprocess.Popen(["python", "pages/FinRag/main.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Give the server some time to start
    time.sleep(3)

# Start FastAPI automatically
start_fastapi()
