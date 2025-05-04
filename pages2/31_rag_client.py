
"""
Author: wangjia
Date: 2024-05-21 01:29:51
LastEditors: wangjia
LastEditTime: 2024-05-21 10:53:28
Description: file content
URL: https://github.com/AI4Finance-Foundation/FinRAG
"""

import streamlit as st
import requests
import subprocess
import time

# Function to start FastAPI server if it's not running
def start_fastapi():
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            st.success("✅ FastAPI server is already running.")
            return
    except requests.exceptions.RequestException:
        pass  # If request fails, start the server

    st.info("🚀 Starting FastAPI server...")
    subprocess.Popen(["python3", "pages/FinRag/main.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to start
    time.sleep(5)

# Start FastAPI automatically
start_fastapi()

# FinRAG Client to query FastAPI server
class FinRAGClient:
    BASE_URL = "http://localhost:8000"

    def query(self, user_query):
        payload = {
            "chatId": "12345",
            "ownerId": "user",
            "chatName": "Finance Assistant",
            "initInputs": {"categoryIds": []},
            "initOpening": "",
            "chatMessages": [{"role": "user", "rawContent": user_query}]
        }

        try:
            response = requests.post(f"{self.BASE_URL}/chat", json=payload)
            if response.status_code == 200:
                return response.json().get("data", {}).get("result", {})  # Ensures dictionary response
            else:
                return {"answer": f"⚠️ Error: Unable to fetch response. Status Code: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"answer": "❌ Error: FastAPI server is not running."}

# Initialize FinRAG Client
finrag = FinRAGClient()

# Streamlit UI
st.title("📊 FinRAG Financial Assistant")
st.write("Ask any finance-related question, and I'll fetch the best answer for you!")

# User input
query = st.text_input("🔎 Enter your financial query:")

response = {}  # Ensure response is always defined
if st.button("Get Answer"):
    if query:
        with st.spinner("🔄 Searching..."):
            response = finrag.query(query)  # This now always returns a dictionary
        
        # Display answer
        st.subheader("💡 Answer:")
        st.write(response.get("answer", "No response received."))

        # Display retrieved documents (if available)
        retrieved_docs = response.get("chunks", [])  # Use 'chunks' instead of 'retrieved_docs'
        if retrieved_docs:
            st.subheader("📄 Relevant Sources:")
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**{i+1}.** {doc.get('chunk', 'No content available')[:300]}...")  # Snippet view
    else:
        st.warning("⚠️ Please enter a query.")
