import streamlit as st
import os
from finrag import FinRAG

# Initialize FinRAG
finrag = FinRAG()

# Streamlit UI
# st.title("📊 FinRAG Financial Assistant")
st.write("Ask any finance-related question, and I'll fetch the best answer for you!")

# User input
query = st.text_input("Enter your financial query:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching..."):
            response = finrag.query(query)
        
        # Display response
        st.subheader("💡 Answer:")
        st.write(response)
    else:
        st.warning("Please enter a query.")

# Optional: Display retrieved documents (if applicable)
if "retrieved_docs" in response:
    st.subheader("📄 Relevant Sources:")
    for i, doc in enumerate(response["retrieved_docs"]):
        st.write(f"**{i+1}. {doc['title']}**")
        st.write(doc["content"][:300] + "...")  # Show a snippet
