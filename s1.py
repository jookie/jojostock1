# To integrate Streamlit for user interaction with Lumibot and Alpaca while resolving threading issues (ValueError: signal only works in main thread), use subprocesses to separate the frontend and backend logic. Here's how you can achieve this:

# 1. Streamlit Frontend
# Create a Streamlit app to collect the stock symbol and order details from the user and trigger the Lumibot backend via a subprocess:
import subprocess
import streamlit as st

# Streamlit UI
st.title("Trading App with Lumibot and Alpaca")
symbol = st.text_input("Enter Stock Symbol", value="AAPL")
order_type = st.selectbox("Order Type", ["Buy", "Sell"])
quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

if st.button("Place Order"):
    # Pass user input to Lumibot backend via subprocess
    st.write("=======BEFORE RUNNING SUBPROCESS 1=========")
    result = subprocess.run(
        ["python", "lumibot_logic.py", symbol, order_type, str(quantity)],
        capture_output=True, text=True
    )
    st.write("=======AFTER RUNNING SUBPROCESS 1=========")
    
    if result.returncode == 0:
        st.success(f"Order placed successfully: {result.stdout}")
    else:
        st.error(f"Error Placing Trade order:")
        # st.error(f"Error placing order: {result.stderr}")
        
    st.write("=======AFTER RUNNING SUBPROCESS 2=========") 