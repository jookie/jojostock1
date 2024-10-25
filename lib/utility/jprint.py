from __future__ import annotations

import streamlit as st

def jprint(s1 = '', s2 = '' , s3 = '', s4 = ""):
  a1 = str(s1) + str(s2) + str(s3) +str(s4)
  print   (a1)
  st.write(a1)
  
# def jprint2(*args):
#     # Convert all inputs to strings and handle lists/arrays
#     result = []
#     for arg in args:
#         if isinstance(arg, (list, tuple)):  # Check if the argument is a list or tuple
#             result.extend(map(str, arg))    # Convert each item in the list to a string
#         else:
#             result.append(str(arg))         # Convert individual elements to strings

#     # Join the result list with spaces and print in a single line
#     print(" ".join(result), end=' ')
  
  
  