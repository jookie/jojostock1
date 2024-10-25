from __future__ import annotations

import streamlit as st

def jprint(s1 = '', s2 = '' , s3 = '', s4 = ""):
  a1 = str(s1) + str(s2) + str(s3) +str(s4)
  print   (a1)
  st.write(a1)
 
  
  