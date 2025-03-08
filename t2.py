# Part 4: Preprocess fundamental data
# ==========
# - Import finanical data downloaded from Compustat via WRDS(Wharton Research Data Service)
# - Preprocess the dataset and calculate financial ratios
# - Add those ratios to the price data preprocessed in Part 3
# - Calculate price-related ratios such as P/E and P/B

import pandas as pd

# Import fundamental data from my GitHub repository
url = 'https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv'



# /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2718: DtypeWarning: Columns (16,25) have mixed types.Specify dtype option on import or set low_memory=False.
#   interactivity=interactivity, compiler=compiler, result=result)

fund = pd.read_csv(url)

# Check the imported dataset
# fund.head()
print(fund)