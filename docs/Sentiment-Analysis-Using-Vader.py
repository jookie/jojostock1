# https://github.com/akshaytekam/Sentiment-Analysis-Using-Vader/blob/main/Sentiment%20Analysis

import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

url = 'https://www.flipkart.com/hamtex-polycotton-double-bed-cover/product-reviews/itma5c9f08efe504?pid=BCVG2ZGSDZ3WSGTF&lid=LSTBCVG2ZGSDZ3WSGTFDBZ9IO&marketplace=FLIPKART'

response = requests.get(url)
content = response.content

soup = BeautifulSoup(content, 'html.parser')

reviews_container = soup.find('div', {'class': '_1YokD2 _3Mn1Gg col-9-12'})

review_divs = reviews_container.find_all('div', {'class': 't-ZTKy'})

reviews = []
for child in review_divs:
    third_div = child.div.div
    text = third_div.text.strip()
    reviews.append(text)

# Save the reviews to an Excel file in current directory
data = pd.DataFrame({'review': reviews})
data.to_excel('reviews.xlsx', index=False)

def sentiment_Vader(text):
    over_all_polarity = sid.polarity_scores(text)
    if over_all_polarity['compound'] >= 0.05:
        return "positive"
    elif over_all_polarity['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()
data['polarity'] = data['review'].apply(lambda review: sentiment_Vader(review))

result_data = data.to_excel('G:/DS Programs/WebScrapingEnv/sentiment_result.xlsx')