# This script performs a basic sentiment analysis on sample crypto headlines.
# It requires the 'requests', 'pandas', and 'nltk' libraries.

import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
import random

# --- 1. CONFIGURATION ---
NEWS_API_KEY = "YOUR_API_KEY_HERE"  # Note: A real API key would be required for a live feed
CRYPTO_TERM = "Bitcoin OR Cryptocurrency OR BTC"
SORT_BY = "publishedAt"
LANGUAGE = "en"

# --- 2. DATA ACQUISITION (Simulated for this script) ---

def fetch_crypto_headlines(api_key, query):
    """
    Simulates fetching recent news headlines using the News API.
    Since we cannot guarantee a live API key, this function returns mock data.
    
    If you have a key, uncomment the request and replace the mock data.
    """
    print("--- 1. Fetching Sample Headlines ---")
    
    # --- MOCK DATA FOR DEMONSTRATION ---
    mock_headlines = [
        "Major Investment Firm announces $1 Billion Bitcoin acquisition, boosting market confidence.",
        "Regulatory crackdown in Asia causes brief market instability and fear.",
        "New decentralized finance (DeFi) project launches with strong community support.",
        "Cryptocurrency exchange suffers security breach, resulting in minor losses.",
        "BTC price maintains steady rise, signaling strong investor holding.",
        "Analyst warns of upcoming correction after three weeks of extreme volatility.",
        "Blockchain technology adopted by global logistics company for supply chain efficiency.",
        "Social media speculation drives minor meme coin prices to new highs.",
        "Inflation concerns push more institutional investors toward Bitcoin as a hedge.",
        "Energy usage of mining operations criticized in environmental report."
    ]
    # --- END MOCK DATA ---

    # --- REAL API CALL (Requires actual API Key) ---
    # try:
    #     url = (f'https://newsapi.org/v2/everything?q={query}&sortBy={SORT_BY}&language={LANGUAGE}&apiKey={api_key}')
    #     response = requests.get(url)
    #     response.raise_for_status() # Raises an exception for bad status codes
    #     articles = response.json().get('articles', [])
    #     headlines = [article['title'] for article in articles if article.get('title')]
    #     if headlines:
    #         print(f"Successfully fetched {len(headlines)} headlines from API.")
    #         return headlines
    # except requests.exceptions.RequestException as e:
    #     print(f"API Request Failed: {e}. Using mock data instead.")
    #     return mock_headlines

    print("Using 10 mock headlines for sentiment analysis.")
    return mock_headlines


# --- 3. SENTIMENT ANALYSIS ---

def analyze_sentiment(headlines):
    """
    Analyzes the sentiment of each headline using the VADER model.
    """
    print("--- 2. Performing VADER Sentiment Analysis ---")
    
    # Initialize VADER (requires the 'vader_lexicon' data to be downloaded)
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError:
        # Suggest download if the lexicon is missing
        print("\n!!! NLTK Data Missing !!!")
        print("Please run 'python -c \"import nltk; nltk.download(\'vader_lexicon\')\"' in your terminal.")
        return pd.DataFrame()


    results = []
    
    for headline in headlines:
        vs = analyzer.polarity_scores(headline)
        
        # Determine the overall label based on the 'compound' score
        if vs['compound'] >= 0.05:
            label = 'Positive'
        elif vs['compound'] <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
            
        results.append({
            'Headline': headline,
            'Compound_Score': vs['compound'],
            'Sentiment_Label': label
        })

    # Convert results list to a DataFrame
    sentiment_df = pd.DataFrame(results)
    return sentiment_df


# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # 1. Fetch data
    headlines = fetch_crypto_headlines(NEWS_API_KEY, CRYPTO_TERM)
    
    if not headlines:
        print("Could not retrieve any headlines for analysis.")
    else:
        # 2. Analyze sentiment
        sentiment_data = analyze_sentiment(headlines)

        if not sentiment_data.empty:
            
            # 3. Print Results
            print("\n--- 3. Sentiment Analysis Results ---")
            print(sentiment_data.to_string(index=False))

            # 4. Summarize
            summary = sentiment_data['Sentiment_Label'].value_counts()
            
            print("\n--- 4. Summary Count ---")
            print(summary.to_string())
            
            # 5. Save results to CSV (for use in the Streamlit GUI)
            sentiment_data.to_csv("crypto_sentiment_results.csv", index=False)
            print("\nSentiment analysis complete! Results saved to 'crypto_sentiment_results.csv'.")
            
            print("\n-------------------------------------------------------------")
            print("Phase 3 complete (Prophet Model + Sentiment Analysis).")
            print("Next step is Phase 4: Build the Graphical User Interface (GUI).")
            print("-------------------------------------------------------------")
