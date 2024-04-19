import pandas as pd 

df = pd.read_csv('dfSentiment.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)



##only dates from 3/15-4/16 work for these tickers
#'AMZN','TSLA','META','MSFT','GOOG','NVDA','AAPL'

#because this is the data we webscraped. To change the webscraped data, chnage the tickers in ticker list and re run

def getSentimentScore(ticker, startDate, endDate):
    def getDfStock(ticker, startDate, endDate):
        # Convert startDate and endDate to datetime objects to ensure proper comparison
        start_date = pd.to_datetime(startDate)
        end_date = pd.to_datetime(endDate)

        # Ensure the 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Filter the DataFrame for the given ticker and the date range inclusive
        filtered_df = df[(df['ticker'] == ticker) & (df['date'] >= start_date) & (df['date'] <= end_date)]

        return filtered_df 

    filtered_df = getDfStock(ticker, startDate, endDate)

    sentiments = filtered_df['sentiment']

    # Calculate the total count of each sentiment
    sentiment_counts = sentiments.value_counts(normalize=True)  # This gives the relative frequency of each sentiment


    # Calculate the weighted score
    # Multiply each sentiment count by its weight and sum up
    weighted_score = (sentiment_counts.get(0, 0) * 0 +  # Negative sentiment weight
                      sentiment_counts.get(1, 0) * 0.5 +  # Neutral sentiment weight
                      sentiment_counts.get(2, 0) * 1)  # Positive sentiment weight

    return weighted_score

# Example usage
ticker = "AAPL"
startDate = "2024-03-16"
endDate = "2024-04-12"
score = getSentimentScore(ticker, startDate, endDate)
print(f"Sentiment Score for {ticker} from {startDate} to {endDate}: {score:.2f}")


ticker2 ="MSFT"
score = getSentimentScore(ticker2, startDate, endDate)
print(f"Sentiment Score for {ticker2} from {startDate} to {endDate}: {score:.2f}")


