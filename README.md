NLP stock predictions. 

![Image of meme](https://i.imgur.com/2fS9YEz.jpg)

[Data from here](https://www.kaggle.com/aaron7sun/stocknews)

Plan:
	Do sentiment analysis of each piece of news. 
	For each day, combine all the sentiment analysis of all the news in that day and take the average. This becomes the sentiment of the given day. 
	To predict the stock market, take the past 5 days' sentiments and average them. Use this data to run ML algo to predict the stock market. 1 means up and 0 means down
	
Sentiment analysis tools. 
	We will have primarily 2 tooks. (Can add more if time permitted)
		1. TextBlob. This model is used to analysis tweet sentiment, but whatever 
			Installation: 
				$ pip install -U textblob
				$ python -m textblob.download_corpora
			
			Example Usage:
				msg = "I love you so much!!"
				analysis = TextBlob(msg)
				sentiment = analysis.sentiment.polarity

		2. vader SentimentIntensityAnalyzer
			Installation:
				import nltk
				nltk.download('vader_lexicon')	
			Example Usage:
				from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
   				analyser = SentimentIntensityAnalyzer()
    				sentence = "i love you so much"
    				score = analyser.polarity_scores(sentence)
