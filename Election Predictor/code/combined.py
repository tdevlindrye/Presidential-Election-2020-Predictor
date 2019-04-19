import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

dataset = pd.read_csv("all_incumbents.csv")

#print(dataset.shape)

#print(dataset.head())

X = dataset.drop(['Election Year', 'Birth Year','Name', 'Winner', 'KEY_1', 'KEY_2'], axis=1)

y = dataset['Winner']

#print(X)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

#print(X)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#classifier = DecisionTreeClassifier()  
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) 
#y_prob = classifier.predict_proba(X_test)

#print('Probability for the test set is: ')
#print(y_prob)

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_score
cm = confusion_matrix(y_test, y_pred)
#print(cm)  
#print(classification_report(y_test, y_pred)) 
#print(precision_score(y_test, y_pred, average = 'weighted'))

g = 0
total = 0
while (g <= 1000):
	X = imp.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	classifier = GaussianNB() 
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test) 
	g = g + 1
	total = total + precision_score(y_test, y_pred, average = 'weighted')

avg = total/10
print('Average accuracy for the Naive Bayes classifier: ')
print(str(round(avg, 2)) + '%')
#print(y_pred)

#feat_importance = classifier.tree_.compute_feature_importances(normalize=True)
#print("feat importance = " + str(feat_importance))

print('Here is a new prediction for Trump: ')

candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate)

print(prediction[0]) 

print('Here is a new prediction for Bernie: ')

print(prediction[1])

print(pred_prob)
#print(cm[0][0])
#print(cm[0][1])
#print(cm[1][0])
#print(cm[1][1])

g = 0
total = 0
while (g <= 1000):
	X = imp.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	classifier = DecisionTreeClassifier() 
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test) 
	g = g + 1
	total = total + precision_score(y_test, y_pred, average = 'weighted')

avg = total/10
print('Average accuracy for the Decision Tree classifier: ')
print(str(round(avg, 2)) + '%')
#print(y_pred)

print('Here is a new prediction for Trump: ')

candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate)

print(prediction[0]) 

print('Here is a new prediction for Bernie: ')

print(prediction[1])

print(pred_prob)

g = 0
total = 0
while (g <= 1000):
	X = imp.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test) 
	g = g + 1
	total = total + precision_score(y_test, y_pred, average = 'weighted')

avg = total/10
print('Average accuracy for the Random Forest classifier: ')
print(str(round(avg, 2)) + '%')
#print(y_pred)

print('Here is a new prediction for Trump: ')

candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate)

print(prediction[0]) 

print('Here is a new prediction for Bernie: ')

print(prediction[1])

print(pred_prob)

g = 0
total = 0
while (g <= 1000):
	X = imp.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	classifier = QuadraticDiscriminantAnalysis()
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test) 
	g = g + 1
	total = total + precision_score(y_test, y_pred, average = 'weighted')

avg = total/10
print('Average accuracy for the Quadratic classifier: ')
print(str(round(avg, 2)) + '%')
#print(y_pred)

print('Here is a new prediction for Trump: ')

candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate)

print(prediction[0]) 

print('Here is a new prediction for Bernie: ')

print(prediction[1])

print(pred_prob)

class TwitterClient(object): 
	''' 
	Generic Twitter Class for sentiment analysis. 
	'''
	def __init__(self): 
		''' 
		Class constructor or initialization method. 
		'''
		# keys and tokens from the Twitter Dev Console 
		consumer_key = '7TDTYjMYolTTZmmemdhiYHEkF'
		consumer_secret = '7UxpqfucZCIBlilIr4qmO9UpDbEhUN5uMc5yfSJCxoHxHTdg4e'
		access_token = '2639643778-11yy6rCj0UvKPvrtiiXlxCFZO1YVAprN6soPbGB'
		access_token_secret = 'G5P21F7oIMTZfX1aYDr4xTh53FJqA2PtNT7oiMo7AxL3O'

		# attempt authentication 
		try: 
			# create OAuthHandler object 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			# set access token and secret 
			self.auth.set_access_token(access_token, access_token_secret) 
			# create tweepy API object to fetch tweets 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 

	def clean_tweet(self, tweet): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

	def get_tweet_sentiment(self, tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		# create TextBlob object of passed tweet text 
		analysis = TextBlob(self.clean_tweet(tweet)) 
		# set sentiment 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'

	def get_tweets(self, query, count = 10): 
		''' 
		Main function to fetch tweets and parse them. 
		'''
		# empty list to store parsed tweets 
		tweets = [] 

		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 

			# parsing tweets one by one 
			for tweet in fetched_tweets: 
				# empty dictionary to store required params of a tweet 
				parsed_tweet = {} 

				# saving text of tweet 
				parsed_tweet['text'] = tweet.text 
				# saving sentiment of tweet 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# return parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			# print error (if any) 
			print("Error : " + str(e)) 

def main(): 
	# creating object of TwitterClient Class 
	api = TwitterClient() 
	# calling function to get tweets 
	candidate = "Bernie Sanders"
	tweets = api.get_tweets(query = candidate, count = 2000) 
        
	print("Sentiment analysis for:", candidate)
	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	# percentage of positive tweets 
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	# picking negative tweets from tweets 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	# percentage of negative tweets 
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
	# percentage of neutral tweets 
	#print(len(ptweets), len(ntweets), len(tweets))
	print("Neutral tweets percentage: {} %".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets))) 

	# printing first 5 positive tweets 
	#print("\n\nPositive tweets:") 
	#for tweet in ptweets[:10]: 
		#print(tweet['text']) 

	# printing first 5 negative tweets 
	#print("\n\nNegative tweets:") 
	#for tweet in ntweets[:10]: 
		#print(tweet['text']) 

if __name__ == "__main__": 
	# calling main function 
	main() 


