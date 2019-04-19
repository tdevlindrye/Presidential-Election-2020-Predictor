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
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_score

#get the dataset from csv
dataset = pd.read_csv("all_incumbents.csv")

#assign features to X dataset and classification to y dataset
X = dataset.drop(['Election Year', 'Birth Year','Name', 'Winner', 'KEY_1', 'KEY_2'], axis=1)
y = dataset['Winner']

#replace any missing data with an average for that feature
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)
 
#first loop for Naive Bayes classifier
g = 0
total = 0
while (g <= 1000):
	#split datasets into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	#create the classifier
	classifier = GaussianNB()
	#train the classifier with the training datasets 
	classifier.fit(X_train, y_train)
	#use the trained model to predict the test dataset
	y_pred = classifier.predict(X_test)
	g = g + 1
	#add all the precision scores together
	total = total + precision_score(y_test, y_pred, average = 'weighted')

print('\n')

#get and print the average precision
avg = total/10
NB_avg = avg
print('Naive Bayes classififier:')
print('Average accuracy is ' + str(round(avg, 2)) + '%')

#get and print the prediction for Trump
candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate) 
if prediction[0]:
	print('Trump is expected to be a winner with a ' + str(round((pred_prob[0][1]*100), 2)) + '% confidence.')
	NB_score1 = round(((.125*avg)*(pred_prob[0][1]*100)/10000),5)
else:
	print('Trump is expected to be a loser with a ' + str(round((pred_prob[0][0]*100), 2)) + '% confidence.')
	NB_score1 = -round(((.125*avg)*(pred_prob[0][0]*100)/10000),5)

#get and print the prediction for Biden
if prediction[1]:
	print('Biden is expected to be a winner with a ' + str(round((pred_prob[1][1]*100), 2)) + '% confidence.')
	NB_score2 = round(((.125*avg)*(pred_prob[1][1]*100)/10000),5)
else:
	print('Biden is expected to be a loser with a ' + str(round((pred_prob[1][0]*100), 2)) + '% confidence.')
	NB_score2 = -round(((.125*avg)*(pred_prob[1][0]*100)/10000),5)

print('Trump NB score: ' + str(NB_score1))
print('Biden NB score: ' + str(NB_score2))
print('\n')

#second loop for Decision Tree classifier
g = 0
total = 0
while (g <= 1000):
	#split datasets into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	#create the classifier
	classifier = DecisionTreeClassifier() 
	#train the classifier with the training datasets 
	classifier.fit(X_train, y_train)
	#use the trained model to predict the test dataset
	y_pred = classifier.predict(X_test)
	g = g + 1
	#add all the precision scores together
	total = total + precision_score(y_test, y_pred, average = 'weighted')

#get and print the average precision
avg = total/10
DT_avg = avg
print('Decision Tree classifier:')
print('Average accuracy is ' + str(round(avg, 2)) + '%')

#get and print the prediction for Trump
candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate) 
if prediction[0]:
	print('Trump is expected to be a winner with a ' + str(round((pred_prob[0][1]*100), 2)) + '% confidence.')
	DT_score1 = round(((.125*avg)*(pred_prob[0][1]*100)/10000),5)
else:
	print('Trump is expected to be a loser with a ' + str(round((pred_prob[0][0]*100), 2)) + '% confidence.')
	DT_score1 = -round(((.125*avg)*(pred_prob[0][0]*100)/10000),5)

#get and print the prediction for Biden
if prediction[1]:
	print('Biden is expected to be a winner with a ' + str(round((pred_prob[1][1]*100), 2)) + '% confidence.')
	DT_score2 = round(((.125*avg)*(pred_prob[1][1]*100)/10000),5)
else:
	print('Biden is expected to be a loser with a ' + str(round((pred_prob[1][0]*100), 2)) + '% confidence.')
	DT_score2 = -round(((.125*avg)*(pred_prob[1][0]*100)/10000),5)

print('Trump DT score: ' + str(DT_score1))
print('Biden DT score: ' + str(DT_score2))
print('\n')

#third loop for Random Forest classifier
g = 0
total = 0
while (g <= 1000):
	#split datasets into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	#create the classifier
	classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4)
	#train the classifier with the training datasets 
	classifier.fit(X_train, y_train)
	#use the trained model to predict the test dataset
	y_pred = classifier.predict(X_test)
	g = g + 1
	#add all the precision scores together
	total = total + precision_score(y_test, y_pred, average = 'weighted')

#get and print the average precision
avg = total/10
RF_avg = avg
print('Random Forest classifier:')
print('Average accuracy is ' + str(round(avg, 2)) + '%')

#get and print the prediction for Trump
candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate) 
if prediction[0]:
	print('Trump is expected to be a winner with a ' + str(round((pred_prob[0][1]*100), 2)) + '% confidence.')
	RF_score1 = round(((.125*avg)*(pred_prob[0][1]*100)/10000),5)
else:
	print('Trump is expected to be a loser with a ' + str(round((pred_prob[0][0]*100), 2)) + '% confidence.')
	RF_score1 = -round(((.125*avg)*(pred_prob[0][0]*100)/10000),5)

#get and print the prediction for Biden
if prediction[1]:
	print('Biden is expected to be a winner with a ' + str(round((pred_prob[1][1]*100), 2)) + '% confidence.')
	RF_score2 = round(((.125*avg)*(pred_prob[1][1]*100)/10000),5)
else:
	print('Biden is expected to be a loser with a ' + str(round((pred_prob[1][0]*100), 2)) + '% confidence.')
	RF_score2 = -round(((.125*avg)*(pred_prob[1][0]*100)/10000),5)

print('Trump RF score: ' + str(RF_score1))
print('Biden RF score: ' + str(RF_score2))
print('\n')

#fourth loop for Quadratic Discriminant Analysis
g = 0
total = 0
while (g <= 1000):
	#split datasets into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	#create the classifier
	classifier = QuadraticDiscriminantAnalysis()
	#train the classifier with the training datasets 
	classifier.fit(X_train, y_train)
	#use the trained model to predict the test dataset
	y_pred = classifier.predict(X_test)
	g = g + 1
	#add all the precision scores together
	total = total + precision_score(y_test, y_pred, average = 'weighted')

#get and print the average precision
avg = total/10
QDA_avg = avg
print('Quadratic Discriminant Analysis classififier:')
print('Average accuracy is ' + str(round(avg, 2)) + '%')

#get and print the prediction for Trump
candidate = pd.read_csv("Predict_test.csv")
prediction = classifier.predict(candidate)
pred_prob = classifier.predict_proba(candidate) 
if prediction[0]:
	print('Trump is expected to be a winner with a ' + str(round((pred_prob[0][1]*100), 2)) + '% confidence.')
	QDA_score1 = round(((.125*avg)*(pred_prob[0][1]*100)/10000),5)
else:
	print('Trump is expected to be a loser with a ' + str(round((pred_prob[0][0]*100), 2)) + '% confidence.')
	QDA_score1 = -round(((.125*avg)*(pred_prob[0][0]*100)/10000),5)

#get and print the prediction for Biden
if prediction[1]:
	print('Biden is expected to be a winner with a ' + str(round((pred_prob[1][1]*100), 2)) + '% confidence.')
	QDA_score2 = round(((.125*avg)*(pred_prob[1][1]*100)/10000),5)
else:
	print('Biden is expected to be a loser with a ' + str(round((pred_prob[1][0]*100), 2)) + '% confidence.')
	QDA_score2 = -round(((.125*avg)*(pred_prob[1][0]*100)/10000),5)

print('Trump QDA score: ' + str(QDA_score1))
print('Biden QDA score: ' + str(QDA_score2))
print('\n')

avg_avg = ((NB_avg+DT_avg+RF_avg+QDA_avg)/4)/100
classifier_score1 = (NB_score1+DT_score1+RF_score1+QDA_score1)*avg_avg
classifier_score2 = (NB_score2+DT_score2+RF_score2+QDA_score2)*avg_avg

print('The overall accuracy of the classifiers is: ' + str(round((avg_avg*100),2)) + '%')
print('Trump overall classifier score: ' + str(round((classifier_score1),5)))
print('Biden overall classifier score: ' + str(round((classifier_score2),5)))

print('\n')

#Twitter sentiment 
class TwitterClient(object): 

	#access info
	def __init__(self): 
		consumer_key = '7TDTYjMYolTTZmmemdhiYHEkF'
		consumer_secret = '7UxpqfucZCIBlilIr4qmO9UpDbEhUN5uMc5yfSJCxoHxHTdg4e'
		access_token = '2639643778-11yy6rCj0UvKPvrtiiXlxCFZO1YVAprN6soPbGB'
		access_token_secret = 'G5P21F7oIMTZfX1aYDr4xTh53FJqA2PtNT7oiMo7AxL3O'

		#authenticate
		try: 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			self.auth.set_access_token(access_token, access_token_secret) 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 

	#clean Tweet
	def clean_tweet(self, tweet): 
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

	#use TextBlob to get sentiment
	def get_tweet_sentiment(self, tweet): 
		analysis = TextBlob(self.clean_tweet(tweet)) 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'

	#fetch Tweets
	def get_tweets(self, query, count = 10): 
		tweets = [] 

		try: 
			fetched_tweets = self.api.search(q = query, count = count) 

			for tweet in fetched_tweets: 
				parsed_tweet = {} 

				parsed_tweet['text'] = tweet.text 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				if tweet.retweet_count > 0: 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			return tweets 

		except tweepy.TweepError as e: 
			print("Error : " + str(e)) 

def main(): 

	#create object and get Tweets for first candidate
	api = TwitterClient() 
	candidate1 = "Donald Trump"
	tweets = api.get_tweets(query = candidate1, count = 200) 
    
    #calculate percentage of positive and negative Tweets
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	perc_PTweets = 100*len(ptweets)/len(tweets)
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	perc_NTweets = 100*len(ntweets)/len(tweets)

	#calculate Twitter score for candidate
	fixed_perc_PTweets1 = perc_PTweets/(perc_NTweets+perc_PTweets)
	fixed_perc_NTweets1 = perc_NTweets/(perc_NTweets+perc_PTweets)
	Twitter_score1 = (fixed_perc_PTweets1 - fixed_perc_NTweets1) * (1-.059)

	#repeat for second candidate
	candidate2 = "Joe Biden"
	tweets = api.get_tweets(query = candidate2, count = 200) 
        
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	perc_PTweets = 100*len(ptweets)/len(tweets)
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	perc_NTweets = 100*len(ntweets)/len(tweets)

	fixed_perc_PTweets2 = perc_PTweets/(perc_NTweets+perc_PTweets)
	fixed_perc_NTweets2 = perc_NTweets/(perc_NTweets+perc_PTweets)
	Twitter_score2 = (fixed_perc_PTweets2 - fixed_perc_NTweets2) * (1-.059)

	#print out info for Twitter sentiment
	print('Trump tweets: ' + str(round((fixed_perc_PTweets1*100),2)) + '% positive and ' + str(round((fixed_perc_NTweets1*100),2)) + '% negative')
	print('Biden tweets: ' + str(round((fixed_perc_PTweets2*100),2)) + '% positive and ' + str(round((fixed_perc_NTweets2*100),2)) + '% negative' + '\n')	

	print('Trump Twitter score is: ' + str(round((Twitter_score1),2)))
	print('Biden Twitter score is: ' + str(round((Twitter_score2),2)) + '\n')

	#add scores and print them
	final_score1 = Twitter_score1 + classifier_score1
	final_score2 = Twitter_score2 + classifier_score2
	print('Trump final score is: ' + str(round((final_score1),2)))
	print('Biden final score is: ' + str(round((final_score2),2)))


if __name__ == "__main__": 
	main() 


