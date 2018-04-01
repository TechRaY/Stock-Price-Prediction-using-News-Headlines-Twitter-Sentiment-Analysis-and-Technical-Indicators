import sys
import os
import requests
import json
import string
import re
from unidecode import unidecode
import math
import schedule
import time
import codecs 
import csv

# Source: http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

tweet_text = dict()



#read the words from the sentiwordnet (list with positivity and negativity scores for over 13,000 words)
def create_dict():
	scores_file = open("SentiWordNet.txt", "r")
	scores_dict = dict()

	#add the word into the dictionary with the positive score - the negative score 
	for line in scores_file:
		line = line.strip()
		if line[0] != "#":
			tmp = line.split()
			word = tmp[4].split("#")
			scores_dict[str(word[0])] = float(tmp[2]) - float(tmp[3])

	#read in emoji dictionary
	reader = csv.reader(open('EmojiSentiment.csv'))

	emojiDict = dict()

	#add emoji scores to the dictionary
	for row in reader:
		emojiDict[row[0]] = row[1]
	scores_dict.update(emojiDict);

	return scores_dict

'''
#collect the tweets with a keyword(company name)
def collectTweets(keyword):
	global tweet_text

	#search query for twitter api
	url = "https://api.twitter.com/1.1/search/tweets.json?q=" + keyword + "&result_type=recent&count=100"
	returned_tweets = oauth_req(url, 'abd','hey')

	#convert tweet from json to dictionary
	res = json.loads(returned_tweets)

	#collect all tweet text and tweeter follower count
	for status in res["statuses"]:
		tweet_text[status["id"]] = [status["text"], status["user"]["followers_count"]]
		#tweet_text.append(status["text"])
	
	#iterate through more pages in twitter to collect as many tweets as possible and store those as well
	for num in range(0,100):
		if res is None:
			break
		if "next_results" not in res:
			break
		url = "https://api.twitter.com/1.1/search/tweets.json" + res["search_metadata"]["next_results"]
		returned_tweets = oauth_req(url, 'abd','hey')

		res = json.loads(returned_tweets)
		if res is None:
			break
		for status in res["statuses"]:
			tweet_text[status["id"]] = [status["text"], status["user"]["followers_count"]]
	
	return tweet_text
'''

# Function to tokenize text
def tokenizeText(line, wordNet_dict):
	
	tokens = []
	#split the tweet into words by spaces
	words = line.strip().split()

	tmplist = list()
	for word in words:
		try:
			word = unidecode(word)
		except:
			pass
        # APOSTROPHES:
        # Tokenize "'" if in contraction:
		if "'" in word:
			if word in contractions:
				expansion = contractions[word]
				all_words = expansion.split()
				for all_word in all_words:
					tmplist.append(all_word)
		else:
			tmplist.append(word)

	final_tokenized_list = list()

	for tmp_words in tmplist:
		replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
		tmp_words = tmp_words.translate(replace_punctuation)
		tmp_word = tmp_words.split()
		final_tokenized_list.extend(tmp_word)

	#return a list of the words of one tweet 
	return final_tokenized_list

#returns the score of one tweet
def sentimentAnalysis(tweet_list, scores_dict, tweet_followers):
	count = 0.0
	tweet_score = 0.0

	#iterate through each word in a single tweet and get the score from the sentiment library
	for each_tweet_word in tweet_list:
		if each_tweet_word.lower() in scores_dict:
			tweet_score = tweet_score + scores_dict[each_tweet_word.lower()]
			count = count + 1.0

	#if no tweets are collected 0 is returned
	if count != 0.0:
		tweet_followers = float(tweet_followers) + 1.0 # throw out tweets with authors who have 0 followers
		weight_factor = math.log(float(tweet_followers), 10)
		return weight_factor*(float(tweet_score)/count) #normalziing per tweet to avoid length factor
	else:
		return 0.0

#computes the final score by summing the tweets and normalizing
def scoreTweets(all_tweets_scores):
	max_abs_val = math.fabs(all_tweets_scores[0])
	for num in range(1, len(all_tweets_scores)):
		if math.fabs(all_tweets_scores[num]) > max_abs_val:
			max_abs_val = math.fabs(all_tweets_scores[num])

	total_score = 0.0
	for each_tweet_score in all_tweets_scores:
		if float(max_abs_val) > 0:
			total_score = total_score + (float(each_tweet_score)/float(max_abs_val))
	total_score_avg = float(total_score)/float(len(all_tweets_scores))

	#return the percent value
	return total_score_avg 

def main():

	#this path should have txt files for tweets of all of one company
	path = sys.argv[1]

	#create the sentiment dictionary using sentiwordnet 
	scores_dict = create_dict()

	words = []
	tweet_id_list = []
	final_tweets = []
	final_tweet_scores = list()

	#iterate through all the files in the directory
	for f in os.listdir(path):
		infile = codecs.open(os.path.join(path, f), 'r')
		while True:
			line = infile.readline()
			if line == '':
				break

			#the lines are formatted id , tuple of tweet text and follower count of tweeter
			values = line.split(',',1)
			id = values[0].split(':')[1]
			id = id[:-1]
			id = id[1:]

		 	#if the tweet had been collected as a dup, do not add it again into the tweet list
			if id in tweet_id_list:
				continue
		 	
			tweet_id_list.append(id)

			values = values[1].rsplit(',',1)
			num_followers = values[1][:-2]
			num_followers = num_followers[1:]

			text = values[0][3:]
			text = text[:-2]
		 	#print num_followers + '\n' + text
		 	
			tokenized_tweets = tokenizeText(text, scores_dict)

			final_tweet_scores.append(sentimentAnalysis(tokenized_tweets, scores_dict, num_followers))
			#use this line instead for all tweets
		 	#print "id = " + str(id) + " text = " + text + " num_followers = " + str(num_followers)
			#get the score of the company
		company_score = scoreTweets(final_tweet_scores) * 100
		company_score = "{:10.3f}".format(company_score)
		if float(company_score) > 0:
			print("Decision: Buy\nConsumer Sentiment: " + str(company_score.lstrip()) + '% +')
		else:
			print("Decision: Sell\nConsumer Sentiment: " + str(company_score.lstrip()) + '% -')

		infile.close()


		
main()







