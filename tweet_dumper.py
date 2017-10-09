#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import tweepy
#https://github.com/tweepy/tweepy
#http://tweepy.readthedocs.io/en/v3.5.0/

#Twitter API credentials
consumer_key = "******"
consumer_secret = "******"
access_key = "******"
access_secret = "******"
# authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	#initialize a list to hold all the tweepy Tweets
	alltweets = []

	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)

	#save most recent tweets
	alltweets.extend(new_tweets)

	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1

	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print "getting tweets before id = %s" % (oldest)

		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name, count=200, max_id=oldest)

		#save most recent tweets
		alltweets.extend(new_tweets)

		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1

		print "...%s tweets downloaded so far" % (len(alltweets))

	print "...%s tweets downloaded totally" % (len(alltweets))

	return alltweets


def get_latest_tweets(screen_name, since_Id):
	count=0
	latest_tweets=[]

	print "getting tweets after id = %s" % (since_Id)

	try:
		new_tweets=api.user_timeline(screen_name=screen_name,since_id=since_Id,count=200)
		old_Id = new_tweets[-1].id - 1
	except IndexError:
		print "No new tweets"
		exit()

	count=len(new_tweets)
	print "%s tweets downloaded..." % (count)
	latest_tweets.extend(new_tweets)

	while len(new_tweets) > 0:
		print "getting tweets after %s but before %s" % (since_Id, old_Id)

		new_tweets=api.user_timeline(screen_name=screen_name,max_id=old_Id, since_id=since_Id,count=200)
		count += len(new_tweets)
		print "in loop %s tweets downloaded..." % (len(new_tweets))

		latest_tweets.extend(new_tweets)
		old_Id=latest_tweets[-1].id - 1

        print count

	print "...%s tweets downloaded totally" % (count)

	return latest_tweets


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("J_tsar")
