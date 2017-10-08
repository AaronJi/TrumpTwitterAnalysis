'''
https://github.com/bear/python-twitter
https://python-twitter.readthedocs.io/en/latest/twitter.html
https://dev.twitter.com/overview/api/tweets
http://www.tweepy.org
'''

import os
import json

# Getting Twitter data using the API
import twitter

from tweet_dumper import *

def mkdir_p(path):
    import errno

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

'''  
Open "path" for writing, creating any parent directories as needed.
'''
def safe_open_w(path, _):
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')

# replace them by keys and tokens of yourself!
consumer_key = "*******"
consumer_secret = "*******"
access_token = "*******"
access_token_secret = "*******"

'''
status:
['contributors', 'truncated', 'text', 'is_quote_status', 'in_reply_to_status_id', 'id', 'favorite_count', '_api', 
'author', '_json', 'coordinates', 'entities', 'in_reply_to_screen_name', 'id_str', 'retweet_count', 
'in_reply_to_user_id', 'favorited', 'source_url', 'user', 'geo', 'in_reply_to_user_id_str', 'possibly_sensitive', 
'lang', 'created_at', 'in_reply_to_status_id_str', 'place', 'source', 'retweeted']
'''

def tweet2df(tweets):
    columns = ['id', 'text', 'favorite_count', 'retweet_count', 'lang', 'source', 'created_y', 'created_m', 'created_d', 'created_h', 'created_min', 'coordinates']
    data = [
        [tweet.id, tweet.text.encode("utf8"), tweet.favorite_count, tweet.retweet_count, tweet.lang, tweet.source,
         tweet.created_at.year, tweet.created_at.month, tweet.created_at.day, tweet.created_at.hour, tweet.created_at.minute, tweet.coordinates]
        for tweet in tweets]
    df = pd.DataFrame(data, columns=columns)
    return df

def combine_from_old_tweets(df, old_df):
    from datetime import datetime

    ids = df['id'].values

    old_tweets = list()
    insert = 0
    for index, row in old_df.iterrows():
        if insert == 0 and row['id'] not in ids:
            # the first time found old tweets
            insert = 1
        if insert > 0:
            dt = datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S")
            old_tweets.append([row['id'], row['text'], dt.year, dt.month, dt.day, dt.hour, dt.minute])

        if insert > 1:
            insert += 1

    old_tweets_df = pd.DataFrame(old_tweets, columns=['id', 'text', 'created_y', 'created_m', 'created_d', 'created_h', 'created_min'])

    new_df = pd.concat([df, old_tweets_df])

    return new_df


if __name__ == '__main__':
    screen_name = 'realDonaldTrump'  # screen_name = 'Donald J. Trump'

    # if we only get new tweets and combine them into the old results, or start from zero
    get_from_old_data = True

    output_filename = screen_name + "_tweets.csv"
    output_path = os.path.join(os.path.expanduser("~"), "PycharmProjects", "twitterAnalyze", "Data", output_filename)

    if get_from_old_data:
        input_path = output_path
        df = pd.read_csv(input_path)
        since_Id = df['id'].loc[0]

        print 'Already have %d tweets' % len(df)

        new_tweets = get_latest_tweets(screen_name, since_Id)
        new_df = tweet2df(new_tweets)

        print new_df

        all_df = pd.concat([new_df, df])
        print 'Now we have %d tweets' % len(all_df)

    else:
        all_tweets = get_all_tweets(screen_name)
        all_df = tweet2df(all_tweets)

    # transform the tweepy tweets into a 2D array that will populate the csv
    all_df.to_csv(output_path, encoding='utf-8', index=False)


    # Get your "home" timeline
    if False:
        # output_filename = screen_name + "_tweets.json"

        authorization = twitter.OAuth(access_token, access_token_secret, consumer_key, consumer_secret)
        t = twitter.Twitter(auth=authorization)
        htl = t.statuses.user_timeline(screen_name=screen_name, count=999)
        print len(htl)
        #print htl
        print htl[0]
        print len(htl[0])
        print htl[0]['user']
        print htl[0]['text']

        twitter_api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token_key=access_token, access_token_secret=access_token_secret)
        print(twitter_api.VerifyCredentials())
        statuses = twitter_api.GetUserTimeline(screen_name=screen_name, count=2, include_rts=False, exclude_replies=True)
        for status in statuses:
            if (status.lang == 'en'):
                print status.text

        # outtweets = [[tweet.id, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
        # write the csv
        # with safe_open_w(output_path, 'wb') as f:
        #	writer = csv.writer(f)
        #	writer.writerow(["id", "created_at", "text"])
        #	writer.writerows(outtweets)

        with safe_open_w(output_filename, 'a') as output_file:
            search_results = t.search.tweets(q="python", count=100)['statuses']
            for tweet in search_results:
                if 'text' in tweet:
                    print(tweet['user']['screen_name'])
                    print(tweet['text'])
                    print()
                    output_file.write(json.dumps(tweet))
                    output_file.write("\n\n")

