# definitions of token categories
token_cat = {-1:'special', 0:'normal', 1:'tweetUser', 2:'Hashtag', 3:'And', 4:'retweet', 5:'linkAddr', 6:'email'}

# object of word tokens
class token_obj(object):
    def __init__(self, content, type, tweetId, inQuote=False, isCapital=False, prefix = '', suffix = ''):
        self.content = content
        self.sentenceNum = 0
        self.type = type
        self.tweetId = tweetId
        self.inQuote = inQuote
        self.isCapital = isCapital
        self.prefix = prefix
        self.suffix = suffix

    # return the token type, for showing purpose
    def gettype(self):
        type = token_cat[self.type]
        if self.isCapital:
            type += '_CAPTIAL'
        if self.inQuote:
            type += '_quote'
        return type

    # set status as 'quoted'; used by tweet_obj
    def setquote(self, q):
        self.inQuote = q
        return

    # show the token with prefix and suffix
    def rebuild(self):
        if self.type == 3:
            return '&'

        if self.isCapital:
            return self.prefix + self.content.upper() + self.suffix
        else:
            return self.prefix + self.content + self.suffix
