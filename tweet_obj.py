import datetime
import nltk
from token_obj import token_obj

# the tweet text as an object; analyze and store tokens of tweet
class tweet_obj(object):
    def __init__(self, tweet):
        self.id = tweet['id']
        self.text = tweet['text']  # text
        self.source = tweet['source']  # from which platform the tweet is published
        self.lang = tweet['lang']  # tweet language
        self.fav_count = tweet['favorite_count']  # the number of favorites
        self.ret_count = tweet['retweet_count']  # the number of retweets
        self.retweet_source = ''  # the source from which the tweet is retweeted
        self.coordinates = []  # geo coordinates
        y = tweet['created_y']  # publish year
        m = tweet['created_m']  # publish month
        d = tweet['created_d']  # publish day
        h = tweet['created_h']  # publish hour
        min = tweet['created_min']  # publish minute
        self.datetime = datetime.datetime(y, m, d, h, min)
        self.tokenList = list()  # list of tokens
        self.sentenceList = list()  # list of identified sentences

    # identify the tweet as list of tokens; analyze types of all tokens
    def text_analyzer(self):
        import string

        # strip the punctuations from the beginning and the end of word, return the indexs of the remained part
        def puncStripper(word):
            if len(word) <= 1:
                return 0, len(word)
            myPuncList = ',.;!?:~-'
            i_start = 0
            i_end = len(word)
            # remove suffix
            while word[i_end-1] in myPuncList and i_start <= i_end:
                i_end -= 1
            # remove prefix
            while word[i_start] in myPuncList and i_start <= i_end:
                i_start += 1
            return i_start, i_end

        # remove all special characters, for twitter users, hashtags
        def specialStripper(word):
            valid_letters = string.letters + string.digits
            valid_symbol = '\'.-_'
            newword = ''
            for l in word:
                if l in valid_letters + valid_symbol:
                    newword += l
            return newword

        # determine if the word is valid (not unicode converted from some face items or others)
        def isFormalWord(word):
            valid_letters = string.letters  # + string.digits
            valid_symbol = '\'.-_'
            for l in word:
                if l not in valid_letters + valid_symbol:
                    return False
            if len(word) <= 1 and word[0] not in valid_letters:
                return False
            return True

        # if the text is a valid email address
        def validateEmail(email):
            import re
            if len(email) > 7:
                if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
                    return True
            return False

        # if its letters is all Captal
        def isAllCaptal(word):
            for l in word:
                if l not in string.uppercase:
                    return False
            return True

        '''
        classify the category the word belongs to
        prerequisite: each punctuation followed by a whitespace!
        word split:
            RT
            @...:
            https://
            special characters
            " ", '': hard to differentiate quote from others, or 'sarcasm'
            #... (hashtag)
            &...
            &amp; (&)
            xx:xx (time)
            xxx@xxx (email)
            ALL CAPITAL
            simplifications of contents, time, address: hard to identify
        '''
        def tokenClassifier(word):
            classifications = list()
            if len(word) <= 0:
                classifications.append(('', 0))
                return classifications

            # 0: text; 1: @; 2: #; 3: &; 4: RT; 5: https; 6: email; -1: special characters
            if len(word) >= 5 and word[:5] == 'https':
                classifications.append((word, 5))
                return classifications
            if len(word) >= 2 and word[:2] == 'RT':
                classifications.append((word, 4))
                return classifications
            if word[:1] == '@':
                classifications.append((specialStripper(word[1:]), 1))
                return classifications
            if word[:1] == '#':
                classifications.append((specialStripper(word[1:]), 2))
                return classifications
            if word[:1] == '&':
                classifications.append((specialStripper(word[1:]), 3))
                return classifications
            if validateEmail(word):
                classifications.append((word, 6))
                return classifications

            # continue to tokenize using nltk
            #tweet0_words = nltk.word_tokenize(tweet0)
            #further_tokenizer = nltk.tokenize.TreebankWordTokenizer()
            #further_tokenizer = nltk.tokenize.PunktWordTokenizer()
            further_tokenizer = nltk.tokenize.WordPunctTokenizer()
            nwords = further_tokenizer.tokenize(word)

            for nword in nwords:
                if isFormalWord(nword):
                    classifications.append((nword, 0))
                else:
                    classifications.append((nword, -1))

            return classifications

        # first split tokens by whitespaces
        tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(self.text)

        # classify each token and its quote status
        current_quote = ''
        quotes = '\'\"'
        is_retweet = False
        for i, token in enumerate(tokens):
            if len(current_quote) == 0 and token[0] in quotes:
                current_quote = token[0]
                token = token[1:]

            inQuote = len(current_quote) > 0

            if len(current_quote) > 0 and token[-1] == current_quote:
                current_quote = ''
                token = token[:-1]

            i_start, i_end = puncStripper(token)
            if i_end < len(token):
                suffix = token[i_end:]
            else:
                suffix = ''
            if i_start > 0:
                prefix = token[:i_start]
            else:
                prefix = ''
            token = token[i_start:i_end]

            classifications = tokenClassifier(token)
            for (token_content, token_type) in classifications:
                if i == 0 and token_type == 4:
                    # identify retweet
                    is_retweet = True
                elif is_retweet and i == 1:
                    # the second token is the source for retweets
                    if token_type == 1:
                        self.retweet_source = token_content
                    else:
                        print 'Warning: \'RT\' not followed by a valid tweet user account!'
                else:
                    # store other types of tokens
                    if len(token_content) > 0:
                        tokenObj = token_obj(token_content.lower(), token_type, self.id, inQuote, isAllCaptal(token_content), prefix, suffix)
                        self.tokenList.append(tokenObj)

        # if a completed pair of quotes is not found, do not set the contents as quotes
        if len(current_quote) > 0:
            for tokenObj in reversed(self.tokenList):
                if not tokenObj.inQuote:
                    break
                else:
                    tokenObj.inQuote = False

        # identify sentences
        sentences = nltk.tokenize.sent_tokenize(self.getClearText())
        self.sentenceList.extend(sentences)

        return

    # rebuild the text without special characters
    def getClearText(self):
        text_clear = ''

        quote_flag = False
        for token in self.tokenList:
            if quote_flag and not token.inQuote:
                text_clear += '\"'
                quote_flag = False

            # add a whitespace between tokens
            if len(text_clear) > 0:
                text_clear += ' '

            if not quote_flag and token.inQuote:
                text_clear += '\"'
                quote_flag = True

            if token.type not in {-1, 4, 5, 6}:
                text_clear += token.rebuild()
        return text_clear

    # analyze the sentence length; can add more functionalites
    def sentence_analysis(self):
        sentenceLens = list()
        for sentence in self.sentenceList:
            words = sentence.split(' ')
            sentenceLens.append(len(words))
        return sentenceLens

    # return contents of all tokens
    def getTokenTexts(self):
        return [tokenObj.content for tokenObj in self.tokenList]

    # get categories of all tokens
    def getTokenTypes(self):
        return [tokenObj.gettype() for tokenObj in self.tokenList]

    # generate an explaination string to show the tweet analysis result
    def convert2analysis(self):
        analysis = ''

        if len(self.retweet_source) > 0:
            analysis += 'Retweet from ' + self.retweet_source + ': '

        for i, tokenObj in enumerate(self.tokenList):
            analysis += tokenObj.content + '(' + tokenObj.gettype() + ')'

            if i < len(self.tokenList) - 1:
                analysis += ' '

        return analysis




pattern = r"""(?x)                   # set flag to allow verbose regexps 
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe 
              |\.\.\.                # ellipsis 
              |(?:[.,;"'?():-_`])    # special characters with meanings 
            """