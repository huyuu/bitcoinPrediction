from twitterscraper import query_tweets
from google.cloud import language_v1 as language
from google.cloud.language_v1 import enums
import datetime as dt
import google


# Optional. If not specified, the language is automatically detected.
# For list of supported languages:
# https://cloud.google.com/natural-language/docs/languages
supportedLanguaeList = ['zh', 'zh-Hant', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'es']


# reference: https://qiita.com/7aguchi/items/b08172f8d108274807f5
class TwitterAnalyzingAI():
    def __init__(self):
        self.nlClient = language.LanguageServiceClient.from_service_account_json('bitcoinprediction-271223-9acaa40c9029.json')
        self.begindate = dt.date(2020, 3, 1)
        self.enddate = dt.date(2020, 3, 2)


    def getResultOfWords(self, words):
        query = words[0]
        for word in words[1:]:
            query += f' OR {word}'
        # https://github.com/taspinar/twitterscraper/blob/master/twitterscraper/query.py
        tweets = query_tweets(query=query, begindate=self.begindate, enddate=self.enddate, limit=1000)
        tweetsAmount = len(tweets)

        # for tweet in tweets:
        #     text = tweet.text
        #     if 'http' in text:
        #         continue
        #     isSentimental, sentimentType, score, magnitude = self.__getSentiment(text)
        #     print('#' * 100)
        #     print(f'{sentimentType} {score}x|{magnitude}|: {text}')
        #     print('#' * 100)
        #     print('\n')
        shouldContinue = input(f'Got {tweetsAmount} from "{query}". Should continue analyzing? [y/n]')
        if shouldContinue.lower() != 'y':
            return

        positiveAmount = 0
        negativeAmount = 0
        neutralAmount = 0
        step = 0
        print('#'*100)
        for tweet in tweets:
            step += 1
            if 'http' in tweet.text:
                continue
            else:
                isSentimental, sentimentType = self.__getSentiment(tweet.text)
                if sentimentType == 'P':
                    positiveAmount += 1
                elif sentimentType == 'N':
                    negativeAmount += 1
                elif sentimentType == 'M':
                    neutralAmount += 1

            doneRate = round(step/tweetsAmount*100)
            if doneRate % 10 == 0:
                print(f'{doneRate}% ({step}/{tweetsAmount}) done.')

        sentimentalAmount = positiveAmount + negativeAmount + neutralAmount
        unsentimentalAmount = tweetsAmount - sentimentalAmount
        print(f'Positive: {round(positiveAmount/tweetsAmount * 100)}%\nNegative: {round(negativeAmount/tweetsAmount * 100)}%\nNeutral: {round(neutralAmount/tweetsAmount * 100)}%\nUnsentimental: {round(unsentimentalAmount/tweetsAmount * 100)}%')
        return positiveAmount, negativeAmount, neutralAmount, unsentimentalAmount


    # https://cloud.google.com/natural-language/docs/analyzing-sentiment#language-sentiment-string-python
    def __getSentiment(self, text_content):
        # text_content = 'I am so happy and joyful.'

        # Available types: PLAIN_TEXT, HTML
        type_ = enums.Document.Type.PLAIN_TEXT


        document = {"content": text_content, "type": type_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8
        try:
            response = self.nlClient.analyze_sentiment(document, encoding_type=encoding_type)
            score = response.document_sentiment.score
            magnitude = response.document_sentiment.magnitude
            if abs(magnitude) < 0.5:
                return False, None
            elif abs(score) <= 0.2:
                return True, 'M'
            elif score > 0:
                return True, 'P'
            else:
                return True, 'N'
        except google.api_core.exceptions.InvalidArgument:
            return False, None


if __name__ == '__main__':
    # natural language

    ai = TwitterAnalyzingAI()
    # ai.getResultOfWords(['仮想通貨', 'BTC'])
    ai.getResultOfWords(['コロナ　AND 収束'])
