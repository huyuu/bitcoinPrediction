import requests as rq
import numpy as nu
import pandas as pd
import datetime as dt
import hmac
import hashlib


" reference: https://qiita.com/ti-ginkgo/items/7e15bdac6618c07534be"
" reference: https://coincheck.com/ja/documents/exchange/api"
class CoincheckClient():
    def __init__(self):
        self.tickerURL = 'https://coincheck.com/api/ticker'
        self.accessKey = 'fvch5v'
        self.secretKey = 'fvch5vA'


    def ticker(self):
        json = rq.get(self.tickerURL).json()
        json['timestamp'] = dt.datetime.fromtimestamp(json['timestamp'])
        json['ltp'] = json['last']
        return json


    def buy(self, rate, amount):
        url = 'https://coincheck.com/api/exchange/orders'
        nonce = f'{int(dt.datetime.utcnow().timestamp())}'
        params = {
            "pair": "btc_jpy",
            "order_type": "buy",
            "rate": f'{rate}',
            "amount": f'{amount}',
        }
        signature = self.__getSignature(nonce, url, params)
        header = self.__getHeader(nonce, signature)
        response = rq.post(url, data=params, headers=header)
        if 'json' in response.headers.get('content-type'):
            return response.json()
        else:
            return response.text


    def sell(self, rate, amount):
        url = 'https://coincheck.com/api/exchange/orders'
        nonce = f'{int(dt.datetime.utcnow().timestamp())}'
        params = {
            "pair": "btc_jpy",
            "order_type": "sell",
            "rate": f'{rate}',
            "amount": f'{amount}',
        }
        signature = self.__getSignature(nonce, url, params)
        header = self.__getHeader(nonce, signature)
        response = rq.post(url, data=params, headers=header)
        if 'json' in response.headers.get('content-type'):
            return response.json()
        else:
            return response.text


    def __getSignature(self, nonce, url, body):
        message = f'{nonce}{url}{body}'
        # https://kaworu.jpn.org/python/Pythonでhmacを計算する
        return hmac.new(bytes(self.secretKey.encode('ascii')), bytes(message.encode('ascii')), hashlib.sha256).hexdigest()


    def __getHeader(self, nonce, signature):
        header = {
            'ACCESS-KEY': self.accessKey,
            'ACCESS-NONCE': nonce,
            'ACCESS-SIGNATURE': signature,
            'Content-Type': 'application/json' # 超重要。
        }
        return header


if __name__ == '__main__':
    client = CoincheckClient()
    response = client.buy(rate=100, amount=0.00001)
    print(response)
