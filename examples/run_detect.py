import logging
import warnings
warnings.simplefilter("ignore")

from datetime import datetime

import signal, threading, os, time
import logging

from IPython.core.debugger import set_trace
from IPython.terminal.embed import embed

import time
from multiprocessing import Pool, TimeoutError
from functools import partial

import os, sys
import asyncio

import nest_asyncio
nest_asyncio.apply()

import inspect
#import ccxt.async_support as ccxt
import ccxt

import pandas as pd

# from djow_core.base.logger import get_logger
from logging import Logger

HTTP_PROXY = None

import httpx

logger = Logger('Harmonic')

from .harmonic_functions import HarmonicDetector


from .settings import NOTIFY_URL
from .settings import MAIN_SYMBOLS, ALT_SYMBOLS, PERIODS, ERROR_RATE
from .settings import PROCESS_COUNT


import redis
redis_client = redis.Redis()


import sys
def get_frame_fname(level=0):
    return sys._getframe(level+1).f_code.co_name

def send_alert(title: str, body: str):
    """
    用Redis缓存一下，主要是为了不重复发送
    """
    body = body.replace('\n', '\n\n')
    template='''
    \n\n
    {body}
    \n\n
    '''
    if not redis_client.exists(body):
        r = httpx.post(NOTIFY_URL, data={'text': title, 'desp': template.format(body=body)})
        redis_client.setex(body, 60 * 60 * 30, 1)
        return r.status_code
    else:
        return None


def kline_to_df(arr) -> pd.DataFrame:
    kline = pd.DataFrame(
        arr,
        columns=['ts', 'open', 'high', 'low', 'close', 'volume' ])
    kline.index = pd.to_datetime(kline.ts, unit='ms')
    kline.drop('ts', axis=1, inplace=True)
    return kline


def search_function(detector, exchange_id, symbols, periods=PERIODS, ccxt_args={}, savefig=False, predict=True, only_last=False, alert=False, plot=False):
    client = getattr(ccxt, exchange_id)(ccxt_args)
    client.load_markets()
    RETRY_TIMES=3
    for symbol in symbols:
        for period in periods:
            logger.info(f'------------------calculating {symbol} {period}------------------')
            retry = RETRY_TIMES
            while retry>0:
                try:
                    df = kline_to_df(client.fetch_ohlcv(symbol, period, limit=1000))

                    patterns, predict_patterns = detector.search_patterns(df, only_last=only_last, last_n=4, plot=plot, predict=predict)
                    break
                except Exception as e:
                    logger.error(e)
                    retry -= 1
                    if retry==0: raise
                    continue
            for pat in patterns:
                msg = f'{symbol} {period} \npatterns found: {pat[1]}, {pat[0]}, \n {pat[2]}, {pat[3]}'
                logger.info(msg)
                if alert and pat[0][-1][2] == len(df)-1:
                    send_alert(f'Pattern_Found_{symbol}_{period}', msg)

            for pat in predict_patterns:
                msg = '\n'.join([f'{p} {v}' for p,v in list(zip([str(dt) for dt in pat[1]], [p for p in pat[0]]))])
                msg = f'{symbol} {period} {msg} {pat[2]} {pat[3]}'
                logger.info(msg)
                if alert:
                    send_alert(f'Pattern_Predict_{symbol}_{period}', msg)


def main():
    #signal.signal(signal.SIGINT, partial(debug_handler, engine))

    PROXIES = {
        'http': HTTP_PROXY,
        'https': HTTP_PROXY,
    }

    ccxt_options = {'proxies': PROXIES}

    ok = 'okex'
    bn = 'binance'
    hb = 'huobipro'

    notify_msgs = []
    while True:
        epoch_start_time = datetime.now()
        predict_results = []
        #call_repl(engine)

        detector = HarmonicDetector(error_allowed=ERROR_RATE, strict=True)
        client = hb

        symbols = [*MAIN_SYMBOLS, *ALT_SYMBOLS]

        search = partial(search_function, detector, ccxt_args=ccxt_options)

        try:
            with Pool(PROCESS_COUNT) as p:
                # 检测主流币和山寨是否出现谐波模式
                r  = p.map_async(partial(search, client,  periods=PERIODS,  predict=PREDICT, only_last=True, alert=True, plot=False), [[si] for si in symbols])
                # 检测平台币
                r1 = p.map_async(partial(search,  hb, periods=PERIODS,  predict=PREDICT, only_last=True, alert=True, plot=False), [['HT/USDT']])
                r2 = p.map_async(partial(search, ok, periods=PERIODS,  predict=PREDICT, only_last=True, alert=True, plot=False), [['OKB/USDT']])
                r3 = p.map_async(partial(search, bn, periods=PERIODS,  predict=PREDICT, only_last=True, alert=True, plot=False), [['BNB/USDT']])
                r.get(timeout=360)
                r1.get(timeout=120)
                r2.get(timeout=120)
                r3.get(timeout=120)
        except TimeoutError as e:
            logger.error(e)
            continue
        except Exception as e:
            logger.error(e)
            continue
        finally:
            pass


        epoch_end_time = datetime.now()
        run_time = (epoch_end_time - epoch_start_time).total_seconds()
        print(f'------------|Total seconds: {run_time}s|---------------')
        if len(predict_results)>0:
            send_alert('Patterns predict', '\n\n'.join(predict_results))
        #time.sleep(10)


if __name__ == '__main__':
    # 从异步改成同步，用multiprocessing来达到并发效果
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(main())
    main()
