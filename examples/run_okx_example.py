"""
Harmonic Patterns Detector - OKX Real-time Scanning and Plotting Example

This example demonstrates how to:
1. Initialize the `HarmonicDetector`.
2. Connect to the OKX exchange using `ccxt`.
3. Fetch historical OHLCV (candle) data for BTC/USDT and ETH/USDT on various timeframes.
4. Detect completed and predicting harmonic patterns.
5. Print findings in a beautiful format.
6. Plot and save detected patterns to the local file system.

Prerequisites:
Make sure your virtual environment is activated and dependencies are installed.
Run:
    python examples/run_okx_example.py
"""

import sys
import os
import time
import logging
import pandas as pd
import ccxt

# Import the library modules
from harmonic_patterns import HarmonicDetector, kline_to_df

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('HarmonicExample')

def run_scanner():
    # 1. Initialize OKX Client
    logger.info("Initializing OKX client...")
    okx = ccxt.okx()
    okx.enableRateLimit = True  # Enable rate limiting to respect exchange policies
    
    # 2. Instantiate the detector
    # error_allowed: error rate (0.05 is standard 5%)
    # strict: True forces pattern ratios to align closely with standard Fibonacci values
    detector = HarmonicDetector(error_allowed=0.05, strict=True)
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h', '1d']
    limit = 150  # Fetch 150 candles to keep the plot clean and readable
    
    output_dir = './data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info("========== OKX Harmonic Pattern Scanner Started ==========")
    
    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"Fetching candles for {symbol} on {tf} timeframe...")
            try:
                # Fetch historical OHLCV data
                ohlcv = okx.fetch_ohlcv(symbol, tf, limit=limit)
                df = kline_to_df(ohlcv)
                
                logger.info(f"Running detection algorithm for {symbol} {tf}...")
                
                # Check for patterns
                # save_fig_name will save the image if a pattern is found
                # return_dt_idx=True converts pattern node indices to datetime timestamps
                save_filename = f"{symbol.replace('/', '_')}_{tf}_pattern.png"
                
                patterns, predict_patterns = detector.search_patterns(
                    df, 
                    only_last=False, 
                    last_n=4, 
                    plot=True, 
                    predict=True,
                    return_dt_idx=True,
                    save_fig_name=save_filename
                )
                
                # Log Completed Patterns
                if patterns:
                    logger.info(f"✅ Found {len(patterns)} COMPLETED patterns for {symbol} {tf}:")
                    for idx, pat in enumerate(patterns):
                        direction = "Bullish" if pat[0] == 1 else "Bearish"
                        timeline = " -> ".join([d.strftime('%Y-%m-%d %H:%M') for d in pat[1]])
                        logger.info(f"  [{idx+1}] Pattern: {pat[3]} ({direction})")
                        logger.info(f"      Timeline: {timeline}")
                        logger.info(f"      Ratios: {pat[2]}")
                        logger.info(f"      Chart saved as: {os.path.join(output_dir, save_filename)}")
                else:
                    logger.info(f"No completed patterns found for {symbol} {tf}.")
                
                # Log Predicting Patterns (under construction)
                if predict_patterns:
                    logger.info(f"🔮 Found {len(predict_patterns)} PREDICTING patterns for {symbol} {tf}:")
                    for idx, pat in enumerate(predict_patterns):
                        direction = "Bullish (Predicting)" if pat[0] == 1 else "Bearish (Predicting)"
                        timeline = " -> ".join([d.strftime('%Y-%m-%d %H:%M') for d in pat[1]])
                        logger.info(f"  [{idx+1}] Pattern: {pat[3]} ({direction})")
                        logger.info(f"      Timeline: {timeline}")
                        logger.info(f"      Target D Price: {pat[3].get('predict_D', 'N/A')}")
                        logger.info(f"      Chart saved as: {os.path.join(output_dir, save_filename)}")
                else:
                    logger.info(f"No predicting patterns found for {symbol} {tf}.")
                    
            except Exception as e:
                logger.error(f"Failed to scan {symbol} {tf}: {e}")
                
            # Sleep briefly to avoid aggressive requests
            time.sleep(1)
            
    logger.info("========== OKX Harmonic Pattern Scanner Completed ==========")

if __name__ == '__main__':
    run_scanner()
