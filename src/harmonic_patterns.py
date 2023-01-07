
"""
Author Djoffrey
some harmonic pattern scanner functions
(TODO: calculate using (high+low)/2 or etc, not just close price)
(TODO: should return each peak distance)
"""

import pandas as pd
import numpy as np
import talib.abstract as ta
# from scipy.signal import argrelextrema
import mplfinance as mpf

MAIN_FIBB_RATIOS = [0.618, 1.618]
SECOND_FIBB_RATIOS = [0.786, 0.886, 1.13, 1.27]
ALT_FIBB_RATIOS = [0.382, 0.5, 0.707, 1.41, 2.0, 2.24, 2.618, 3.14, 3.618]
AB_CD = [1.27, 1.618, 2.236]

FIBB_RATIOS = [*MAIN_FIBB_RATIOS, *SECOND_FIBB_RATIOS, *ALT_FIBB_RATIOS]

from IPython.core.debugger import set_trace
import os


def kline_to_df(arr) -> pd.DataFrame:
    kline = pd.DataFrame(
        arr,
        columns=['ts', 'open', 'high', 'low', 'close', 'volume' ])
    kline.index = pd.to_datetime(kline.ts, unit='ms')
    kline.drop('ts', axis=1, inplace=True)
    return kline


class HarmonicDetector(object):
    def __init__(self, error_allowed:float=0.05, strict:bool=True, predict_err_rate:float=None):
        self.err = error_allowed
        self.predict_err_rate = self.err if predict_err_rate is None else predict_err_rate

        self.strict = strict

    def is_eq(self, n: float, m: float, err:float=None, l_closed:bool=False, r_closed:bool=True) -> bool:
        _err = self.err if err is None else err
        left = m if l_closed else m * (1 - _err)
        right = m if r_closed else m * (1 + _err)
        return (n >= left) and (n <= right)

    def is_in(self, n: float, l: float, r: float, err:float=None, l_closed:bool=True, r_closed:bool=True) -> bool:
        _err = self.err if err is None else err
        left = l if l_closed else l * (1 - _err)
        right = r if r_closed else r * (1 + _err)
        if self.strict:
            fibb_rates = [
                self.is_eq(n, f_rate) for f_rate in FIBB_RATIOS
                if (f_rate >= left) and (f_rate <= right)
            ]
            return np.any(fibb_rates)
        else:
            return (n >= left) and (n <= right)

    def get_zigzag(self, df: pd.DataFrame, period: int):

        # translated from https://www.tradingview.com/script/mRbjBGdL-Double-Zig-Zag-with-HHLL/

        zigzag_pattern = []
        direction = 0
        changed = False
        for idx in range(1, len(df)):
            highest_high = ta.MAX(df.high[:idx], timeperiod=period)[-1]
            lowest_low = ta.MIN(df.low[:idx], timeperiod=period)[-1]


            new_high = df.high[idx] >= highest_high
            new_low = df.low[idx] <= lowest_low

            if new_high and not new_low:
                if direction != 1:
                    direction = 1
                    changed = True
                elif direction == 1:
                    changed = False
            elif not new_high and new_low:
                if direction != -1:
                    direction = -1
                    changed = True
                elif direction == -1:
                    changed = False

            if new_high or new_low:
                if changed or len(zigzag_pattern)==0:
                    if direction == 1:
                        pat = ['H', df.high[idx], idx]
                        zigzag_pattern.append(pat)
                    elif direction == -1:
                        pat = ['L', df.low[idx], idx]
                        zigzag_pattern.append(pat)
                else:
                    if direction == 1 and df.high[idx] > zigzag_pattern[-1][1]:
                        pat = ['H', df.high[idx], idx]
                        zigzag_pattern[-1] = pat
                    elif direction == -1 and df.low[idx] < zigzag_pattern[-1][1]:
                        pat = ['L', df.low[idx], idx]
                        zigzag_pattern[-1] = pat
                    else:
                        pass
        return zigzag_pattern

    def detect_abcd(self, current_pat: list, predict:bool=False, predict_mode:str='direct'):
        """
        AB=CD is a common pattern in harmonic trading, it is many pattern's inner structure
        AB=CD has two main alternatives 1.27AB=CD and 1.618AB=CD
        """
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'AB': ABC,
                'BC': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(ABC, 0.382, 0.886), # AB
                self.is_in(BCD, 1.13, 1.168), # CD
                self.is_eq(ABCD, 1) # strictly 1/AB = CD
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        elif predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            #XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(ABC, 0.382, 0.886) # AB
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = C - (C - B) * (1 / ABC)
                else:
                    D = C + (B - C) * (1 / ABC)
                BCD = abs(C-D) / abs(B-C)
                #XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 1.13, 2.168)
                ABCD = BCD / (1/ABC)
                ret_dict = {
                    'AB': ABC,
                    'CD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        elif predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            #XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                #self.is_eq(XAB, 0.618), # LEG 1
                self.is_in(ABC, 0.382, 0.886) # AB
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = C - (C - B)*0.786
                else:
                    D = C + (B - C)*0.786
                BCD = abs(C-D) / abs(B-C)
                #XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 1.13, 2.168)
                ABCD = BCD / (1/ABC)
                ret_dict = {
                    'AB': ABC,
                    'CD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None
            else:
                return None

    def detect_gartley(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_eq(XAB, 0.618), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 1.13, 1.168), # LEG 3
                self.is_eq(XAD, 0.786), # LEG 4
                self.is_eq(ABCD, 1) # AB=CD
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        elif predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.618), # LEG 1
                self.is_in(ABC, 0.382, 0.886) # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.786
                else:
                    D = A + (X - A)*0.786
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 1.13, 1.168),
                    self.is_eq(ABCD, 1) # AB=CD
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        elif predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.618), # LEG 1
                self.is_in(ABC, 0.382, 0.886) # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.786
                else:
                    D = A + (X - A)*0.786
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 1.13, 1.168),
                    self.is_eq(ABCD, 1) # AB=CD
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1

                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None
            else:
                return None

    def detect_bat(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(XAB, 0.382, 0.5), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 1.618, 2.168), # LEG 3
                self.is_eq(XAD, 0.886), # LEG 4
                self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.5), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.886
                else:
                    D = A + (X - A)*0.886
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 1.618, 2.168),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.5), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.886
                else:
                    D = A + (X - A)*0.886
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 1.618, 2.168),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
                ]))
                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None
            else:
                return None
    def detect_altbat(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_eq(XAB, 0.382), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 2, 3.168), # LEG 3
                self.is_eq(XAD, 1.13) # LEG 4
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.382), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.13
                else:
                    D = A + (X - A)*1.13
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 2, 3.168)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None

        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.382), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.13
                else:
                    D = A + (X - A)*1.13
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 2, 3.168)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1

                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None

            else:
                return None

    def detect_butterfly(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_eq(XAB, 0.786), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 1.618, 2.24), # LEG 3
                self.is_eq(XAD, 1.27), # LEG 4
                self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.786), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.27
                else:
                    D = A + (X - A)*1.27
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 1.618, 2.24),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.786), # LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.27
                else:
                    D = A + (X - A)*1.27
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 1.618, 2.24)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None

            else:
                return None

    def detect_crab(self, current_pat: list, predict: bool=False,  predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)
            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(XAB, 0.382, 0.618),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 2.618, 3.618), # LEG 3
                self.is_eq(XAD, 1.618), # LEG 4
                self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        elif predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.618),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.618
                else:
                    D = A + (X - A)*1.618
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 2.618, 3.618),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
                ]))


                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        elif predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.618),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.618
                else:
                    D = A + (X - A)*1.618
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 2.618, 3.618),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False

                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None
            else:
                return None
    def detect_deepcrab(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_eq(XAB, 0.886),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
                self.is_in(BCD, 2.24, 3.618), # LEG 3
                self.is_eq(XAD, 1.618), # LEG 4
                self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.886),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.618
                else:
                    D = A + (X - A)*1.618
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 2.24, 3.618),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_eq(XAB, 0.886),# LEG 1
                self.is_in(ABC, 0.382, 0.886), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.618
                else:
                    D = A + (X - A)*1.618
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_in(BCD, 2.24, 3.618),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 2.236), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None

            else:
                return None

    def detect_shark(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(XAB, 0.5, 0.886), # LEG 1
                self.is_in(ABC, 1.13, 1.618), # LEG 2
                self.is_in(BCD, 1.618, 2.24), # LEG 3
                self.is_in(XAD, 0.886, 1.13) # LEG 4
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        elif predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.5, 0.886), # LEG 1
                self.is_in(ABC, 1.13, 1.618), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.886
                else:
                    D = A + (X - A)*0.886
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 1.618, 2.24)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        elif predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.5, 0.886), # LEG 1
                self.is_in(ABC, 1.13, 1.618), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*0.886
                else:
                    D = A + (X - A)*0.886
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                pattern_predict_s2 = self.is_in(BCD, 1.618, 2.24)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1

                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None
            else:
                return None
    def detect_5o(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)

            ABCD = BCD / (1/ABC)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'AB=CD': ABCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(XAB, 1.13, 1.618), # LEG 1
                self.is_in(ABC, 1.618, 2.24), # LEG 2
                self.is_eq(BCD, 0.5), # LEG 3
                #self.is_eq(XAD, 0.5), # LEG 4
                self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 1.13, 1.618), # LEG 1
                self.is_in(ABC, 1.618, 2.24), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.27
                else:
                    D = A + (X - A)*1.27
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_eq(BCD, 0.5),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                if pattern_predict_s2:
                    return [direction, ret_dict]
                else:
                    return None
        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 1.13, 1.618), # LEG 1
                self.is_in(ABC, 1.618, 2.24), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = A - (A - X)*1.27
                else:
                    D = A + (X - A)*1.27
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                ABCD = BCD / (1/ABC)
                pattern_predict_s2 = np.all(np.array([
                    self.is_eq(BCD, 0.5),
                    self.is_eq(ABCD, 1) or self.is_in(ABCD, 0.786, 1.618), # AB=CD and its alternatives
                ]))

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'AB=CD': ABCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s2 and pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None

            else:
                return None
    def detect_cypher(self, current_pat: list, predict: bool=False, predict_mode:str='direct'):
        # current_pat: [['H', new_high, idx],...]
        # Legs
        if not predict:
            X, A, B, C, D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            BCD = abs(C-D) / abs(B-C)
            XCD = abs(D-C) / abs(X-C)

            ret_dict = {
                'XAB': XAB,
                'XAD': XAD,
                'ABC': ABC,
                'BCD': BCD,
                'XCD': XCD
            }

            # Detect
            pattern_found = np.all(np.array([
                self.is_in(XAB, 0.382, 0.786), # LEG 1
                self.is_in(ABC, 1.272, 1.414), # LEG 2
                #self.is_in(BCD, 1.618, 2.24), # LEG 3
                self.is_eq(XCD, 0.786) or self.is_eq(XAD, 0.786) ,  # LEG 4
            ]))
            direction = 1 if D<C else -1
            if pattern_found:
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='reverse':
            _, X, A, B, C = [pat[1] for pat in current_pat]
            last_direction = current_pat[-1][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)
            #XCD = abs(D-C) / abs(X-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.786), # LEG 1
                self.is_in(ABC, 1.272, 1.414), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = C - (C - X)*0.786
                else:
                    D = C + (X - C)*0.786
                XCD = abs(D-C) / abs(X-C)
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                # pattern_predict_s2 = self.is_eq(BCD, 0.786)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'XCD': XCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                return [direction, ret_dict]
            else:
                return None
        if predict_mode=='direct':
            X, A, B, C, origin_D = [pat[1] for pat in current_pat]
            last_direction = current_pat[-2][0]
            XAB = abs(B-A) / abs(X-A)
            #XAD = abs(A-D) / abs(X-A)
            ABC = abs(B-C) / abs(A-B)
            #BCD = abs(C-D) / abs(B-C)
            #XCD = abs(D-C) / abs(X-C)

            # Detect
            pattern_predict_s1 = np.all(np.array([
                self.is_in(XAB, 0.382, 0.786), # LEG 1
                self.is_in(ABC, 1.272, 1.414), # LEG 2
            ]))
            if pattern_predict_s1:
                if last_direction == 'H':
                    # XAD
                    D = C - (C - X)*0.786
                else:
                    D = C + (X - C)*0.786
                XCD = abs(D-C) / abs(X-C)
                BCD = abs(C-D) / abs(B-C)
                XAD = abs(A-D) / abs(X-A)
                # pattern_predict_s2 = self.is_eq(BCD, 0.786)

                ret_dict = {
                    'XAB': XAB,
                    'XAD': XAD,
                    'ABC': ABC,
                    'BCD': BCD,
                    'XCD': XCD,
                    'predict_D': D
                }
                direction = 1 if D<C else -1
                pattern_predict_s3 = False
                if (direction == 1 and D < origin_D) or \
                   (direction == -1 and D > origin_D):
                    pattern_predict_s3 = True

                if pattern_predict_s3:
                    return [direction, ret_dict]
                else:
                    return None

            else:
                return None

    def detect_pattern(self, zigzag_pattern):
        res = []
        # price default to close
        for idx in range(0, len(zigzag_pattern) - 5 + 1):
            current_pat = [pat for pat in zigzag_pattern[idx:idx+5]]
            current_idx = [pat[2] for pat in zigzag_pattern[idx:idx+5]]
            detect_funcions = [
                (self.detect_gartley, 'gartley'),
                (self.detect_bat, 'bat'),
                (self.detect_altbat, 'altbat'),
                (self.detect_butterfly, 'butterfly'),
                (self.detect_crab, 'crab'),
                (self.detect_deepcrab, 'deepcrab'),
                (self.detect_shark, 'shark'),
                (self.detect_5o, '5o'),
                (self.detect_cypher, 'cypher'),
                (self.detect_abcd, 'abcd'),
            ]
            if len(current_pat) == 5:
                for func, func_name in detect_funcions:
                    r = func(current_pat)
                    if r is not None:
                        direction, ret_dict = r
                        bull_or_bear = 'bullish' if direction==1 else 'bearish'
                        label = f'{bull_or_bear} {func_name}'
                        res.append([current_pat, current_idx, label, ret_dict])
        return res

    def predict_pattern(self, zigzag_pattern):
        res = []
        # predict reverse pattern
        current_pat = zigzag_pattern[-5:]
        current_idx = [pat[2] for pat in zigzag_pattern[-5:]]

        detect_funcions = [
            (self.detect_gartley, 'gartley'),
            (self.detect_bat, 'bat'),
            (self.detect_altbat, 'altbat'),
            (self.detect_butterfly, 'butterfly'),
            (self.detect_crab, 'crab'),
            (self.detect_deepcrab, 'deepcrab'),
            (self.detect_shark, 'shark'),
            (self.detect_5o, '5o'),
            (self.detect_cypher, 'cypher'),
        ]
        if len(current_pat) == 5:
            for func, func_name in detect_funcions:
                r = func(current_pat, predict=True, predict_mode='reverse')
                if r is not None:
                    direction, ret_dict = r
                    _, X, A, B, C = [pat[1] for pat in current_pat]
                    D = ret_dict['predict_D']
                    bull_or_bear = 'bullish' if direction==1 else 'bearish'
                    label = f'{bull_or_bear} {func_name} predict next D: {D}'
                    # Funny ha? but shit happens
                    if D > 0:
                        p_pattern = [X, A, B, C, D]
                        p_idx = [*current_idx[1:], -1]
                        res.append([p_pattern, p_idx, label, ret_dict])

            for func, func_name in detect_funcions:
                r = func(current_pat, predict=True, predict_mode='direct')
                if r is not None:
                    direction, ret_dict = r
                    X, A, B, C, origin_D = [pat[1] for pat in current_pat]
                    D = ret_dict['predict_D']
                    bull_or_bear = 'bullish' if direction==1 else 'bearish'
                    label = f'{bull_or_bear} {func_name} predict current D: {D}'
                    # Funny ha? but shit happens
                    if D > 0:
                        p_pattern = [X, A, B, C, D]
                        res.append([p_pattern, current_idx, label, ret_dict])
        return res

    def get_patterns(self, df: pd.DataFrame, window: int, predict: bool=False, plot:bool=False):
        zigzag_pattern = self.get_zigzag(df, window)
        patterns = self.detect_pattern(zigzag_pattern)
        if plot:
            points = [(df.index[k[2]], k[1]) for k in zigzag_pattern]
            patterns_line = [list(zip(
                [df.index[dt_idx] for dt_idx in pat[1]],
                [val_idx[1] for val_idx in pat[0]]
            )) for pat in patterns]
            mpf.plot(df,type='candle', alines=dict(alines=patterns_line, colors=['b','r','c','k','g']))
        if predict:
            predict_res = self.predict_pattern(zigzag_pattern)
            return patterns, predict_res
        return patterns, None

    def filter_duplicats(self, patterns, predict:bool=False):
        pat_set = set()
        ret = []
        if not predict:
            for pat in patterns:
                X,A,B,C,D = pat[1]
                pat_str = f'{X}-{A}-{B}-{C}-{D}'
                if pat_str not in pat_set:
                    ret.append(pat)
                    pat_set.add(pat_str)
        else:
            # predict
            for pat in patterns:
                X,A,B,C,D = pat[1] # get index
                label = pat[2] # get_label
                #predict_D = pat[3]['predict_D'] # get_predict_D
                pat_str = f'{X}-{A}-{B}-{C}-{D}-{label}'
                if pat_str not in pat_set:
                    ret.append(pat)
                    pat_set.add(pat_str)
        return ret

    def plot_patterns(self, df, patterns, predict_patterns, plot_predict:bool=True, save_fig:bool=False, file_name:str=None, file_data_path='./data/'):
        """
        plot results, if save_fig, will save_fig to {file_name}
        """
        # points = [(df.index[k[2]], k[1]) for k in zigzag_pattern]
        patterns_line = [list(zip(
            [df.index[dt_idx] for dt_idx in pat[1]],
            [val_idx[1] for val_idx in pat[0]]
        )) for pat in patterns if pat[2][-4:]!='abcd']
        if plot_predict:
            new_index = df.index.append(pd.Index([
                df.index[-1] + 10*(df.index[-1] - df.index[-2])
            ]))
            df = pd.DataFrame(df, index=new_index)
            df.fillna(0)
            patterns_predict_line = [list(zip(
                [df.index[dt_idx] for dt_idx in pat[1]],
                [val_idx for val_idx in pat[0]]
            )) for pat in predict_patterns if pat[2][-4:]!='abcd']

            predict_Ds = [pat[3]['predict_D'] for pat in predict_patterns]
            patterns_line.extend(patterns_predict_line)
        else:
            predict_Ds = []

        mc = mpf.make_marketcolors(base_mpf_style='yahoo')
        # Create a style based on `seaborn` using those market colors:
        style  = mpf.make_mpf_style(base_mpl_style='seaborn', marketcolors=mc)

        if len(patterns_line) > 0:
            if save_fig and file_name is not None:
                if not os.path.exists(file_data_path):
                    os.mkdir(file_data_path)
                file_path = file_data_path + file_name
                mpf.plot(df,
                         type='candle',
                         mav=(10,21,55,120),
                         style=style,
                         alines=dict(alines=patterns_line,
                                     colors=['b','r','c','k','g'],
                                     linewidths=3,
                                     alpha=0.5
                                     ),
                         hlines=dict(hlines=predict_Ds, colors=['g'],linestyle='-.'),
                         savefig=file_path
                     )
            else:
                mpf.plot(df,
                         type='candle',
                         mav=(10,21,55,120),
                         style=style,
                         alines=dict(alines=patterns_line,
                                     colors=['b','r','c','k','g'],
                                     linewidths=3,
                                     alpha=0.5
                                     ),
                         hlines=dict(hlines=predict_Ds, colors=['g'],linestyle='-.'),
                         #savefig='test.svg'
                         )

    def search_patterns(self, df: pd.DataFrame, predict: bool=False, only_last:bool=False, last_n:int=0, plot:bool=False, plot_predict:bool=True, return_dt_idx=True, save_fig_name:str=None):
        assert last_n < 5
        all_patterns = list()
        all_predict_patterns = list()
        for window in [8, 13, 21, 34, 55]:
            zigzag_pattern = self.get_zigzag(df, window)
            if predict:
                ori_err = self.err
                self.err = self.predict_err_rate
                predict_res = self.predict_pattern(zigzag_pattern)
                all_predict_patterns.extend(predict_res)
                self.err = ori_err
            patterns = self.detect_pattern(zigzag_pattern)
            if only_last:
                all_patterns.extend(patterns[-1:])
            else:
                all_patterns.extend(patterns)

        if only_last and last_n>0:
            all_patterns = [pat for pat in all_patterns if pat[0][-1][2] == len(df) - last_n]

        all_patterns = self.filter_duplicats(all_patterns)
        all_predict_patterns = self.filter_duplicats(all_predict_patterns, predict=True)
        if plot:
            if save_fig_name is not None:
                self.plot_patterns(df, all_patterns, all_predict_patterns, save_fig=True, file_name=save_fig_name)
            else:
                self.plot_patterns(df, all_patterns, all_predict_patterns, save_fig=False)

        if return_dt_idx:
            all_patterns = [
                [pat[0],
                 [df.index[dt_idx] for dt_idx in pat[1]],
                 pat[2],
                 pat[3]]
                for pat in all_patterns]
            all_predict_patterns = [
                [pat[0],
                 [df.index[dt_idx] for dt_idx in pat[1]],
                 pat[2],
                 pat[3]]
                for pat in all_predict_patterns]

        return all_patterns, all_predict_patterns


if __name__ == '__main__':
    detector = HarmonicDetector()
    print('done')
