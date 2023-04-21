import numpy as np
import pandas as pd 
from pandas import DataFrame, Series

""" 
           .__.__                
      ____ |__|  |  __ _____  ___
     /    \|  |  | |  |  \  \/  /
    |   |  \  |  |_|  |  />    < 
    |___|  /__|____/____//__/\_ \
         \/                    \/

    Legendary TA Collection
    
    A collection of indicators ported from MT4 and Pine
                                           
     ✓ No lookahead, repainting or rewriting of candles
     ✓ Tested with Freqtrade and FreqAI
     ✓ More to come!
    
    Telegram >>> @nlxtrading
    
    
    Usage:
    
            import legendary_ta as lta

            # Fisher Stochastic Center of Gravity
            dataframe = lta.fisher_cg(dataframe)
            # dataframe["fisher_cg"]                    Float (Center Line)
            # dataframe["fisher_trigger"]               Float (Trigger Line)
            
            # Leledc Exhaustion Bars
            dataframe = lta.exhaustion_bars(dataframe)
            # dataframe["lelec"]                        1 (Up) / 0 (Neutral) / -1 (Down)
            
            # Pinbar Reversals
            dataframe = lta.pinbar(dataframe, smi)
            # dataframe["pinbar_buy"]                   Bool
            # dataframe["pinbar_sell"]                  Bool
            
            # Breakouts and Retests
            dataframe = lta.breakouts(dataframe)
            # dataframe['support_level']                Float (Support)
            # dataframe['resistance_level']             Float (Resistance)
            # dataframe['support_breakout']             Bool
            # dataframe['resistance_breakout']          Bool
            # dataframe['support_retest']               Bool
            # dataframe['potential_support_retest']     Bool
            # dataframe['resistance_retest']            Bool
            # dataframe['potential_resistance_retest']  Bool
            
            # Stochastic Momentum Index
            dataframe = lta.smi_momentum(dataframe)
            # dataframe["smi"]                          Float (SMI)
            
"""

def fisher_cg(df: DataFrame, length=20, min_period=10):
    """ 
    Fisher Stochastic Center of Gravity
    
    Original Pinescript by dasanc
    https://tradingview.com/script/5BT3a9mJ-Fisher-Stochastic-Center-of-Gravity/
        
    :return: DataFrame with fisher_cg and fisher_sig column populated
    """
    
    df['hl2'] = (df['high'] + df['low']) / 2
    
    if length < min_period:
        length = min_period

    num = 0.0
    denom = 0.0
    CG = 0.0
    MaxCG = 0.0
    MinCG = 0.0
    Value1 = 0.0
    Value2 = 0.0
    Value3 = 0.0

    for i in range(length):
        num += (1 + i) * df['hl2'].shift(i)
        denom += df['hl2'].shift(i)

    CG = -num / denom + (length + 1) / 2
    MaxCG = CG.rolling(window=length).max()
    MinCG = CG.rolling(window=length).min()

    Value1 = np.where(MaxCG != MinCG, (CG - MinCG) / (MaxCG - MinCG), 0)
    Value2 = (4 * Value1 + 3 * np.roll(Value1, 1) + 2 * np.roll(Value1, 2) + np.roll(Value1, 3)) / 10
    Value3 = 0.5 * np.log((1 + 1.98 * (Value2 - 0.5)) / (1 - 1.98 * (Value2 - 0.5)))

    df['fisher_cg'] = pd.Series(Value3) # Center of Gravity
    df['fisher_sig'] = pd.Series(Value3).shift(1) # Signal / Trigger

    return df


def exhaustion_bars(df: DataFrame, lookback=6, length=30) -> DataFrame:
    """ 
    NLX Leledc Exhaustion Bars
    Infamous S/R Reversal Indicator for Scalping
    
    Inpired by glaz https://www.tradingview.com/script/2rZDPyaC-Leledc-Exhaustion-Bar/ 
    Original (MT4) https://www.abundancetradinggroup.com/leledc-exhaustion-bar-mt4-indicator/

    :return: DataFrame with columns populated
    """
    
    df['resistance'] = np.nan
    df['support'] = np.nan
    df['lelec'] = 0

    bindex = np.zeros(len(df))
    sindex = np.zeros(len(df))

    for i in range(4, len(df)):
        bindex[i] = bindex[i - 1] + 1 if df.loc[i, 'close'] > df.loc[i - 4, 'close'] else bindex[i - 1]
        sindex[i] = sindex[i - 1] + 1 if df.loc[i, 'close'] < df.loc[i - 4, 'close'] else sindex[i - 1]

        is_bearish_engulfing = (
            bindex[i] > lookback and
            df.loc[i, 'close'] < df.loc[i, 'open'] and
            df.loc[i, 'high'] >= df.loc[i - length:i, 'high'].max()
        )
        is_bullish_engulfing = (
            sindex[i] > lookback and
            df.loc[i, 'close'] > df.loc[i, 'open'] and
            df.loc[i, 'low'] <= df.loc[i - length:i, 'low'].min()
        )

        if is_bearish_engulfing:
            bindex[i] = 0
            df.loc[i, 'lelec'] = -1
            df.loc[i, 'resistance'] = df.loc[i, 'high']
        elif is_bullish_engulfing:
            sindex[i] = 0
            df.loc[i, 'lelec'] = 1
            df.loc[i, 'support'] = df.loc[i, 'low']
        else:
            if df.loc[i - 1, 'lelec'] == -1 and df.loc[i, 'close'] > df.loc[i - 1, 'resistance']:
                df.loc[i, 'lelec'] = 0
            elif df.loc[i - 1, 'lelec'] == 1 and df.loc[i, 'close'] < df.loc[i - 1, 'support']:
                df.loc[i, 'lelec'] = 0
            else:
                df.loc[i, 'lelec'] = df.loc[i - 1, 'lelec']

            df.loc[i, 'resistance'] = df.loc[i - 1, 'resistance']
            df.loc[i, 'support'] = df.loc[i - 1, 'support']

    return df


def breakouts(df: DataFrame, length=20):
    """ 
    S/R Breakouts and Retests
    
    Makes it easy to work with Support and Resistance 
    Find Retests, Breakouts and the next levels 
    
    :return: DataFrame with event columns populated
    """
    
    high = df['high']
    low = df['low']
    close = df['close']

    pl = low.rolling(window=length*2+1, center=True).min()
    ph = high.rolling(window=length*2+1, center=True).max()
    
    s_yLoc = low.shift(length + 1).where(low.shift(length + 1) > low.shift(length - 1), low.shift(length - 1))
    r_yLoc = high.shift(length + 1).where(high.shift(length + 1) > high.shift(length - 1), high.shift(length + 1))

    cu = close < s_yLoc.shift(length)
    co = close > r_yLoc.shift(length)

    s1 = (high >= s_yLoc.shift(length)) & (close <= pl.shift(length))
    s2 = (high >= s_yLoc.shift(length)) & (close >= pl.shift(length)) & (close <= s_yLoc.shift(length))
    s3 = (high >= pl.shift(length)) & (high <= s_yLoc.shift(length))
    s4 = (high >= pl.shift(length)) & (high <= s_yLoc.shift(length)) & (close < pl.shift(length))

    r1 = (low <= r_yLoc.shift(length)) & (close >= ph.shift(length))
    r2 = (low <= r_yLoc.shift(length)) & (close <= ph.shift(length)) & (close >= r_yLoc.shift(length))
    r3 = (low <= ph.shift(length)) & (low >= r_yLoc.shift(length))
    r4 = (low <= ph.shift(length)) & (low >= r_yLoc.shift(length)) & (close > ph.shift(length))

    # Events
    df['support_level'] = pl.diff().where(pl.diff().notna())
    df['resistance_level'] = ph.diff().where(ph.diff().notna())
    
    # Use the last S/R levels instead of nan
    df['support_level'] = df['support_level'].combine_first(df['support_level'].shift())
    df['resistance_level'] = df['resistance_level'].combine_first(df['resistance_level'].shift())
    
    df['support_breakout'] = cu
    df['resistance_breakout'] = co
    df['support_retest'] = s1 | s2 | s3 | s4
    df['potential_support_retest'] = s1 | s2 | s3
    df['resistance_retest'] = r1 | r2 | r3 | r4
    df['potential_resistance_retest'] = r1 | r2 | r3
    
    return df


def pinbar(df: DataFrame, smi=None):
    """ 
    Pinbar - Price Action Indicator
    
    Pinbars are an easy but sure indication
    of incoming price reversal. 
    Signal confirmation with SMI.

    :return: DataFrame with buy / sell signals columns populated
    """
    
    low = df['low']
    high = df['high']
    close = df['close']
    
    tr = true_range(df)
    
    if smi is None:
        df = smi_momentum(df)
        smi = df['smi']
    
    df['pinbar_sell'] = (
        (high < high.shift(1)) &
        (close < high - (tr * 2 / 3)) &
        (smi < smi.shift(1)) &
        (smi.shift(1) > 40) &
        (smi.shift(1) < smi.shift(2))
    )

    df['pinbar_buy'] = (
        (low > low.shift(1)) &
        (close > low + (tr * 2 / 3)) &
        (smi.shift(1) < -40) &
        (smi > smi.shift(1)) &
        (smi.shift(1) > smi.shift(2))
    )
    
    return df


def smi_momentum(df: DataFrame, k_length=9, d_length=3):
    """     
    The Stochastic Momentum Index (SMI) Indicator was developed by 
    William Blau in 1993 and is considered to be a momentum indicator 
    that can help identify trend reversal points
        
    :return: DataFrame with smi column populated
    """
    
    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    df['smi'] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    
    return df


"""
Misc. Helper Functions
"""

def linear_growth(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear growth function. Grows from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (end - start) / (end_time - start_time)

    return min(end, start + (rate * time))


def linear_decay(start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
    """
    Simple linear decay function. Decays from start to end after end_time minutes (starts after start_time minutes)
    """
    time = max(0, trade_time - start_time)
    rate = (start - end) / (end_time - start_time)

    return max(end, start - (rate * time))

def true_range(dataframe):
    prev_close = dataframe['close'].shift()
    tr = pd.concat([dataframe['high'] - dataframe['low'], abs(dataframe['high'] - prev_close), abs(dataframe['low'] - prev_close)], axis=1).max(axis=1)
    return tr