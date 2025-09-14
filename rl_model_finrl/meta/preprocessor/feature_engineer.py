import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Union, Optional

class FeatureEngineer:
    """
    金融特征工程类
    
    用于计算和添加金融技术指标和特征
    """
    
    def __init__(self, 
                 use_technical_indicators: bool = True,
                 use_vix: bool = False,
                 use_turbulence: bool = False,
                 use_sentiment: bool = False):
        """
        初始化特征工程器
        
        参数:
            use_technical_indicators: 是否使用技术指标
            use_vix: 是否使用波动率指数
            use_turbulence: 是否使用市场波动指标
            use_sentiment: 是否使用情绪指标
        """
        self.use_technical_indicators = use_technical_indicators
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.use_sentiment = use_sentiment
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据并添加特征
        
        参数:
            df: 原始数据DataFrame
            
        返回:
            处理后的DataFrame
        """
        data = df.copy()
        
        # 添加技术指标
        if self.use_technical_indicators:
            data = self.add_technical_indicators(data)
        
        # 添加波动率指数
        if self.use_vix:
            data = self.add_vix(data)
        
        # 添加市场波动指标
        if self.use_turbulence:
            data = self.add_turbulence(data)
        
        # 添加情绪指标
        if self.use_sentiment:
            data = self.add_sentiment(data)
        
        # 填充NaN值
        data = self.fill_missing_values(data)
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        参数:
            data: 价格数据DataFrame
            
        返回:
            添加技术指标后的DataFrame
        """
        df = data.copy()
        
        # 确保列名标准化
        price_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        # 使用pandas_ta计算技术指标
        # MACD
        macd_df = ta.macd(df[price_col], fast=12, slow=26, signal=9)
        df['macd'] = macd_df['MACD_12_26_9']
        df['macd_signal'] = macd_df['MACDs_12_26_9']
        df['macd_hist'] = macd_df['MACDh_12_26_9']

        # RSI
        df['rsi_6'] = ta.rsi(df[price_col], length=6)
        df['rsi_14'] = ta.rsi(df[price_col], length=14)
        df['rsi_30'] = ta.rsi(df[price_col], length=30)

        # CCI
        df['cci'] = ta.cci(df[high_col], df[low_col], df[price_col], length=14)

        # ADX
        adx_df = ta.adx(df[high_col], df[low_col], df[price_col], length=14)
        df['adx'] = adx_df['ADX_14']

        # 布林带
        bbands_df = ta.bbands(df[price_col], length=20, std=2)
        df['boll_upper'] = bbands_df['BBU_20_2.0']
        df['boll_middle'] = bbands_df['BBM_20_2.0']
        df['boll_lower'] = bbands_df['BBL_20_2.0']

        # ATR
        df['atr'] = ta.atr(df[high_col], df[low_col], df[price_col], length=14)

        # 移动平均线
        df['sma_5'] = ta.sma(df[price_col], length=5)
        df['sma_10'] = ta.sma(df[price_col], length=10)
        df['sma_20'] = ta.sma(df[price_col], length=20)
        df['sma_60'] = ta.sma(df[price_col], length=60)

        # WILLR
        df['willr'] = ta.willr(df[high_col], df[low_col], df[price_col], length=14)

        # ROC
        df['roc'] = ta.roc(df[price_col], length=10)

        # OBV
        df['obv'] = ta.obv(df[price_col], df[volume_col])
        
        # 涨跌幅
        df['daily_return'] = df[price_col].pct_change()
        
        # 波动率
        df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def add_vix(self, data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        添加VIX波动率指数
        
        参数:
            data: 原始数据DataFrame
            vix_data: VIX数据DataFrame（可选）
            
        返回:
            添加VIX后的DataFrame
        """
        df = data.copy()
        
        if vix_data is not None:
            # 如果提供了VIX数据，合并到主数据中
            vix_data = vix_data.rename(columns={'close': 'vix'})
            df = pd.merge(df, vix_data[['vix']], 
                          left_index=True, right_index=True, 
                          how='left')
        else:
            # 如果没有VIX数据，使用收益率的滚动波动率作为VIX代理
            price_col = 'close' if 'close' in df.columns else 'Close'
            returns = df[price_col].pct_change()
            df['vix'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        return df
    
    def add_turbulence(self, data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        添加市场波动指标
        
        参数:
            data: 原始数据DataFrame
            window: 计算窗口
            
        返回:
            添加波动指标后的DataFrame
        """
        df = data.copy()
        
        # 计算收益率
        price_col = 'close' if 'close' in df.columns else 'Close'
        df['return'] = df[price_col].pct_change()
        
        # 计算波动指标
        df['turbulence'] = df['return'].rolling(window=window).apply(
            lambda x: np.sum(np.square(x - x.mean())) / len(x)
        )
        
        return df
    
    def add_sentiment(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        添加市场情绪指标
        
        参数:
            data: 原始数据DataFrame
            sentiment_data: 情绪数据DataFrame（可选）
            
        返回:
            添加情绪指标后的DataFrame
        """
        df = data.copy()
        
        if sentiment_data is not None:
            # 如果提供了情绪数据，合并到主数据中
            df = pd.merge(df, sentiment_data, 
                          left_index=True, right_index=True, 
                          how='left')
        else:
            # 如果没有情绪数据，简单地添加一个基于技术指标的情绪代理
            if 'rsi_14' in df.columns:
                # RSI的简单情绪指标: RSI高表示乐观，RSI低表示悲观
                df['sentiment'] = (df['rsi_14'] - 50) / 50  # 归一化到[-1, 1]
            elif 'macd' in df.columns and 'macd_signal' in df.columns:
                # 基于MACD信号的情绪: MACD > 信号线表示乐观
                df['sentiment'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            else:
                # 如果没有其他指标，使用收益率的动量作为情绪代理
                price_col = 'close' if 'close' in df.columns else 'Close'
                returns = df[price_col].pct_change()
                df['sentiment'] = returns.rolling(window=5).mean() / returns.rolling(window=5).std()
        
        return df
    
    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值
        
        参数:
            data: 包含缺失值的DataFrame
            
        返回:
            填充缺失值后的DataFrame
        """
        df = data.copy()
        
        # 前向填充（填充交易日中间缺失的数据）
        df = df.fillna(method='ffill')
        
        # 后向填充（处理最早的数据）
        df = df.fillna(method='bfill')
        
        # 对于仍然缺失的值，用0填充
        df = df.fillna(0)
        
        return df 