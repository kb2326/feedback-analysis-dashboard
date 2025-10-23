"""
Sentiment Analysis Module

This module provides sentiment analysis capabilities using VADER sentiment analyzer
and TextBlob for feedback text analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A comprehensive sentiment analyzer for feedback text.
    
    Uses VADER sentiment analyzer for social media text and TextBlob
    for additional sentiment metrics.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.vader_analyzer = None
        self._initialize_nltk_data()
    
    def _initialize_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('vader_lexicon')
            nltk.data.find('punkt')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores
        """
        if not text or pd.isna(text):
            return {
                'vader_compound': np.nan,
                'vader_positive': np.nan,
                'vader_negative': np.nan,
                'vader_neutral': np.nan,
                'textblob_polarity': np.nan,
                'textblob_subjectivity': np.nan,
                'sentiment_label': 'neutral'
            }
        
        text = str(text).strip()
        if not text:
            return self._empty_sentiment()
        
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        compound_score = vader_scores['compound']
        if compound_score >= 0.05:
            sentiment_label = 'positive'
        elif compound_score <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'vader_compound': compound_score,
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'sentiment_label': sentiment_label
        }
    
    def _empty_sentiment(self) -> Dict[str, float]:
        """Return empty sentiment scores."""
        return {
            'vader_compound': np.nan,
            'vader_positive': np.nan,
            'vader_negative': np.nan,
            'vader_neutral': np.nan,
            'textblob_polarity': np.nan,
            'textblob_subjectivity': np.nan,
            'sentiment_label': 'neutral'
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple text columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text columns
            text_columns (List[str]): List of column names to analyze
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment columns
        """
        df_copy = df.copy()
        
        for col in text_columns:
            if col not in df_copy.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            logger.info(f"Analyzing sentiment for column: {col}")
            
            # Analyze each text
            sentiment_results = df_copy[col].apply(self.analyze_text)
            
            # Extract sentiment metrics into separate columns
            for metric in ['vader_compound', 'vader_positive', 'vader_negative', 
                          'vader_neutral', 'textblob_polarity', 'textblob_subjectivity', 
                          'sentiment_label']:
                df_copy[f"{col}_{metric}"] = sentiment_results.apply(lambda x: x[metric])
        
        return df_copy
    
    def get_sentiment_summary(self, df: pd.DataFrame, sentiment_column: str) -> Dict[str, any]:
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment scores
            sentiment_column (str): Name of the sentiment score column
            
        Returns:
            Dict[str, any]: Summary statistics
        """
        if sentiment_column not in df.columns:
            raise ValueError(f"Column '{sentiment_column}' not found in DataFrame")
        
        sentiment_scores = df[sentiment_column].dropna()
        
        if len(sentiment_scores) == 0:
            return {"error": "No valid sentiment scores found"}
        
        # Calculate statistics
        stats = {
            'count': len(sentiment_scores),
            'mean': sentiment_scores.mean(),
            'median': sentiment_scores.median(),
            'std': sentiment_scores.std(),
            'min': sentiment_scores.min(),
            'max': sentiment_scores.max(),
            'positive_count': (sentiment_scores > 0.05).sum(),
            'negative_count': (sentiment_scores < -0.05).sum(),
            'neutral_count': ((sentiment_scores >= -0.05) & (sentiment_scores <= 0.05)).sum()
        }
        
        # Calculate percentages
        total = stats['count']
        stats['positive_percentage'] = (stats['positive_count'] / total) * 100
        stats['negative_percentage'] = (stats['negative_count'] / total) * 100
        stats['neutral_percentage'] = (stats['neutral_count'] / total) * 100
        
        return stats
    
    def compare_sentiments(self, df: pd.DataFrame, 
                          sentiment_col1: str, sentiment_col2: str) -> Dict[str, float]:
        """
        Compare sentiment between two text columns.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment columns
            sentiment_col1 (str): First sentiment column
            sentiment_col2 (str): Second sentiment column
            
        Returns:
            Dict[str, float]: Comparison metrics
        """
        if sentiment_col1 not in df.columns or sentiment_col2 not in df.columns:
            raise ValueError("One or both sentiment columns not found")
        
        # Get valid pairs
        valid_mask = df[sentiment_col1].notna() & df[sentiment_col2].notna()
        col1_scores = df.loc[valid_mask, sentiment_col1]
        col2_scores = df.loc[valid_mask, sentiment_col2]
        
        if len(col1_scores) == 0:
            return {"error": "No valid sentiment pairs found"}
        
        # Calculate comparison metrics
        correlation = col1_scores.corr(col2_scores)
        mean_diff = (col1_scores - col2_scores).mean()
        
        return {
            'correlation': correlation,
            'mean_difference': mean_diff,
            'col1_mean': col1_scores.mean(),
            'col2_mean': col2_scores.mean(),
            'pairs_count': len(col1_scores)
        }
    
    def get_most_positive_negative(self, df: pd.DataFrame, 
                                  text_column: str, 
                                  sentiment_column: str, 
                                  n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the most positive and negative feedback texts.
        
        Args:
            df (pd.DataFrame): DataFrame with text and sentiment
            text_column (str): Column containing text
            sentiment_column (str): Column containing sentiment scores
            n (int): Number of examples to return
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Most positive and negative examples
        """
        if text_column not in df.columns or sentiment_column not in df.columns:
            raise ValueError("Required columns not found in DataFrame")
        
        # Filter valid entries
        valid_df = df[[text_column, sentiment_column]].dropna()
        
        if len(valid_df) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get most positive and negative
        most_positive = valid_df.nlargest(n, sentiment_column)
        most_negative = valid_df.nsmallest(n, sentiment_column)
        
        return most_positive, most_negative