"""
Tests for sentiment analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from ml.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        
        # Sample data
        self.sample_texts = [
            "I love this course! It's amazing and very helpful.",
            "This is terrible. I hate everything about it.",
            "It's okay, nothing special but not bad either.",
            "",
            None,
            "The content is good but could be improved with more examples."
        ]
        
        self.sample_df = pd.DataFrame({
            'feedback_positive': [
                "Great course, learned a lot!",
                "Excellent content and presentation",
                "Very helpful and informative"
            ],
            'feedback_negative': [
                "Could be better with more examples",
                "Too fast paced for beginners",
                "Needs more interactive elements"
            ],
            'rating': [4.5, 5.0, 4.0]
        })
    
    def test_analyze_text_positive(self):
        """Test sentiment analysis for positive text."""
        result = self.analyzer.analyze_text("I love this course! It's amazing!")
        
        assert isinstance(result, dict)
        assert 'vader_compound' in result
        assert 'sentiment_label' in result
        assert result['vader_compound'] > 0
        assert result['sentiment_label'] == 'positive'
    
    def test_analyze_text_negative(self):
        """Test sentiment analysis for negative text."""
        result = self.analyzer.analyze_text("This is terrible and awful!")
        
        assert isinstance(result, dict)
        assert result['vader_compound'] < 0
        assert result['sentiment_label'] == 'negative'
    
    def test_analyze_text_neutral(self):
        """Test sentiment analysis for neutral text."""
        result = self.analyzer.analyze_text("This is a course about programming.")
        
        assert isinstance(result, dict)
        assert abs(result['vader_compound']) <= 0.05
        assert result['sentiment_label'] == 'neutral'
    
    def test_analyze_text_empty(self):
        """Test sentiment analysis for empty text."""
        result = self.analyzer.analyze_text("")
        
        assert isinstance(result, dict)
        assert pd.isna(result['vader_compound'])
        assert result['sentiment_label'] == 'neutral'
    
    def test_analyze_text_none(self):
        """Test sentiment analysis for None input."""
        result = self.analyzer.analyze_text(None)
        
        assert isinstance(result, dict)
        assert pd.isna(result['vader_compound'])
        assert result['sentiment_label'] == 'neutral'
    
    def test_analyze_dataframe(self):
        """Test sentiment analysis on DataFrame."""
        result_df = self.analyzer.analyze_dataframe(
            self.sample_df, 
            ['feedback_positive', 'feedback_negative']
        )
        
        # Check that new columns are added
        expected_columns = [
            'feedback_positive_vader_compound',
            'feedback_positive_sentiment_label',
            'feedback_negative_vader_compound',
            'feedback_negative_sentiment_label'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns
        
        # Check that original columns are preserved
        assert 'rating' in result_df.columns
        assert len(result_df) == len(self.sample_df)
    
    def test_get_sentiment_summary(self):
        """Test sentiment summary statistics."""
        df_with_sentiment = self.analyzer.analyze_dataframe(
            self.sample_df, 
            ['feedback_positive']
        )
        
        summary = self.analyzer.get_sentiment_summary(
            df_with_sentiment, 
            'feedback_positive_vader_compound'
        )
        
        assert isinstance(summary, dict)
        assert 'count' in summary
        assert 'mean' in summary
        assert 'positive_count' in summary
        assert 'negative_count' in summary
        assert 'neutral_count' in summary
        assert summary['count'] == 3
    
    def test_compare_sentiments(self):
        """Test sentiment comparison between columns."""
        df_with_sentiment = self.analyzer.analyze_dataframe(
            self.sample_df, 
            ['feedback_positive', 'feedback_negative']
        )
        
        comparison = self.analyzer.compare_sentiments(
            df_with_sentiment,
            'feedback_positive_vader_compound',
            'feedback_negative_vader_compound'
        )
        
        assert isinstance(comparison, dict)
        assert 'correlation' in comparison
        assert 'mean_difference' in comparison
        assert 'pairs_count' in comparison
        assert comparison['pairs_count'] == 3
    
    def test_get_most_positive_negative(self):
        """Test getting most positive and negative examples."""
        df_with_sentiment = self.analyzer.analyze_dataframe(
            self.sample_df, 
            ['feedback_positive']
        )
        
        most_positive, most_negative = self.analyzer.get_most_positive_negative(
            df_with_sentiment,
            'feedback_positive',
            'feedback_positive_vader_compound',
            n=2
        )
        
        assert isinstance(most_positive, pd.DataFrame)
        assert isinstance(most_negative, pd.DataFrame)
        assert len(most_positive) <= 2
        assert len(most_negative) <= 2
    
    def test_invalid_column(self):
        """Test handling of invalid column names."""
        with pytest.raises(ValueError):
            self.analyzer.get_sentiment_summary(self.sample_df, 'nonexistent_column')
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result_df = self.analyzer.analyze_dataframe(empty_df, [])
        
        assert len(result_df) == 0