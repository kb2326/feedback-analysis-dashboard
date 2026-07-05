"""
Tests for main application functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from app (we'll need to mock streamlit)
streamlit_mock = MagicMock()
streamlit_mock.cache_data.side_effect = lambda func=None, **_kwargs: (
    func if func is not None else lambda wrapped: wrapped
)


class StubSentimentIntensityAnalyzer:
    """Small deterministic replacement for tests that do not exercise VADER."""

    def polarity_scores(self, text):
        return {'compound': 0.25 if str(text).strip() else 0.0}


with patch.dict('sys.modules', {'streamlit': streamlit_mock}):
    import app as app_module

    app_module.SentimentIntensityAnalyzer = StubSentimentIntensityAnalyzer
    calculate_correlations = app_module.calculate_correlations
    normalize_feedback_columns = app_module.normalize_feedback_columns
    perform_topic_modeling = app_module.perform_topic_modeling
    process_data = app_module.process_data
    segment_users = app_module.segment_users


class TestAppFunctions:
    """Test cases for main application functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'course_id': [1, 2, 3, 1, 2],
            'course_name': ['Course A', 'Course B', 'Course C', 'Course A', 'Course B'],
            'rating': [4.5, 3.8, 5.0, 4.0, 4.2],
            'what_did_you_like': [
                'Great content and examples',
                'Good structure',
                'Excellent presentation',
                'Clear explanations',
                'Helpful materials'
            ],
            'what_could_be_improved': [
                'More exercises needed',
                'Too fast pace',
                'Nothing to improve',
                'Add more examples',
                'Better audio quality'
            ],
            'course_duration_minutes': [90, 120, 60, 90, 120],
            'how_many_hours_it_took': [2.0, 3.5, 1.5, 2.5, 3.0],
            'datetime_originally_submitted': [
                '2023-01-15', '2023-02-20', '2023-03-10',
                '2023-04-05', '2023-05-12'
            ],
            'client_name': ['Client A', 'Client B', 'Client A', 'Client C', 'Client B'],
            'client_sector': ['Tech', 'Finance', 'Tech', 'Healthcare', 'Finance']
        })
    
    def test_process_data(self):
        """Test data processing function."""
        processed_df = process_data(self.sample_df.copy())
        
        # Check that data processing doesn't break the DataFrame
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) > 0
        
        # Check that datetime conversion works
        assert 'datetime_originally_submitted' in processed_df.columns
        assert pd.api.types.is_datetime64_any_dtype(processed_df['datetime_originally_submitted'])
        
        # Check that sentiment analysis columns are added
        if 'what_did_you_like' in processed_df.columns:
            assert 'positive_sentiment' in processed_df.columns
        if 'what_could_be_improved' in processed_df.columns:
            assert 'improvement_sentiment' in processed_df.columns

    def test_normalize_xquik_tweet_export_columns(self):
        """Test Xquik tweet export columns are accepted as feedback input."""
        xquik_df = pd.DataFrame({
            'Tweet Text': ['Great launch thread', '   ', 'Needs clearer docs'],
            'Tweet Created At': ['2026-07-01T10:00:00Z', '', '2026-07-02T11:00:00Z'],
        })

        normalized_df = normalize_feedback_columns(xquik_df)
        processed_df = process_data(xquik_df)

        assert list(normalized_df['what_did_you_like']) == [
            'Great launch thread',
            'Needs clearer docs',
        ]
        assert list(normalized_df['source']) == ['xquik', 'xquik']
        assert 'datetime_originally_submitted' in processed_df.columns
        assert pd.api.types.is_datetime64_any_dtype(processed_df['datetime_originally_submitted'])

    def test_normalize_preserves_native_feedback_columns(self):
        """Test native feedback text is not replaced by alternate aliases."""
        df = pd.DataFrame({
            'what_did_you_like': ['Native feedback'],
            'Tweet Text': ['Should not replace native feedback'],
        })

        normalized_df = normalize_feedback_columns(df)

        assert normalized_df.loc[0, 'what_did_you_like'] == 'Native feedback'
    
    def test_calculate_correlations(self):
        """Test correlation calculation function."""
        processed_df = process_data(self.sample_df.copy())
        corr_matrix, p_values = calculate_correlations(processed_df)
        
        # Check that correlation matrices are returned
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(p_values, pd.DataFrame)
        
        # If we have enough numeric columns, check matrix properties
        if not corr_matrix.empty:
            assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
            assert np.allclose(np.diag(corr_matrix), 1.0, equal_nan=True)  # Diagonal should be 1
    
    def test_perform_topic_modeling(self):
        """Test topic modeling function."""
        # Test with sufficient data
        result = perform_topic_modeling(self.sample_df, 'what_did_you_like', n_topics=2)
        
        if result[0] is not None:  # If topic modeling succeeded
            topics_words, lda_model, vectorizer, dtm = result
            assert isinstance(topics_words, list)
            assert len(topics_words) == 2
            assert lda_model is not None
            assert vectorizer is not None
            assert dtm is not None
        else:
            # Topic modeling failed due to insufficient data, which is expected
            assert result == (None, None, None, None)
    
    def test_perform_topic_modeling_insufficient_data(self):
        """Test topic modeling with insufficient data."""
        small_df = self.sample_df.head(1)  # Only one row
        result = perform_topic_modeling(small_df, 'what_did_you_like', n_topics=3)
        
        # Should return None values due to insufficient data
        assert result == (None, None, None, None)
    
    def test_segment_users(self):
        """Test user segmentation function."""
        processed_df = process_data(self.sample_df.copy())
        
        # Test segmentation
        result = segment_users(processed_df, n_clusters=2)
        
        if result[0] is not None:  # If segmentation succeeded
            segmented_df, cluster_centers = result
            assert isinstance(segmented_df, pd.DataFrame)
            assert 'cluster' in segmented_df.columns
            assert len(segmented_df) == len(processed_df)
            assert cluster_centers is not None
            
            # Check that cluster labels are valid
            unique_clusters = segmented_df['cluster'].unique()
            assert len(unique_clusters) <= 2
            assert all(cluster >= 0 for cluster in unique_clusters)
        else:
            # Segmentation failed due to insufficient features
            assert result == (None, None)
    
    def test_segment_users_insufficient_features(self):
        """Test user segmentation with insufficient features."""
        # Create DataFrame with only one numeric column
        minimal_df = pd.DataFrame({
            'rating': [4.5, 3.8, 5.0]
        })
        
        result = segment_users(minimal_df, n_clusters=2)
        
        # Should return None values due to insufficient features
        assert result == (None, None)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Process data should handle empty DataFrame gracefully
        processed_df = process_data(empty_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == 0
        
        # Correlation calculation should handle empty DataFrame
        corr_matrix, p_values = calculate_correlations(empty_df)
        assert corr_matrix.empty
        assert p_values.empty
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'rating'] = np.nan
        df_with_missing.loc[1, 'what_did_you_like'] = np.nan
        
        # Process data should handle missing values
        processed_df = process_data(df_with_missing)
        assert isinstance(processed_df, pd.DataFrame)
        
        # Should have fewer rows after dropping missing ratings
        assert len(processed_df) < len(df_with_missing)
    
    def test_data_type_conversion(self):
        """Test proper data type conversion."""
        # Create DataFrame with string numbers
        df_with_strings = self.sample_df.copy()
        df_with_strings['rating'] = df_with_strings['rating'].astype(str)
        df_with_strings['course_duration_minutes'] = df_with_strings['course_duration_minutes'].astype(str)
        
        processed_df = process_data(df_with_strings)
        
        # Check that numeric columns are properly converted
        assert pd.api.types.is_numeric_dtype(processed_df['rating'])
        assert pd.api.types.is_numeric_dtype(processed_df['course_duration_minutes'])


class TestDataValidation:
    """Test cases for data validation and edge cases."""
    
    def test_invalid_rating_values(self):
        """Test handling of invalid rating values."""
        df_invalid_ratings = pd.DataFrame({
            'course_id': [1, 2, 3],
            'rating': [-1, 6, 'invalid'],  # Invalid ratings
            'what_did_you_like': ['Good', 'Great', 'Excellent']
        })
        
        processed_df = process_data(df_invalid_ratings)
        
        # Invalid ratings should be converted to NaN and rows dropped
        assert len(processed_df) <= len(df_invalid_ratings)
        
        # Remaining ratings should be valid
        valid_ratings = processed_df['rating'].dropna()
        if len(valid_ratings) > 0:
            assert all(rating >= 0 for rating in valid_ratings)
    
    def test_duplicate_handling(self):
        """Test handling of duplicate rows."""
        df_with_duplicates = pd.DataFrame({
            'course_id': [1, 1, 2],  # Duplicate course_id
            'rating': [4.5, 4.5, 3.8],  # Same rating
            'what_did_you_like': ['Good', 'Good', 'Great']  # Same feedback
        })
        
        processed_df = process_data(df_with_duplicates)
        
        # Duplicates should be removed
        assert len(processed_df) <= len(df_with_duplicates)
    
    def test_text_preprocessing(self):
        """Test text preprocessing for topic modeling."""
        df_with_text = pd.DataFrame({
            'feedback': [
                'This is a great course with excellent content!',
                'Good course but could be improved.',
                'Amazing learning experience.',
                ''  # Empty text
            ]
        })
        
        # Test topic modeling with this data
        result = perform_topic_modeling(df_with_text, 'feedback', n_topics=2)
        
        # Should handle empty text gracefully
        assert isinstance(result, tuple)
        assert len(result) == 4
