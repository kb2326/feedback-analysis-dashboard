"""
Machine Learning modules for feedback analysis.

This package contains modules for:
- Sentiment analysis
- Topic modeling
- Predictive modeling
- User segmentation
"""

__version__ = "1.0.0"
__author__ = "Feedback Analysis Team"

from .sentiment_analyzer import SentimentAnalyzer
from .topic_modeler import TopicModeler
from .user_segmentation import UserSegmentation

__all__ = [
    "SentimentAnalyzer",
    "TopicModeler", 
    "UserSegmentation"
]