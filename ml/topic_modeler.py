"""
Topic Modeling Module

This module provides topic modeling capabilities using Latent Dirichlet Allocation (LDA)
and Non-negative Matrix Factorization (NMF) for analyzing feedback text.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """
    A comprehensive topic modeling class using LDA and NMF algorithms.
    
    Provides functionality for preprocessing text, extracting topics,
    and analyzing topic distributions.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the topic modeler.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.stop_words = None
        self.lemmatizer = None
        self._initialize_nltk_data()
    
    def _initialize_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('stopwords')
            nltk.data.find('punkt')
            nltk.data.find('wordnet')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stop words
        self.stop_words.update([
            'course', 'training', 'module', 'lesson', 'content',
            'good', 'great', 'nice', 'well', 'really', 'would',
            'could', 'should', 'like', 'think', 'know', 'get',
            'go', 'see', 'use', 'make', 'take', 'come', 'way'
        ])
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic modeling.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.
        
        Args:
            documents (List[str]): List of documents to preprocess
            
        Returns:
            List[str]: List of preprocessed documents
        """
        logger.info(f"Preprocessing {len(documents)} documents...")
        return [self.preprocess_text(doc) for doc in documents]
    
    def fit_lda(self, documents: List[str], n_topics: int = 5, 
                max_features: int = 1000, min_df: int = 2, max_df: float = 0.95) -> Dict[str, Any]:
        """
        Fit LDA topic model on documents.
        
        Args:
            documents (List[str]): List of documents
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            
        Returns:
            Dict[str, Any]: Model results including topics and document-topic matrix
        """
        # Preprocess documents
        processed_docs = self.preprocess_documents(documents)
        
        # Filter out empty documents
        processed_docs = [doc for doc in processed_docs if doc.strip()]
        
        if len(processed_docs) < n_topics:
            logger.warning(f"Not enough documents ({len(processed_docs)}) for {n_topics} topics")
            return {"error": "Insufficient documents for topic modeling"}
        
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        try:
            doc_term_matrix = self.vectorizer.fit_transform(processed_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            if doc_term_matrix.shape[1] < n_topics:
                logger.warning(f"Not enough features ({doc_term_matrix.shape[1]}) for {n_topics} topics")
                return {"error": "Insufficient features for topic modeling"}
            
            # Fit LDA model
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.random_state,
                max_iter=100,
                learning_method='batch'
            )
            
            doc_topic_matrix = self.model.fit_transform(doc_term_matrix)
            
            # Extract topics
            topics = self._extract_topics(n_words=10)
            
            return {
                "topics": topics,
                "doc_topic_matrix": doc_topic_matrix,
                "doc_term_matrix": doc_term_matrix,
                "n_topics": n_topics,
                "n_documents": len(processed_docs),
                "n_features": len(self.feature_names),
                "model_type": "LDA"
            }
            
        except Exception as e:
            logger.error(f"Error in LDA fitting: {e}")
            return {"error": str(e)}
    
    def fit_nmf(self, documents: List[str], n_topics: int = 5,
                max_features: int = 1000, min_df: int = 2, max_df: float = 0.95) -> Dict[str, Any]:
        """
        Fit NMF topic model on documents.
        
        Args:
            documents (List[str]): List of documents
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            
        Returns:
            Dict[str, Any]: Model results including topics and document-topic matrix
        """
        # Preprocess documents
        processed_docs = self.preprocess_documents(documents)
        
        # Filter out empty documents
        processed_docs = [doc for doc in processed_docs if doc.strip()]
        
        if len(processed_docs) < n_topics:
            logger.warning(f"Not enough documents ({len(processed_docs)}) for {n_topics} topics")
            return {"error": "Insufficient documents for topic modeling"}
        
        # Create TF-IDF matrix for NMF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            if tfidf_matrix.shape[1] < n_topics:
                logger.warning(f"Not enough features ({tfidf_matrix.shape[1]}) for {n_topics} topics")
                return {"error": "Insufficient features for topic modeling"}
            
            # Fit NMF model
            self.model = NMF(
                n_components=n_topics,
                random_state=self.random_state,
                max_iter=200
            )
            
            doc_topic_matrix = self.model.fit_transform(tfidf_matrix)
            
            # Extract topics
            topics = self._extract_topics(n_words=10)
            
            return {
                "topics": topics,
                "doc_topic_matrix": doc_topic_matrix,
                "tfidf_matrix": tfidf_matrix,
                "n_topics": n_topics,
                "n_documents": len(processed_docs),
                "n_features": len(self.feature_names),
                "model_type": "NMF"
            }
            
        except Exception as e:
            logger.error(f"Error in NMF fitting: {e}")
            return {"error": str(e)}
    
    def _extract_topics(self, n_words: int = 10) -> List[Dict[str, Any]]:
        """
        Extract topics from fitted model.
        
        Args:
            n_words (int): Number of top words per topic
            
        Returns:
            List[Dict[str, Any]]: List of topics with words and weights
        """
        if self.model is None or self.feature_names is None:
            return []
        
        topics = []
        
        for topic_idx, topic in enumerate(self.model.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                "topic_id": topic_idx,
                "words": top_words,
                "weights": top_weights,
                "word_weight_pairs": list(zip(top_words, top_weights))
            })
        
        return topics
    
    def get_document_topics(self, doc_topic_matrix: np.ndarray, 
                           threshold: float = 0.1) -> List[List[Tuple[int, float]]]:
        """
        Get topic assignments for documents.
        
        Args:
            doc_topic_matrix (np.ndarray): Document-topic probability matrix
            threshold (float): Minimum probability threshold
            
        Returns:
            List[List[Tuple[int, float]]]: Topic assignments per document
        """
        document_topics = []
        
        for doc_idx, doc_topics in enumerate(doc_topic_matrix):
            # Get topics above threshold
            topic_assignments = [
                (topic_idx, prob)
                for topic_idx, prob in enumerate(doc_topics)
                if prob > threshold
            ]
            
            # Sort by probability
            topic_assignments.sort(key=lambda x: x[1], reverse=True)
            document_topics.append(topic_assignments)
        
        return document_topics
    
    def get_topic_evolution(self, df: pd.DataFrame, text_column: str, 
                           date_column: str, n_topics: int = 5, 
                           freq: str = 'M') -> pd.DataFrame:
        """
        Analyze topic evolution over time.
        
        Args:
            df (pd.DataFrame): DataFrame with text and date columns
            text_column (str): Name of text column
            date_column (str): Name of date column
            n_topics (int): Number of topics
            freq (str): Frequency for grouping ('M' for monthly, 'Q' for quarterly)
            
        Returns:
            pd.DataFrame: Topic proportions over time
        """
        # Group by time period
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy['period'] = df_copy[date_column].dt.to_period(freq)
        
        evolution_data = []
        
        for period in df_copy['period'].unique():
            period_df = df_copy[df_copy['period'] == period]
            period_texts = period_df[text_column].dropna().tolist()
            
            if len(period_texts) < 5:  # Skip periods with too few documents
                continue
            
            # Fit topic model for this period
            results = self.fit_lda(period_texts, n_topics=n_topics)
            
            if "error" not in results:
                # Calculate topic proportions
                doc_topic_matrix = results["doc_topic_matrix"]
                topic_proportions = doc_topic_matrix.mean(axis=0)
                
                period_data = {"period": period.to_timestamp()}
                for i, prop in enumerate(topic_proportions):
                    period_data[f"topic_{i}"] = prop
                
                evolution_data.append(period_data)
        
        return pd.DataFrame(evolution_data)
    
    def get_topic_coherence(self, documents: List[str], topics: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate topic coherence scores (simplified version).
        
        Args:
            documents (List[str]): Original documents
            topics (List[Dict[str, Any]]): Extracted topics
            
        Returns:
            List[float]: Coherence scores for each topic
        """
        coherence_scores = []
        
        # Simple coherence based on word co-occurrence
        for topic in topics:
            words = topic["words"][:5]  # Top 5 words
            
            # Count co-occurrences
            cooccurrence_count = 0
            total_pairs = 0
            
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    total_pairs += 1
                    
                    # Count documents containing both words
                    cooccurrence = sum(
                        1 for doc in documents
                        if word1.lower() in doc.lower() and word2.lower() in doc.lower()
                    )
                    
                    if cooccurrence > 0:
                        cooccurrence_count += 1
            
            coherence = cooccurrence_count / total_pairs if total_pairs > 0 else 0
            coherence_scores.append(coherence)
        
        return coherence_scores