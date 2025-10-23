"""
User Segmentation Module

This module provides user segmentation capabilities using various clustering algorithms
for analyzing user behavior patterns in feedback data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserSegmentation:
    """
    A comprehensive user segmentation class using various clustering algorithms.
    
    Provides functionality for preprocessing features, applying clustering algorithms,
    and analyzing user segments.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the user segmentation class.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.cluster_centers = None
        self.n_clusters = None
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_columns: List[str],
                        handle_missing: str = 'mean') -> pd.DataFrame:
        """
        Prepare features for clustering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (List[str]): List of feature column names
            handle_missing (str): How to handle missing values ('mean', 'median', 'drop')
            
        Returns:
            pd.DataFrame: Prepared features dataframe
        """
        # Select feature columns
        features_df = df[feature_columns].copy()
        
        # Handle missing values
        if handle_missing == 'mean':
            features_df = features_df.fillna(features_df.mean())
        elif handle_missing == 'median':
            features_df = features_df.fillna(features_df.median())
        elif handle_missing == 'drop':
            features_df = features_df.dropna()
        
        # Store feature names
        self.feature_names = feature_columns
        
        logger.info(f"Prepared {len(features_df)} samples with {len(feature_columns)} features")
        return features_df
    
    def scale_features(self, features_df: pd.DataFrame, 
                      method: str = 'standard') -> np.ndarray:
        """
        Scale features for clustering.
        
        Args:
            features_df (pd.DataFrame): Features dataframe
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            np.ndarray: Scaled features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        scaled_features = self.scaler.fit_transform(features_df)
        logger.info(f"Scaled features using {method} scaling")
        
        return scaled_features
    
    def find_optimal_clusters(self, features: np.ndarray, 
                            max_clusters: int = 10,
                            method: str = 'elbow') -> Dict[str, Any]:
        """
        Find optimal number of clusters using various methods.
        
        Args:
            features (np.ndarray): Scaled features
            max_clusters (int): Maximum number of clusters to test
            method (str): Method to use ('elbow', 'silhouette', 'both')
            
        Returns:
            Dict[str, Any]: Results with optimal cluster numbers and scores
        """
        if len(features) < max_clusters:
            max_clusters = len(features) - 1
        
        cluster_range = range(2, max_clusters + 1)
        results = {
            'cluster_range': list(cluster_range),
            'inertias': [],
            'silhouette_scores': [],
            'calinski_harabasz_scores': []
        }
        
        for n_clusters in cluster_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            results['inertias'].append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                sil_score = silhouette_score(features, cluster_labels)
                ch_score = calinski_harabasz_score(features, cluster_labels)
                results['silhouette_scores'].append(sil_score)
                results['calinski_harabasz_scores'].append(ch_score)
            else:
                results['silhouette_scores'].append(0)
                results['calinski_harabasz_scores'].append(0)
        
        # Find optimal clusters
        if method in ['elbow', 'both']:
            # Simple elbow method - find the point with maximum second derivative
            inertias = results['inertias']
            if len(inertias) >= 3:
                second_derivatives = []
                for i in range(1, len(inertias) - 1):
                    second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_derivatives.append(second_deriv)
                
                if second_derivatives:
                    optimal_elbow = cluster_range[np.argmax(second_derivatives) + 1]
                    results['optimal_elbow'] = optimal_elbow
        
        if method in ['silhouette', 'both']:
            # Best silhouette score
            if results['silhouette_scores']:
                optimal_silhouette = cluster_range[np.argmax(results['silhouette_scores'])]
                results['optimal_silhouette'] = optimal_silhouette
        
        return results
    
    def fit_kmeans(self, features: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """
        Fit KMeans clustering model.
        
        Args:
            features (np.ndarray): Scaled features
            n_clusters (int): Number of clusters
            
        Returns:
            Dict[str, Any]: Clustering results
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.model.fit_predict(features)
        
        self.n_clusters = n_clusters
        self.cluster_centers = self.model.cluster_centers_
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(features, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': self.cluster_centers,
            'n_clusters': n_clusters,
            'metrics': metrics,
            'model_type': 'KMeans'
        }
    
    def fit_dbscan(self, features: np.ndarray, eps: float = 0.5, 
                   min_samples: int = 5) -> Dict[str, Any]:
        """
        Fit DBSCAN clustering model.
        
        Args:
            features (np.ndarray): Scaled features
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
            
        Returns:
            Dict[str, Any]: Clustering results
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.model.fit_predict(features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.n_clusters = n_clusters
        
        # Calculate metrics (if we have valid clusters)
        metrics = {}
        if n_clusters > 1:
            metrics = self._calculate_clustering_metrics(features, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'metrics': metrics,
            'model_type': 'DBSCAN'
        }
    
    def fit_hierarchical(self, features: np.ndarray, n_clusters: int,
                        linkage: str = 'ward') -> Dict[str, Any]:
        """
        Fit Agglomerative (Hierarchical) clustering model.
        
        Args:
            features (np.ndarray): Scaled features
            n_clusters (int): Number of clusters
            linkage (str): Linkage criterion
            
        Returns:
            Dict[str, Any]: Clustering results
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = self.model.fit_predict(features)
        
        self.n_clusters = n_clusters
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(features, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'metrics': metrics,
            'model_type': 'Hierarchical'
        }
    
    def _calculate_clustering_metrics(self, features: np.ndarray, 
                                    cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering evaluation metrics.
        
        Args:
            features (np.ndarray): Features used for clustering
            cluster_labels (np.ndarray): Cluster assignments
            
        Returns:
            Dict[str, float]: Clustering metrics
        """
        metrics = {}
        
        # Only calculate if we have valid clusters
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            try:
                metrics['silhouette_score'] = silhouette_score(features, cluster_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, cluster_labels)
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
        
        # Inertia for KMeans
        if hasattr(self.model, 'inertia_'):
            metrics['inertia'] = self.model.inertia_
        
        return metrics
    
    def analyze_segments(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                        feature_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze characteristics of each segment.
        
        Args:
            df (pd.DataFrame): Original dataframe
            cluster_labels (np.ndarray): Cluster assignments
            feature_columns (List[str]): Feature columns used for clustering
            
        Returns:
            Dict[str, Any]: Segment analysis results
        """
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Analyze each cluster
        segment_profiles = {}
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_with_clusters)) * 100,
                'feature_means': {},
                'feature_medians': {},
                'feature_stds': {}
            }
            
            # Calculate statistics for each feature
            for feature in feature_columns:
                if feature in cluster_data.columns:
                    feature_data = cluster_data[feature].dropna()
                    if len(feature_data) > 0:
                        profile['feature_means'][feature] = feature_data.mean()
                        profile['feature_medians'][feature] = feature_data.median()
                        profile['feature_stds'][feature] = feature_data.std()
            
            segment_profiles[f'segment_{cluster_id}'] = profile
        
        return segment_profiles
    
    def get_segment_comparison(self, segment_profiles: Dict[str, Any],
                             feature_columns: List[str]) -> pd.DataFrame:
        """
        Create a comparison table of segment characteristics.
        
        Args:
            segment_profiles (Dict[str, Any]): Segment profiles from analyze_segments
            feature_columns (List[str]): Feature columns to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for segment_name, profile in segment_profiles.items():
            row = {
                'segment': segment_name,
                'size': profile['size'],
                'percentage': profile['percentage']
            }
            
            # Add feature means
            for feature in feature_columns:
                if feature in profile['feature_means']:
                    row[f'{feature}_mean'] = profile['feature_means'][feature]
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def reduce_dimensions_for_visualization(self, features: np.ndarray, 
                                          n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions for visualization using PCA.
        
        Args:
            features (np.ndarray): High-dimensional features
            n_components (int): Number of components to keep
            
        Returns:
            np.ndarray: Reduced features
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced_features = pca.fit_transform(features)
        
        logger.info(f"Reduced features to {n_components} dimensions")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return reduced_features
    
    def predict_segment(self, new_data: np.ndarray) -> np.ndarray:
        """
        Predict segment for new data points.
        
        Args:
            new_data (np.ndarray): New data points to classify
            
        Returns:
            np.ndarray: Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_* method first.")
        
        if self.scaler is not None:
            new_data_scaled = self.scaler.transform(new_data)
        else:
            new_data_scaled = new_data
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(new_data_scaled)
        else:
            raise ValueError("Model does not support prediction")
    
    def get_feature_importance(self, features: np.ndarray, 
                             cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance for clustering (simplified approach).
        
        Args:
            features (np.ndarray): Features used for clustering
            cluster_labels (np.ndarray): Cluster assignments
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_names is None:
            return {}
        
        importance_scores = {}
        
        # Calculate variance between clusters for each feature
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features[:, i]
            
            # Calculate between-cluster variance
            cluster_means = []
            for cluster_id in np.unique(cluster_labels):
                if cluster_id != -1:  # Skip noise points
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_mean = np.mean(feature_values[cluster_mask])
                        cluster_means.append(cluster_mean)
            
            if len(cluster_means) > 1:
                between_cluster_variance = np.var(cluster_means)
                importance_scores[feature_name] = between_cluster_variance
            else:
                importance_scores[feature_name] = 0.0
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        if max_score > 0:
            importance_scores = {k: v/max_score for k, v in importance_scores.items()}
        
        return importance_scores