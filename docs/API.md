# API Documentation

## Machine Learning Modules

### SentimentAnalyzer

The `SentimentAnalyzer` class provides comprehensive sentiment analysis capabilities for feedback text.

#### Methods

##### `__init__()`
Initialize the sentiment analyzer with VADER and TextBlob.

##### `analyze_text(text: str) -> Dict[str, float]`
Analyze sentiment of a single text string.

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
- Dictionary containing:
  - `vader_compound`: Overall sentiment score (-1 to 1)
  - `vader_positive`: Positive sentiment component
  - `vader_negative`: Negative sentiment component
  - `vader_neutral`: Neutral sentiment component
  - `textblob_polarity`: TextBlob polarity score
  - `textblob_subjectivity`: TextBlob subjectivity score
  - `sentiment_label`: 'positive', 'negative', or 'neutral'

**Example:**
```python
from ml.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_text("I love this course!")
print(result['sentiment_label'])  # 'positive'
```

##### `analyze_dataframe(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame`
Analyze sentiment for multiple text columns in a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing text columns
- `text_columns` (List[str]): List of column names to analyze

**Returns:**
- DataFrame with added sentiment columns for each text column

##### `get_sentiment_summary(df: pd.DataFrame, sentiment_column: str) -> Dict[str, any]`
Get summary statistics for sentiment analysis.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with sentiment scores
- `sentiment_column` (str): Name of the sentiment score column

**Returns:**
- Dictionary with statistics including count, mean, median, positive/negative/neutral counts and percentages

### TopicModeler

The `TopicModeler` class provides topic modeling using LDA and NMF algorithms.

#### Methods

##### `__init__(random_state: int = 42)`
Initialize the topic modeler.

##### `fit_lda(documents: List[str], n_topics: int = 5, max_features: int = 1000, min_df: int = 2, max_df: float = 0.95) -> Dict[str, Any]`
Fit LDA topic model on documents.

**Parameters:**
- `documents` (List[str]): List of documents to analyze
- `n_topics` (int): Number of topics to extract
- `max_features` (int): Maximum number of features
- `min_df` (int): Minimum document frequency
- `max_df` (float): Maximum document frequency

**Returns:**
- Dictionary containing topics, document-topic matrix, and model information

**Example:**
```python
from ml.topic_modeler import TopicModeler

modeler = TopicModeler()
documents = ["Great course content", "Needs more examples", "Excellent presentation"]
results = modeler.fit_lda(documents, n_topics=2)

if "error" not in results:
    for i, topic in enumerate(results["topics"]):
        print(f"Topic {i}: {', '.join(topic['words'][:5])}")
```

##### `fit_nmf(documents: List[str], n_topics: int = 5, ...) -> Dict[str, Any]`
Fit NMF topic model on documents. Similar parameters to `fit_lda`.

##### `get_topic_evolution(df: pd.DataFrame, text_column: str, date_column: str, n_topics: int = 5, freq: str = 'M') -> pd.DataFrame`
Analyze how topics change over time.

### UserSegmentation

The `UserSegmentation` class provides user clustering and segmentation capabilities.

#### Methods

##### `__init__(random_state: int = 42)`
Initialize the user segmentation class.

##### `prepare_features(df: pd.DataFrame, feature_columns: List[str], handle_missing: str = 'mean') -> pd.DataFrame`
Prepare features for clustering.

##### `find_optimal_clusters(features: np.ndarray, max_clusters: int = 10, method: str = 'elbow') -> Dict[str, Any]`
Find optimal number of clusters using elbow method or silhouette analysis.

##### `fit_kmeans(features: np.ndarray, n_clusters: int) -> Dict[str, Any]`
Fit KMeans clustering model.

**Example:**
```python
from ml.user_segmentation import UserSegmentation
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'rating': [4.5, 3.8, 5.0, 4.2],
    'hours_spent': [2.0, 3.5, 1.5, 2.8],
    'sentiment_score': [0.8, -0.2, 0.9, 0.3]
})

segmenter = UserSegmentation()
features_df = segmenter.prepare_features(df, ['rating', 'hours_spent', 'sentiment_score'])
scaled_features = segmenter.scale_features(features_df)

# Find optimal clusters
optimal_results = segmenter.find_optimal_clusters(scaled_features)
print(f"Optimal clusters: {optimal_results.get('optimal_silhouette', 3)}")

# Fit clustering
results = segmenter.fit_kmeans(scaled_features, n_clusters=3)
cluster_labels = results['cluster_labels']
```

### PredictiveModels

The `PredictiveModels` class provides predictive modeling capabilities.

#### Methods

##### `train_rating_predictor(df: pd.DataFrame, feature_columns: List[str], target_column: str = 'rating', model_type: str = 'random_forest', test_size: float = 0.2) -> Dict[str, Any]`
Train a model to predict ratings.

**Parameters:**
- `df` (pd.DataFrame): Training data
- `feature_columns` (List[str]): Feature columns to use
- `target_column` (str): Target column name
- `model_type` (str): Type of model ('random_forest', 'linear_regression', 'gradient_boosting', 'svr')
- `test_size` (float): Proportion of data for testing

**Returns:**
- Dictionary with training results, metrics, and feature importance

##### `train_satisfaction_classifier(df: pd.DataFrame, feature_columns: List[str], target_column: str = 'got_what_you_needed', model_type: str = 'random_forest', test_size: float = 0.2) -> Dict[str, Any]`
Train a model to classify user satisfaction.

**Example:**
```python
from ml.predictive_models import PredictiveModels
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'course_duration': [90, 120, 60, 90],
    'sentiment_score': [0.8, -0.2, 0.9, 0.3],
    'rating': [4.5, 3.8, 5.0, 4.2]
})

predictor = PredictiveModels()
results = predictor.train_rating_predictor(
    df, 
    feature_columns=['course_duration', 'sentiment_score'],
    target_column='rating',
    model_type='random_forest'
)

print(f"R² Score: {results['metrics']['r2_score']:.3f}")
print(f"RMSE: {results['metrics']['rmse']:.3f}")
```

## Main Application Functions

### Data Processing

##### `process_data(df: pd.DataFrame) -> pd.DataFrame`
Process and clean the uploaded dataframe.

**Features:**
- Removes duplicates
- Handles missing values
- Converts data types
- Adds sentiment analysis columns
- Creates datetime features

### Analytics Functions

##### `calculate_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
Calculate correlation matrix and p-values for numeric columns.

##### `perform_topic_modeling(df: pd.DataFrame, column: str, n_topics: int = 5) -> Tuple`
Perform topic modeling on text column using LDA.

##### `segment_users(df: pd.DataFrame, n_clusters: int = 4) -> Tuple`
Perform user segmentation using K-means clustering.

## Database Functions

##### `get_db_config() -> Dict[str, str]`
Get database configuration from environment variables or config file.

##### `create_db_connection() -> sqlalchemy.Engine`
Create SQLAlchemy database engine with error handling.

##### `load_data_from_database(dashboard_type: str, filters: Dict = None) -> pd.DataFrame`
Load data from database with optional filters.

## Error Handling

All functions include comprehensive error handling and return appropriate error messages or empty results when operations cannot be completed.

## Performance Considerations

- Functions use caching where appropriate (`@st.cache_data`)
- Database connections include connection pooling
- Large datasets are processed in chunks when possible
- Memory usage is optimized for typical feedback dataset sizes