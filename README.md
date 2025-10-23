# 🚀 Advanced Feedback Analytics Platform

An enterprise-grade analytics platform leveraging machine learning and natural language processing to extract actionable insights from feedback data. Built with modern data science practices and scalable architecture.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Key Features & Technical Highlights

### 🤖 Advanced Machine Learning Pipeline
- **Sentiment Analysis**: VADER sentiment scoring with custom preprocessing
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) with dynamic topic optimization
- **User Segmentation**: K-Means clustering with feature engineering and PCA
- **Predictive Analytics**: Rating prediction models using ensemble methods
- **Anomaly Detection**: Statistical outlier detection for quality assurance

### 📊 Enterprise-Grade Analytics
- **Real-time Dashboards**: Interactive visualizations with Plotly and custom D3.js components
- **Statistical Analysis**: Correlation matrices, ANOVA testing, and significance analysis
- **Time Series Analysis**: Trend detection and seasonal decomposition
- **Cohort Analysis**: User behavior tracking over time
- **A/B Testing Framework**: Statistical testing for feature comparisons

### 🏗️ Scalable Architecture
- **Microservices Design**: Modular components with clear separation of concerns
- **Database Optimization**: Indexed queries and connection pooling
- **Caching Layer**: Redis integration for improved performance
- **API-First Approach**: RESTful endpoints for external integrations
- **Docker Containerization**: Production-ready deployment configuration

## 🏗️ Project Architecture

```
feedback-analytics-platform/
├── 📱 Frontend & UI
│   ├── app.py                      # Main Streamlit application
│   ├── components/                 # Custom UI components
│   └── pages/                      # Multi-page application structure
├── 🧠 Machine Learning Pipeline
│   ├── ml/
│   │   ├── sentiment_analyzer.py   # Advanced sentiment analysis
│   │   ├── topic_modeler.py        # LDA topic modeling
│   │   ├── clustering.py           # User segmentation algorithms
│   │   ├── predictive_models.py    # Rating prediction models
│   │   └── anomaly_detector.py     # Outlier detection
├── 📊 Analytics Engine
│   ├── analytics/
│   │   ├── statistical_tests.py    # ANOVA, t-tests, chi-square
│   │   ├── time_series.py          # Trend analysis
│   │   ├── cohort_analysis.py      # User behavior tracking
│   │   └── performance_metrics.py  # KPI calculations
├── 🗄️ Data Layer
│   ├── data/
│   │   ├── database.py             # Database connection & ORM
│   │   ├── preprocessing.py        # Data cleaning & feature engineering
│   │   ├── validators.py           # Data quality checks
│   │   └── exporters.py            # Report generation
├── 🔧 Infrastructure
│   ├── config/                     # Configuration management
│   ├── docker/                     # Container configurations
│   ├── sql/                        # Database schemas & migrations
│   └── tests/                      # Comprehensive test suite
└── 📚 Documentation
    ├── docs/                       # Technical documentation
    ├── examples/                   # Usage examples
    └── notebooks/                  # Jupyter analysis notebooks
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/feedback-analysis-dashboard.git
cd feedback-analysis-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your database configuration (optional):
```bash
cp config/database_config.py.template config/database_config.py
# Edit the configuration file with your database credentials
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

### Data Format

The application expects CSV files with the following columns (not all required):

**Course Feedback:**
- `course_id`: Unique identifier for the course
- `course_name`: Name of the course
- `course_category`: Category of the course
- `client_name`: Name of the client
- `client_sector`: Client industry sector
- `client_type`: Type of client (Corporate, Individual, Government)
- `course_status`: Status of the course (Active, Completed, Cancelled)
- `rating`: Numeric rating (1-5)
- `country`: Country of the participant
- `course_duration_minutes`: Course duration in minutes
- `how_many_hours_it_took`: Hours spent by participant
- `what_did_you_like`: Text feedback on positives
- `what_could_be_improved`: Text feedback on improvements
- `datetime_originally_submitted`: When feedback was submitted

**AXE Assistant Feedback:**
- `axe_id`: Unique identifier for the AXE assistant
- `axe_name`: Name of the AXE assistant
- `axe_category`: Category of the AXE assistant
- Similar client and feedback fields as course feedback

## Features Overview

### Dashboard Types
1. **Course Feedback Dashboard**: Analyze course delivery and content feedback
2. **AXE Assistant Dashboard**: Analyze AXE assistant interaction feedback

### Analytics Capabilities
- **Performance Analysis**: Top/bottom rated items, trends over time
- **Sentiment Analysis**: VADER sentiment scoring with visualizations
- **Topic Modeling**: LDA-based topic extraction from text feedback
- **Statistical Analysis**: Correlation analysis and significance testing
- **User Segmentation**: K-means clustering for user behavior patterns

### Visualizations
- Interactive charts with Plotly
- Word clouds for text analysis
- Correlation heatmaps
- Time series analysis
- Distribution plots
- Radar charts for segment profiling

## Database Support

The application supports MySQL databases with the following structure:

```sql
CREATE TABLE feedback (
    id INT PRIMARY KEY,
    user_id INT,
    course_id INT,
    client_id INT,
    client_name VARCHAR(255),
    client_type VARCHAR(100),
    client_sector VARCHAR(100),
    country VARCHAR(100),
    course_name VARCHAR(255),
    course_status VARCHAR(50),
    course_duration_minutes INT,
    course_category VARCHAR(100),
    rating DECIMAL(3,2),
    got_what_you_needed VARCHAR(10),
    what_did_you_like TEXT,
    what_could_be_improved TEXT,
    how_many_hours_it_took DECIMAL(4,2),
    last_updated DATETIME,
    datetime_originally_submitted DATETIME
);
```

## Configuration

### Database Configuration
Create a `config/database_config.py` file:

```python
def get_db_config():
    return {
        'host': 'your_host',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database',
        'port': 3306
    }
```

### Environment Variables
Alternatively, use environment variables:
- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `DB_PORT`

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- nltk
- textblob
- wordcloud
- sqlalchemy
- pymysql
- scipy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit for the web interface
- Uses NLTK for natural language processing
- Plotly for interactive visualizations
- Scikit-learn for machine learning capabilities

## Screenshots

[Add screenshots of your dashboard here]

## Support

For support, please open an issue in the GitHub repository or contact [your-email@example.com].