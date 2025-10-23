# Deployment Guide

## Overview

This guide covers different deployment options for the Feedback Analysis Dashboard, from local development to production cloud deployment.

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/feedback-analysis-dashboard.git
cd feedback-analysis-dashboard
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.template .env
# Edit .env with your database credentials
```

5. **Run the application:**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Clone and configure:**
```bash
git clone https://github.com/yourusername/feedback-analysis-dashboard.git
cd feedback-analysis-dashboard
cp .env.template .env
# Edit .env with your configuration
```

2. **Start services:**
```bash
docker-compose up -d
```

This will start:
- The Streamlit application on port 8501
- MySQL database on port 3306

3. **Access the application:**
Open `http://localhost:8501` in your browser

4. **Stop services:**
```bash
docker-compose down
```

### Using Docker Only

1. **Build the image:**
```bash
docker build -t feedback-dashboard .
```

2. **Run the container:**
```bash
docker run -p 8501:8501 \
  -e DB_HOST=your_db_host \
  -e DB_USER=your_db_user \
  -e DB_PASSWORD=your_db_password \
  -e DB_NAME=your_db_name \
  feedback-dashboard
```

## Cloud Deployment

### Streamlit Cloud

1. **Fork the repository** on GitHub

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Deploy your app:**
   - Connect your GitHub account
   - Select your forked repository
   - Choose the main branch
   - Set the main file path: `app.py`

4. **Configure secrets:**
   In the Streamlit Cloud dashboard, add these secrets:
   ```toml
   [database]
   DB_HOST = "your_host"
   DB_USER = "your_user"
   DB_PASSWORD = "your_password"
   DB_NAME = "your_database"
   DB_PORT = "3306"
   ```

### Heroku

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create Heroku app:**
```bash
heroku create your-app-name
```

3. **Set environment variables:**
```bash
heroku config:set DB_HOST=your_host
heroku config:set DB_USER=your_user
heroku config:set DB_PASSWORD=your_password
heroku config:set DB_NAME=your_database
heroku config:set DB_PORT=3306
```

4. **Create Procfile:**
```bash
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

5. **Deploy:**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### AWS EC2

1. **Launch EC2 instance:**
   - Choose Ubuntu 20.04 LTS
   - Select appropriate instance type (t3.medium recommended)
   - Configure security group to allow HTTP (80) and HTTPS (443)

2. **Connect to instance and setup:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/yourusername/feedback-analysis-dashboard.git
cd feedback-analysis-dashboard
```

3. **Configure environment:**
```bash
cp .env.template .env
nano .env  # Edit with your configuration
```

4. **Deploy with Docker Compose:**
```bash
docker-compose up -d
```

5. **Set up reverse proxy (optional):**
Install and configure Nginx to proxy requests to the Streamlit app.

### Google Cloud Platform

1. **Enable required APIs:**
   - Cloud Run API
   - Container Registry API

2. **Build and push image:**
```bash
# Configure gcloud
gcloud auth configure-docker

# Build and tag image
docker build -t gcr.io/your-project-id/feedback-dashboard .

# Push to Container Registry
docker push gcr.io/your-project-id/feedback-dashboard
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy feedback-dashboard \
  --image gcr.io/your-project-id/feedback-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DB_HOST=your_host,DB_USER=your_user,DB_PASSWORD=your_password,DB_NAME=your_database
```

## Database Setup

### MySQL Setup

1. **Create database:**
```sql
CREATE DATABASE feedback_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. **Run schema:**
```bash
mysql -u root -p feedback_db < sql/schema.sql
```

3. **Import sample data (optional):**
```bash
python bulk_insert.py
```

### Cloud Database Options

#### AWS RDS
- Create MySQL 8.0 instance
- Configure security groups
- Use connection details in environment variables

#### Google Cloud SQL
- Create MySQL instance
- Configure authorized networks
- Use connection details in environment variables

#### Azure Database for MySQL
- Create MySQL server
- Configure firewall rules
- Use connection details in environment variables

## Environment Variables

### Required Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=feedback_db
DB_PORT=3306

# Application Configuration (optional)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Security Considerations

1. **Never commit sensitive data:**
   - Use `.env` files for local development
   - Use cloud provider secret management for production
   - Keep `.env` in `.gitignore`

2. **Database security:**
   - Use strong passwords
   - Enable SSL connections
   - Restrict network access
   - Regular backups

3. **Application security:**
   - Keep dependencies updated
   - Use HTTPS in production
   - Implement proper authentication if needed

## Monitoring and Logging

### Application Monitoring

1. **Health checks:**
   - Streamlit provides built-in health endpoint
   - Monitor application responsiveness

2. **Error tracking:**
   - Implement Sentry for error tracking
   - Monitor application logs

3. **Performance monitoring:**
   - Monitor memory usage
   - Track response times
   - Monitor database connections

### Database Monitoring

1. **Connection monitoring:**
   - Monitor active connections
   - Track query performance

2. **Resource monitoring:**
   - Monitor CPU and memory usage
   - Track storage usage

## Scaling Considerations

### Horizontal Scaling

1. **Load balancing:**
   - Use multiple application instances
   - Implement load balancer (Nginx, AWS ALB, etc.)

2. **Database scaling:**
   - Read replicas for read-heavy workloads
   - Connection pooling

### Vertical Scaling

1. **Application scaling:**
   - Increase CPU and memory
   - Optimize Streamlit configuration

2. **Database scaling:**
   - Increase instance size
   - Optimize queries and indexes

## Backup and Recovery

### Database Backups

1. **Automated backups:**
   - Configure daily automated backups
   - Test restore procedures

2. **Manual backups:**
```bash
mysqldump -u username -p feedback_db > backup.sql
```

### Application Backups

1. **Code repository:**
   - Use Git for version control
   - Tag releases

2. **Configuration:**
   - Backup environment configurations
   - Document deployment procedures

## Troubleshooting

### Common Issues

1. **Database connection errors:**
   - Check credentials and network connectivity
   - Verify database server is running
   - Check firewall settings

2. **Memory issues:**
   - Monitor memory usage during large data processing
   - Implement data pagination
   - Optimize caching

3. **Performance issues:**
   - Profile slow queries
   - Optimize data processing
   - Implement caching strategies

### Debugging

1. **Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check application logs:**
```bash
# Docker logs
docker-compose logs -f

# Streamlit logs
streamlit run app.py --logger.level=debug
```

## Maintenance

### Regular Tasks

1. **Update dependencies:**
```bash
pip list --outdated
pip install -r requirements.txt --upgrade
```

2. **Database maintenance:**
   - Regular backups
   - Index optimization
   - Clean up old data

3. **Security updates:**
   - Keep OS updated
   - Update application dependencies
   - Review security configurations

### Monitoring Checklist

- [ ] Application is accessible
- [ ] Database connections are working
- [ ] No error logs
- [ ] Performance metrics are normal
- [ ] Backups are running
- [ ] SSL certificates are valid (if applicable)