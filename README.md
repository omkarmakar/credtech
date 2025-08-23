# CredTech - Explainable Credit Intelligence Platform

## 🚀 Overview

CredTech is a real-time explainable credit intelligence platform that continuously ingests multi-source financial data to generate dynamic creditworthiness scores. Unlike traditional credit rating agencies that update infrequently, our platform provides real-time, explainable credit assessments with transparent feature-level explanations.

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Feature Engine  │───▶│  ML Pipeline    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ • Alpha Vantage │    │ • Financial      │    │ • CatBoost      │
│ • News API      │    │   Metrics        │    │ • Neural Nets   │
│ • Finnhub       │    │ • Sentiment      │    │ • SHAP          │
│ • FMP           │    │   Analysis       │    │ • Ensemble      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                          │
│  • Risk Gauges  • SHAP Waterfall  • News Sentiment Timeline     │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

**Data Ingestion Layer:**
- **Structured Data**: Alpha Vantage (financial overview), Finnhub (market data), FMP (financial statements)
- **Unstructured Data**: News API (sentiment analysis), real-time news processing
- **Rate Limiting**: Built-in fallback mechanisms and caching to handle API limits

**Feature Engineering:**
- Financial ratios calculation (FCF/NI, Debt/EBITDA, Quick Ratio, etc.)
- Sentiment analysis using VADER and TextBlob
- Corporate metrics computation with Black-Cox probability of default
- Time-series feature extraction

**ML Pipeline:**
- **Ensemble Model**: CatBoost + Neural Networks + Risk Score ANN
- **Explainability**: SHAP TreeExplainer for feature importance
- **Architecture**: Multi-modal learning with graph embeddings and text embeddings (BERT)
- **Training**: Incremental learning capability with model persistence

**Presentation Layer:**
- Interactive Streamlit dashboard
- Real-time score updates with explanations
- Visualization suite (Plotly-based gauges, waterfalls, timelines)

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.11**: Core runtime environment
- **PyTorch**: Deep learning framework for neural networks
- **CatBoost**: Gradient boosting for structured data
- **Transformers**: BERT embeddings for text analysis
- **PyTorch Geometric**: Graph neural networks
- **SHAP**: Model explainability
- **scikit-learn**: Feature preprocessing and metrics

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Requests**: HTTP API calls
- **TextBlob & VADER**: Sentiment analysis
- **Alpha Vantage, News API, Finnhub**: External data sources

### Frontend & Visualization
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Custom CSS**: Enhanced UI/UX

### DevOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Joblib**: Model serialization
- **Python-dotenv**: Environment management

### Rationale for Tech Stack Selection

1. **Streamlit over Flask/FastAPI**: Rapid prototyping for ML dashboards with built-in interactivity
2. **CatBoost + PyTorch Ensemble**: CatBoost excels at tabular data while PyTorch handles multi-modal inputs
3. **SHAP for Explainability**: Industry standard for model interpretability without LLM dependency
4. **Docker**: Ensures reproducible deployments across environments
5. **Multiple API Sources**: Diversified data pipeline reduces single-point-of-failure risk

## 🐳 Installation & Setup

### Docker Installation (Recommended)

#### Prerequisites
- Docker Engine 20.0+
- Docker Compose 1.29+
- Git

#### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credtech.git
cd credtech
```

2. **Set up environment variables**
```bash
# Create .env file with your API keys
cat > .env << EOF
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
FINNHUB_API_KEY=your_finnhub_key
FMP_KEY=your_fmp_key
TWELVEDATA_API_KEY=your_twelvedata_key
EOF
```

3. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- Open browser to `http://localhost:8501`
- The application will automatically train models on first run

#### Docker Configuration Details

**Dockerfile Structure:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

**Health Checks:**
- Built-in health monitoring via Streamlit's `/_stcore/health`
- Automatic container restart on failure
- 30s interval health checks with 3 retries

**Volume Mounts:**
- `./models:/app/models` - Persistent model storage
- `./data:/app/data` - Data cache directory

### Local Installation (Alternative)

#### Prerequisites
- Python 3.11 (3.9-3.11 supported, avoid 3.13 due to dependency conflicts)
- pip or conda

#### Setup Steps

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🔑 API Configuration

### Required API Keys

| Service | Purpose | Free Tier Limits | Signup URL |
|---------|---------|------------------|------------|
| Alpha Vantage | Company financials | 5 calls/min, 500/day | [alphavantage.co](https://alphavantage.co) |
| News API | News sentiment | 1000 requests/day | [newsapi.org](https://newsapi.org) |
| Finnhub | Market data | 60 calls/min | [finnhub.io](https://finnhub.io) |
| FMP | Financial statements | 250 calls/day | [financialmodelingprep.com](https://financialmodelingprep.com) |

### Configuration Options

**For Streamlit Cloud:**
- Add keys to Streamlit secrets management
- Access via `st.secrets["KEY_NAME"]`

**For Local Development:**
- Use `.env` file with `python-dotenv`
- Environment variables loaded automatically

**For Docker:**
- Pass via environment variables in docker-compose.yml
- Supports `.env` file in project root

## 🤖 Model Architecture

### Advanced Credit Scoring Pipeline

**Multi-Model Ensemble:**
1. **Risk Score ANN**: 32→16→1 neural network for base risk assessment
2. **CatBoost Classifier**: Gradient boosting on engineered features
3. **Main Neural Network**: 128→64→1 with dropout and batch normalization
4. **Ensemble Averaging**: Weighted combination of model predictions

**Feature Engineering:**
- **Financial Metrics**: FCF/NI ratio, Debt/EBITDA, Quick Ratio, Market Leverage
- **Structural Model**: Black-Cox probability of default calculation  
- **Sentiment Features**: News sentiment aggregation and volatility
- **Graph Embeddings**: Company relationship networks (16-dim)
- **Text Embeddings**: BERT-based document representations (768-dim)

**Explainability Layer:**
- **SHAP Values**: Feature contribution analysis
- **Waterfall Charts**: Visual impact breakdown
- **Risk Factor Identification**: Top positive/negative contributors
- **Plain Language Summaries**: Non-technical explanations

### Model Performance Metrics
- **Ensemble AUC**: >0.85 on validation set
- **Training Time**: ~2-3 minutes on CPU
- **Inference Time**: <100ms per prediction
- **Model Size**: ~50MB serialized

## 📊 Key Features

### Real-Time Credit Scoring
- **Dynamic Updates**: Scores react to market events within minutes
- **Multi-Factor Analysis**: 30+ engineered features from diverse data sources
- **Risk Categorization**: Low/Medium/High risk classification with confidence scores

### Explainable AI
- **SHAP Integration**: Feature-level impact analysis without black-box explanations
- **Visual Explanations**: Interactive charts showing "why this score"
- **Trend Analysis**: Historical risk evolution tracking
- **Event Attribution**: Links score changes to specific news/market events

### Interactive Dashboard
- **Risk Gauges**: Real-time creditworthiness visualization
- **News Sentiment Timeline**: Market event impact tracking  
- **Feature Importance**: Dynamic ranking of risk factors
- **Company Comparison**: Side-by-side risk analysis

### Data Integration
- **Multi-Source Fusion**: Combines financial statements, market data, and news
- **Rate Limit Handling**: Intelligent caching and fallback mechanisms
- **Data Quality**: Automated cleaning and normalization pipelines
- **Scalability**: Designed for dozens of entities across sectors

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet for API calls

### Recommended for Production
- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 16GB
- **Storage**: 10GB SSD
- **Network**: Low-latency connection (< 100ms to API endpoints)

## 🚨 Known Limitations & Trade-offs

### API Dependencies
- **Rate Limits**: Free tier APIs limit real-time capabilities
- **Data Quality**: Dependent on external API reliability
- **Cost Scaling**: Production usage requires paid API tiers

### Model Limitations  
- **Training Data**: Uses synthetic data for demonstration
- **Cold Start**: New entities require initial data accumulation
- **Market Coverage**: Optimized for US equity markets

### Technical Constraints
- **Python 3.13 Incompatibility**: UMAP/Numba dependencies limit Python version
- **Memory Usage**: BERT models require significant RAM
- **Compute Requirements**: Real-time inference needs adequate CPU

### Architectural Trade-offs

**Ensemble vs Single Model:**
- ✅ **Chosen**: Ensemble approach for better accuracy and robustness
- ❌ **Rejected**: Single model for simplicity (sacrifices performance)

**Streamlit vs Custom Frontend:**
- ✅ **Chosen**: Streamlit for rapid prototyping and ML-focused UI
- ❌ **Rejected**: React/Vue for production-grade UX (development time)

**Docker vs Native Deployment:**  
- ✅ **Chosen**: Docker for reproducible, portable deployments
- ❌ **Rejected**: Native installation (environment conflicts)

## 📈 Future Enhancements

### Planned Features
- **Real-time WebSocket Updates**: Live score streaming
- **Advanced ML Models**: Transformer-based time series models
- **Extended Market Coverage**: International markets and bonds
- **Alert System**: Configurable risk threshold notifications
- **Historical Backtesting**: Strategy performance analysis

### Scalability Roadmap
- **Microservices Architecture**: Separate data, model, and UI services
- **Database Integration**: PostgreSQL/TimescaleDB for historical data
- **Kubernetes Deployment**: Container orchestration for production
- **CDN Integration**: Global content delivery optimization

## 🤝 Contributors  

We would like to thank all the amazing contributors who have been part of this project:  

- **Om Karmakar** - [omkarmakar07@gmail.com](mailto:omkarmakar07@gmail.com)  
- **Jotiraditya Banerjee** - [joti.ban.2710@gmail.com](mailto:joti.ban.2710@gmail.com)  
- **Rudra Ray** - [itisrudraray@gmail.com](mailto:itisrudraray@gmail.com)  
- **Oyshi Mukherjee** - [oyshi0911@gmail.com](mailto:oyshi0911@gmail.com)  
- **Kingshuk Bhandary** - [kingshukbhandaryedm@gmail.com](mailto:kingshukbhandaryedm@gmail.com)  


## 🏆 Hackathon Context

Developed for the **CredTech Hackathon** organized by The Programming Club, IIT Kanpur, and powered by Deep Root Investments. This platform addresses the challenge of creating transparent, real-time credit intelligence to replace opaque traditional rating methodologies.

---

**Built with ❤️ for transparent financial intelligence**