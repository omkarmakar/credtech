import pandas as pd
import numpy as np
import requests
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
import time
import json
from typing import Dict, List, Optional
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Alpha Vantage API client for financial data"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company overview data"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            # Handle API limits
            if 'Information' in data:
                logger.warning(f"API limit reached for {symbol}")
                return self._get_fallback_data(symbol)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Fallback data when API fails"""
        return {
            'Symbol': symbol,
            'Name': f'{symbol} Corporation',
            'MarketCapitalization': str(np.random.randint(1000000000, 500000000000)),
            'PERatio': str(round(np.random.uniform(5, 50), 2)),
            'ProfitMargin': str(round(np.random.uniform(0.01, 0.3), 4)),
            'ReturnOnAssetsTTM': str(round(np.random.uniform(-0.1, 0.2), 4)),
            'ReturnOnEquityTTM': str(round(np.random.uniform(-0.2, 0.4), 4)),
            'RevenueTTM': str(np.random.randint(1000000000, 100000000000)),
            'GrossProfitTTM': str(np.random.randint(500000000, 50000000000)),
            'BookValue': str(round(np.random.uniform(5, 200), 2)),
            'DividendYield': str(round(np.random.uniform(0, 0.08), 4)),
            'EPS': str(round(np.random.uniform(-5, 20), 2)),
            'Beta': str(round(np.random.uniform(0.5, 2.5), 2))
        }

class NewsClient:
    """News API client for sentiment analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.client = NewsApiClient(api_key=self.api_key) if self.api_key else None
        self.vader = SentimentIntensityAnalyzer()
        
    def get_company_news(self, company_name: str, days_back: int = 7) -> List[Dict]:
        """Get recent news about a company"""
        try:
            if self.client:
                # Real API call
                articles = self.client.get_everything(
                    q=company_name,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20
                )
                news_items = articles.get('articles', [])
            else:
                # Fallback synthetic news
                news_items = self._generate_synthetic_news(company_name)
            
            # Analyze sentiment
            analyzed_news = []
            for item in news_items[:10]:  # Limit to 10 articles
                if self.client:
                    title = item.get('title', '')
                    description = item.get('description', '') or ''
                    content = f"{title} {description}"
                else:
                    content = item['content']
                
                sentiment = self._analyze_sentiment(content)
                
                analyzed_news.append({
                    'title': item.get('title', item.get('headline', 'News Update')),
                    'description': item.get('description', item.get('content', 'Market news')),
                    'published_at': item.get('publishedAt', item.get('date', '2024-08-21')),
                    'sentiment': sentiment,
                    'url': item.get('url', '#')
                })
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {company_name}: {e}")
            return self._generate_synthetic_news(company_name)
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using VADER and TextBlob"""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'polarity': blob.polarity,
            'subjectivity': blob.subjectivity
        }
    
    def _generate_synthetic_news(self, company_name: str) -> List[Dict]:
        """Generate synthetic news for demo purposes"""
        news_templates = [
            f"{company_name} reports strong quarterly earnings",
            f"{company_name} announces new product launch",
            f"Analysts upgrade {company_name} stock rating",
            f"{company_name} faces regulatory challenges",
            f"{company_name} CEO discusses market outlook", 
            f"{company_name} expands into new markets",
            f"Market volatility affects {company_name} shares",
            f"{company_name} invests in sustainable technology",
            f"Competition intensifies for {company_name}",
            f"{company_name} board approves dividend increase"
        ]
        
        synthetic_news = []
        for i, headline in enumerate(news_templates):
            # Generate realistic sentiment based on headline
            if any(word in headline.lower() for word in ['strong', 'upgrade', 'increase', 'launch', 'expands']):
                sentiment_base = 0.3
            elif any(word in headline.lower() for word in ['challenges', 'volatility', 'competition']):
                sentiment_base = -0.2
            else:
                sentiment_base = 0.0
                
            sentiment = {
                'compound': sentiment_base + np.random.normal(0, 0.1),
                'positive': max(0, sentiment_base + 0.3 + np.random.normal(0, 0.1)),
                'negative': max(0, -sentiment_base + 0.2 + np.random.normal(0, 0.1)),
                'neutral': 0.5 + np.random.normal(0, 0.1),
                'polarity': sentiment_base + np.random.normal(0, 0.1),
                'subjectivity': 0.5 + np.random.normal(0, 0.2)
            }
            
            synthetic_news.append({
                'headline': headline,
                'content': f"Recent developments show {headline.lower()}. Market analysts are closely watching the situation.",
                'date': f"2024-08-{21-i}",
                'sentiment': sentiment
            })
        
        return synthetic_news

class FeatureEngineer:
    """Feature engineering for credit scoring"""
    
    def extract_financial_features(self, company_data: Dict) -> Dict:
        """Extract financial features from company data"""
        features = {}
        
        try:
            # Profitability metrics
            if 'ProfitMargin' in company_data:
                features['profit_margin'] = float(company_data['ProfitMargin']) if company_data['ProfitMargin'] != 'None' else 0.0
            
            if 'ReturnOnAssetsTTM' in company_data:
                features['roa'] = float(company_data['ReturnOnAssetsTTM']) if company_data['ReturnOnAssetsTTM'] != 'None' else 0.0
                
            if 'ReturnOnEquityTTM' in company_data:
                features['roe'] = float(company_data['ReturnOnEquityTTM']) if company_data['ReturnOnEquityTTM'] != 'None' else 0.0
            
            # Valuation metrics
            if 'PERatio' in company_data:
                features['pe_ratio'] = float(company_data['PERatio']) if company_data['PERatio'] != 'None' else 15.0
            
            if 'BookValue' in company_data:
                features['book_value'] = float(company_data['BookValue']) if company_data['BookValue'] != 'None' else 10.0
            
            # Market metrics
            if 'MarketCapitalization' in company_data:
                market_cap = float(company_data['MarketCapitalization']) if company_data['MarketCapitalization'] != 'None' else 1e9
                features['log_market_cap'] = np.log(market_cap)
            
            if 'Beta' in company_data:
                features['beta'] = float(company_data['Beta']) if company_data['Beta'] != 'None' else 1.0
            
            # Dividend info
            if 'DividendYield' in company_data:
                features['dividend_yield'] = float(company_data['DividendYield']) if company_data['DividendYield'] != 'None' else 0.0
            
            # EPS
            if 'EPS' in company_data:
                features['eps'] = float(company_data['EPS']) if company_data['EPS'] != 'None' else 1.0
                
        except Exception as e:
            logger.error(f"Error extracting financial features: {e}")
            # Return default features
            features = {
                'profit_margin': 0.1,
                'roa': 0.05,
                'roe': 0.1,
                'pe_ratio': 15.0,
                'book_value': 10.0,
                'log_market_cap': 20.0,
                'beta': 1.0,
                'dividend_yield': 0.02,
                'eps': 2.0
            }
        
        return features
    
    def extract_sentiment_features(self, news_data: List[Dict]) -> Dict:
        """Extract sentiment features from news data"""
        if not news_data:
            return {
                'avg_sentiment': 0.0,
                'sentiment_volatility': 0.1,
                'news_volume': 0,
                'positive_ratio': 0.5
            }
        
        sentiments = [item['sentiment']['compound'] for item in news_data]
        
        return {
            'avg_sentiment': np.mean(sentiments),
            'sentiment_volatility': np.std(sentiments) if len(sentiments) > 1 else 0.1,
            'news_volume': len(news_data),
            'positive_ratio': len([s for s in sentiments if s > 0]) / len(sentiments)
        }
    
    def create_feature_vector(self, financial_features: Dict, sentiment_features: Dict) -> pd.DataFrame:
        """Combine all features into a single vector"""
        all_features = {**financial_features, **sentiment_features}
        
        # Add interaction features
        all_features['sentiment_pe_interaction'] = all_features.get('avg_sentiment', 0) * all_features.get('pe_ratio', 15)
        all_features['volatility_beta_interaction'] = all_features.get('sentiment_volatility', 0.1) * all_features.get('beta', 1.0)
        
        return pd.DataFrame([all_features])

# Test the data pipeline
if __name__ == "__main__":
    # Test Alpha Vantage
    av_client = AlphaVantageClient()
    company_data = av_client.get_company_overview("AAPL")
    print("Company Data Sample:", list(company_data.keys())[:5])
    
    # Test News
    news_client = NewsClient()
    news_data = news_client.get_company_news("Apple")
    print(f"News Articles: {len(news_data)}")
    
    # Test Feature Engineering
    fe = FeatureEngineer()
    financial_features = fe.extract_financial_features(company_data)
    sentiment_features = fe.extract_sentiment_features(news_data)
    feature_vector = fe.create_feature_vector(financial_features, sentiment_features)
    
    print("Feature Vector Shape:", feature_vector.shape)
    print("Features:", list(feature_vector.columns))