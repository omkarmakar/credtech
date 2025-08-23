import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Alpha Vantage API client for financial overview"""

    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"

    def get_company_overview(self, symbol: str) -> Dict:
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            if 'Information' in data or 'Error Message' in data:
                logger.warning(f"Alpha Vantage API limit or error for {symbol}")
                return self._get_fallback_data(symbol)
            return data
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error for {symbol}: {e}")
            return self._get_fallback_data(symbol)

    def _get_fallback_data(self, symbol: str) -> Dict:
        # Fallback synthetic data
        return {
            'Symbol': symbol,
            'MarketCapitalization': str(np.random.randint(1e9, 5e11)),
            'PERatio': str(round(np.random.uniform(5, 50), 2)),
            'ProfitMargin': str(round(np.random.uniform(0.01, 0.3), 4)),
            'ReturnOnAssetsTTM': str(round(np.random.uniform(-0.1, 0.2), 4)),
            'ReturnOnEquityTTM': str(round(np.random.uniform(-0.2, 0.4), 4)),
            'BookValue': str(round(np.random.uniform(5, 200), 2)),
            'DividendYield': str(round(np.random.uniform(0, 0.08), 4)),
            'EPS': str(round(np.random.uniform(-5, 20), 2)),
            'Beta': str(round(np.random.uniform(0.5, 2.5), 2)),
        }

class FinancialModelingPrepClient:
    """FMP API client for detailed financial statements"""

    def __init__(self):
        self.api_key = os.getenv('FMP_KEY')
        self.base_url = 'https://financialmodelingprep.com/api/v3'

    def get_financials(self, symbol: str) -> Dict:
        try:
            url = f"{self.base_url}/financials/income-statement/{symbol}"
            params = {'apikey': self.api_key}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if 'financials' not in data or not data['financials']:
                logger.warning(f"FMP no financials for {symbol}")
                return {}
            latest = data['financials'][0]
            # Extract relevant fields, handle missing keys
            return {
                'free_cash_flow': float(latest.get('Free Cash Flow', 0)),
                'net_income': float(latest.get('Net Income', 0)),
                'capex': float(latest.get('Capital Expenditure', 0)),
                'depreciation': float(latest.get('Depreciation & Amortization', 0)),
                'ebitda': float(latest.get('EBITDA', 0)),
                'revenue': float(latest.get('Revenue', 0)),
            }
        except Exception as e:
            logger.error(f"FMP fetch error for {symbol}: {e}")
            return {}

class FinnhubClient:
    """Finnhub client for market data"""

    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"

    def get_market_data(self, symbol: str) -> Dict:
        try:
            url = f"{self.base_url}/stock/metric"
            params = {'symbol': symbol, 'metric': 'all', 'token': self.api_key}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if 'metric' not in data:
                logger.warning(f"Finnhub missing metric data for {symbol}")
                return {}
            metric = data['metric']
            return {
                'total_debt': float(metric.get('totalDebt', 0)),
                'market_debt': float(metric.get('debt', 0)),
                'market_equity': float(metric.get('marketCapitalization', 0)),
                'interest_expense': float(metric.get('interestExpense', 0)),
                'current_assets': float(metric.get('currentAssets', 0)),
                'current_liabilities': float(metric.get('currentLiabilities', 0)),
                'beta': float(metric.get('beta', 0)),
                'shares_short': float(metric.get('shortPercentOutstanding', 0)) * 1e6,  # approximate
                'avg_daily_volume': float(metric.get('averageDailyVolume', 0))
            }
        except Exception as e:
            logger.error(f"Finnhub fetch error for {symbol}: {e}")
            return {}

class NewsClient:
    """News API client for sentiment analysis"""

    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.client = NewsApiClient(api_key=self.api_key) if self.api_key else None
        self.vader = SentimentIntensityAnalyzer()

    def get_company_news(self, company_name: str, days_back: int = 7) -> List[Dict]:
        try:
            if self.client:
                articles = self.client.get_everything(
                    q=company_name,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20
                )
                news_items = articles.get('articles', [])
            else:
                news_items = self._generate_synthetic_news(company_name)
            analyzed_news = []
            for item in news_items[:10]:
                content = item.get('title', '') + ' ' + (item.get('description') or '')
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
        vader_scores = self.vader.polarity_scores(text)
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
                'date': f"2024-08-{21 - i}",
                'sentiment': sentiment
            })
        return synthetic_news

class FeatureEngineer:
    def extract_financial_features(self, company_data: Dict, fmp_data: Dict = None, finnhub_data: Dict = None) -> Dict:
        features = {}

        # Extract from Alpha Vantage company_data
        try:
            if company_data:
                features['profit_margin'] = float(company_data.get('ProfitMargin', 0) or 0)
                features['roa'] = float(company_data.get('ReturnOnAssetsTTM', 0) or 0)
                features['roe'] = float(company_data.get('ReturnOnEquityTTM', 0) or 0)
                features['pe_ratio'] = float(company_data.get('PERatio', 15) or 15)
                features['book_value'] = float(company_data.get('BookValue', 10) or 10)
                market_cap = float(company_data.get('MarketCapitalization', 1e9) or 1e9)
                features['log_market_cap'] = np.log(market_cap) if market_cap > 0 else 0
                features['beta'] = float(company_data.get('Beta', 1) or 1)
                features['dividend_yield'] = float(company_data.get('DividendYield', 0) or 0)
                features['eps'] = float(company_data.get('EPS', 1) or 1)
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage features: {e}")

        # Override or add from FMP data if available
        if fmp_data:
            try:
                features['free_cash_flow'] = float(fmp_data.get('free_cash_flow', 0) or 0)
                features['net_income'] = float(fmp_data.get('net_income', 0) or 0)
                features['capex'] = float(fmp_data.get('capex', 0) or 0)
                features['depreciation'] = float(fmp_data.get('depreciation', 0) or 0)
                features['ebitda'] = float(fmp_data.get('ebitda', 0) or 0)
                features['revenue'] = float(fmp_data.get('revenue', 0) or 0)
            except Exception as e:
                logger.error(f"Error parsing FMP features: {e}")

        # Add / override from Finnhub market data if available
        if finnhub_data:
            try:
                features['total_debt'] = float(finnhub_data.get('total_debt', 0) or 0)
                features['market_debt'] = float(finnhub_data.get('market_debt', 0) or 0)
                features['market_equity'] = float(finnhub_data.get('market_equity', 0) or 0)
                features['interest_expense'] = float(finnhub_data.get('interest_expense', 0) or 0)
                features['current_assets'] = float(finnhub_data.get('current_assets', 0) or 0)
                features['current_liabilities'] = float(finnhub_data.get('current_liabilities', 0) or 0)
                features['beta'] = float(finnhub_data.get('beta', features.get('beta', 1)) or features.get('beta', 1))
                features['shares_short'] = float(finnhub_data.get('shares_short', 0) or 0)
                features['avg_daily_volume'] = float(finnhub_data.get('avg_daily_volume', 0) or 0)
            except Exception as e:
                logger.error(f"Error parsing Finnhub features: {e}")

        # Ensure all required features exist with defaults
        required_features = ['free_cash_flow', 'net_income', 'capex', 'depreciation', 'dio', 'dso', 'dpo', 'market_debt',
                             'market_equity', 'total_debt', 'ebitda', 'interest_expense', 'op_cf', 'cash_eq', 'current_assets',
                             'inventory', 'current_liabilities', 'issuer_yield', 'benchmark_yield', 'beta', 'shares_short',
                             'avg_daily_volume', 'profit_margin', 'roa', 'roe', 'pe_ratio', 'book_value', 'dividend_yield', 'eps',
                             'revenue']

        for feat in required_features:
            if feat not in features:
                features[feat] = 0.0

        return features

    def extract_sentiment_features(self, news_data: List[Dict]) -> Dict:
        if not news_data:
            return {
                'avg_sentiment': 0.0,
                'sentiment_volatility': 0.1,
                'news_volume': 0,
                'positive_ratio': 0.5
            }
        sentiments = [item.get('sentiment', {}).get('compound', 0) for item in news_data]
        return {
            'avg_sentiment': float(np.mean(sentiments)),
            'sentiment_volatility': float(np.std(sentiments)) if len(sentiments) > 1 else 0.1,
            'news_volume': len(news_data),
            'positive_ratio': float(len([s for s in sentiments if s > 0]) / len(sentiments)),
        }

    def create_feature_vector(self, financial_features: Dict, sentiment_features: Dict,
                              model_feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        all_features = {**financial_features, **sentiment_features}
        all_features['sentiment_pe_interaction'] = all_features.get('avg_sentiment', 0) * all_features.get('pe_ratio', 15)
        all_features['volatility_beta_interaction'] = all_features.get('sentiment_volatility', 0.1) * all_features.get('beta', 1.0)
        if model_feature_names is not None:
            for feat in model_feature_names:
                all_features.setdefault(feat, 0.0)
            ordered_vector = {f: all_features[f] for f in model_feature_names}
            return pd.DataFrame([ordered_vector])
        else:
            return pd.DataFrame([all_features])
