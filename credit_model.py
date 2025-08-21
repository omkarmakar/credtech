import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import shap
from typing import Dict, Tuple, List
import logging
import os
import warnings  # Added missing import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditScoringModel:
    """Credit scoring model with explainability"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.is_trained = False
        
    def generate_training_data(self, n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for credit scoring"""
        np.random.seed(42)
        
        # Generate realistic financial features
        data = {
            'profit_margin': np.random.normal(0.08, 0.15, n_samples),
            'roa': np.random.normal(0.05, 0.1, n_samples),
            'roe': np.random.normal(0.12, 0.2, n_samples),
            'pe_ratio': np.random.lognormal(2.5, 0.8, n_samples),
            'book_value': np.random.lognormal(2.5, 1.0, n_samples),
            'log_market_cap': np.random.normal(22, 2, n_samples),
            'beta': np.random.normal(1.0, 0.5, n_samples),
            'dividend_yield': np.random.exponential(0.025, n_samples),
            'eps': np.random.normal(2.0, 5.0, n_samples),
            'avg_sentiment': np.random.normal(0.0, 0.3, n_samples),
            'sentiment_volatility': np.random.exponential(0.2, n_samples),
            'news_volume': np.random.poisson(8, n_samples),
            'positive_ratio': np.random.beta(2, 2, n_samples),
            'sentiment_pe_interaction': np.zeros(n_samples),
            'volatility_beta_interaction': np.zeros(n_samples)
        }
        
        # Create DataFrame
        X = pd.DataFrame(data)
        
        # Calculate interaction features
        X['sentiment_pe_interaction'] = X['avg_sentiment'] * X['pe_ratio']
        X['volatility_beta_interaction'] = X['sentiment_volatility'] * X['beta']
        
        # Generate target variable (credit risk)
        # Higher risk companies have:
        # - Lower profitability (profit_margin, roa, roe)
        # - Higher volatility (beta, sentiment_volatility)
        # - Negative sentiment
        # - Extreme PE ratios (very high or very low)
        
        risk_score = (
            -X['profit_margin'] * 2.0 +
            -X['roa'] * 1.5 +
            -X['roe'] * 1.0 +
            np.where(X['pe_ratio'] > 30, (X['pe_ratio'] - 30) * 0.02, 0) +
            np.where(X['pe_ratio'] < 8, (8 - X['pe_ratio']) * 0.05, 0) +
            X['beta'] * 0.3 +
            -X['avg_sentiment'] * 1.2 +
            X['sentiment_volatility'] * 0.8 +
            -X['dividend_yield'] * 5.0 +
            np.where(X['eps'] < 0, 0.5, 0)
        )
        
        # Add some noise
        risk_score += np.random.normal(0, 0.2, n_samples)
        
        # Convert to binary (1 = high risk, 0 = low risk)
        # Use 70th percentile as threshold (30% high risk companies)
        threshold = np.percentile(risk_score, 70)
        y = (risk_score > threshold).astype(int)
        
        logger.info(f"Generated {n_samples} samples with {y.sum()} high-risk cases ({y.mean():.1%})")
        
        return X, y
    
    from typing import Optional

    def train(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Dict:
        """Train the credit scoring model"""
        if X is None or y is None:
            logger.info("Generating synthetic training data...")
            X, y = self.generate_training_data()
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)  # Use TreeExplainer for XGBoost
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
        
        self.is_trained = True
        
        logger.info(f"Model trained successfully. ROC AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make credit risk prediction"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet or model is None. Call train() first.")
        
        # Ensure all required features are present
        if self.feature_names is None:
            raise ValueError("Model feature names are not initialized. Train the model first.")
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0.0  # Default value for missing features
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        # Calculate risk score (0-100)
        risk_score = prediction_proba[1] * 100
        
        return {
            'prediction': int(prediction),
            'probability_low_risk': float(prediction_proba[0]),
            'probability_high_risk': float(prediction_proba[1]),
            'risk_score': float(risk_score),
            'risk_level': self._get_risk_level(risk_score)
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 70:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        """Generate SHAP explanations for prediction"""
        if self.shap_explainer is None:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Ensure all required features are present
            if self.feature_names is None:
                raise ValueError("Model feature names are not initialized. Train the model first.")
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0.0
            
            # Reorder columns to match training data
            X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(self.scaler.transform(X))  # Use shap_values for TreeExplainer

            # Extract SHAP values for the positive class (high risk)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification: shap_values[1] is for class 1
                feature_contributions = shap_values[1][0]
                if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) and len(self.shap_explainer.expected_value) > 1:
                    base_value = self.shap_explainer.expected_value[1]
                else:
                    base_value = self.shap_explainer.expected_value
            else:
                # Single output
                feature_contributions = shap_values[0]
                base_value = self.shap_explainer.expected_value

            # Create feature contribution dictionary
            contributions = {}
            for i, feature_name in enumerate(self.feature_names):
                contributions[feature_name] = float(feature_contributions[i])
            
            # Sort by absolute contribution
            sorted_contributions = sorted(contributions.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
            
            # Get top positive and negative factors
            positive_factors = [(name, val) for name, val in sorted_contributions if val > 0][:5]
            negative_factors = [(name, val) for name, val in sorted_contributions if val < 0][:5]
            
            return {
                'feature_contributions': contributions,
                'base_value': float(base_value[0]) if (isinstance(base_value, (list, np.ndarray)) and base_value is not None and len(base_value) > 0 and isinstance(base_value[0], (int, float, np.floating))) else (float(base_value) if isinstance(base_value, (int, float, np.floating)) else None),
                'top_risk_factors': positive_factors,
                'top_protective_factors': negative_factors,
                'explanation_summary': self._generate_explanation_summary(
                    positive_factors, negative_factors
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return {"error": f"Could not generate explanations: {str(e)}"}
    
    def _generate_explanation_summary(self, positive_factors: List, negative_factors: List) -> Dict:
        """Generate human-readable explanation summary"""
        
        # Feature descriptions for better readability
        feature_descriptions = {
            'profit_margin': 'Profit Margin',
            'roa': 'Return on Assets',
            'roe': 'Return on Equity',
            'pe_ratio': 'Price-to-Earnings Ratio',
            'book_value': 'Book Value per Share',
            'log_market_cap': 'Market Capitalization',
            'beta': 'Stock Beta (Volatility)',
            'dividend_yield': 'Dividend Yield',
            'eps': 'Earnings per Share',
            'avg_sentiment': 'News Sentiment',
            'sentiment_volatility': 'News Sentiment Volatility',
            'news_volume': 'News Volume',
            'positive_ratio': 'Positive News Ratio',
            'sentiment_pe_interaction': 'Sentiment-Valuation Interaction',
            'volatility_beta_interaction': 'Market-News Volatility Interaction'
        }
        
        risk_explanations = []
        for feature, contribution in positive_factors:
            desc = feature_descriptions.get(feature, feature.replace('_', ' ').title())
            risk_explanations.append(f"{desc} increases risk (impact: {contribution:.3f})")
        
        protective_explanations = []
        for feature, contribution in negative_factors:
            desc = feature_descriptions.get(feature, feature.replace('_', ' ').title())
            protective_explanations.append(f"{desc} reduces risk (impact: {abs(contribution):.3f})")
        
        return {
            'risk_factors': risk_explanations,
            'protective_factors': protective_explanations
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data.get('is_trained', True)
        
        # Reinitialize SHAP explainer if model is loaded
        if self.is_trained:
            try:
                # Generate sample data for SHAP explainer initialization
                X_sample, _ = self.generate_training_data(100)
                X_sample_scaled = self.scaler.transform(X_sample)
                self.shap_explainer = shap.TreeExplainer(self.model)  # Use TreeExplainer for XGBoost
                logger.info("SHAP explainer reinitialized")
            except Exception as e:
                logger.warning(f"Could not reinitialize SHAP explainer: {e}")
                self.shap_explainer = None
        
        logger.info(f"Model loaded from {filepath}")

# Test the model
if __name__ == "__main__":
    # Initialize and train model
    model = CreditScoringModel()
    metrics = model.train()
    
    print("Training Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test prediction
    X_test, _ = model.generate_training_data(1)
    prediction = model.predict(X_test)
    print(f"\nTest Prediction: {prediction}")  # Fixed incorrect escaping
    
    # Test explanation
    explanation = model.explain_prediction(X_test)
    print(f"\nExplanation available: {'feature_contributions' in explanation}")  # Fixed incorrect escaping
    
    # Save model
    model.save_model('models/credit_model.joblib')
    print("\nModel saved successfully!")  # Fixed incorrect escaping