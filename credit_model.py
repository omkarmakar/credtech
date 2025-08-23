import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import catboost as cb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
from typing import Dict, Tuple, List, Optional
from fairlearn.metrics import MetricFrame, selection_rate
from scipy.stats import norm
import os


# ---------------------------------------------------
# 1. Financial Metrics
# ---------------------------------------------------

def compute_corporate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame(index=df.index)

    # Cash-flow sustainability & quality
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['fcf_ni_ratio'] = np.where(df['net_income'] != 0, df['free_cash_flow'] / df['net_income'], 0.0)
        metrics['capex_depr'] = np.where(df['depreciation'] != 0, df['capex'] / df['depreciation'], 0.0)
    metrics['wcc'] = df['dio'] + df['dso'] - df['dpo']

    # Leverage & solvency
    denom = df['market_debt'] + df['market_equity']
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['market_leverage'] = np.where(denom != 0, df['market_debt'] / denom, 0.0)
        metrics['debt_ebitda'] = np.where(df['ebitda'] != 0, df['total_debt'] / df['ebitda'], 0.0)
        metrics['coverage'] = np.where(df['interest_expense'] != 0, df['ebitda'] / df['interest_expense'], 0.0)
        metrics['adj_coverage'] = np.where(df['interest_expense'] != 0, (df['ebitda'] - df['capex']) / df['interest_expense'], 0.0)

    # Liquidity
    metrics['cash_burn'] = np.maximum(0, -df['op_cf'])
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['runway'] = np.where(metrics['cash_burn'] != 0, df['cash_eq'] / metrics['cash_burn'], 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['quick_ratio'] = np.where(df['current_liabilities'] != 0, (df['current_assets'] - df['inventory']) / df['current_liabilities'], 0.0)

    # Market-based signals
    metrics['yield_spread'] = df['issuer_yield'] - df['benchmark_yield']
    metrics['beta'] = df['beta']
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['short_interest_ratio'] = np.where(df['avg_daily_volume'] != 0, df['shares_short'] / df['avg_daily_volume'], 0.0)

    return metrics.fillna(0)


def compute_sovereign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame(index=df.index)

    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['debt_gdp'] = np.where(df['gdp'] != 0, df['govt_debt'] / df['gdp'], 0.0)
        metrics['debt_service_rev'] = np.where(df['govt_rev'] != 0, (df['interest_due'] + df['principal_due']) / df['govt_rev'], 0.0)
        metrics['primary_balance_gdp'] = np.where(df['gdp'] != 0, (df['revenue'] - df['primary_expenditure']) / df['gdp'], 0.0)
        metrics['ca_gdp'] = np.where(df['gdp'] != 0, df['current_account'] / df['gdp'], 0.0)
        metrics['import_cover'] = np.where(df['monthly_imports'] != 0, df['fx_reserves'] / df['monthly_imports'], 0.0)
        metrics['extdebt_exports'] = np.where(df['exports'] != 0, df['external_debt'] / df['exports'], 0.0)

    metrics['sovereign_spread'] = df['govt_yield'] - df['benchmark_yield']

    return metrics.fillna(0)


# ---------------------------------------------------
# 2. Black–Cox Default Probability
# ---------------------------------------------------

def black_cox_pod(V, B, mu, sigma, T=1.0):
    """
    Black-Cox Probability of Default (Structural Credit Risk).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        A0 = np.log(V / B)
    at = mu - (sigma**2) / 2
    denom = sigma * np.sqrt(T)
    d1 = (-A0 + at * T) / denom
    d2 = (-A0 - at * T) / denom
    pod = norm.cdf(d1) + np.exp((-2 * at * A0) / (sigma**2)) * norm.cdf(d2)
    return pod


# ---------------------------------------------------
# 3. Neural Nets and Models
# ---------------------------------------------------

class CreditNN(nn.Module):
    def __init__(self, input_dim: int):
        super(CreditNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 16):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class RiskScoreANN(nn.Module):
    def __init__(self, n_features: int):
        super(RiskScoreANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------
# 4. Advanced Credit Scoring Model
# ---------------------------------------------------

class AdvancedCreditScoringModel:

    def __init__(self):
        self.catboost_model = None
        self.nn_model = None
        self.gnn_model = None
        self.risk_score_ann = None
        self.scaler = StandardScaler()
        self.feature_names = None  # List[str]
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shap_explainer = None

        # Text embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device)

    # -----------------------------------------------
    # Data generation
    # -----------------------------------------------

    def generate_training_data(self, n_samples=2000) -> Tuple[pd.DataFrame, pd.Series]:
        # synthetic company-level dataset with correct shape and simulated distributions
        df = pd.DataFrame({
            'free_cash_flow': np.random.normal(50, 20, n_samples),
            'net_income': np.random.normal(40, 15, n_samples),
            'capex': np.random.normal(20, 8, n_samples),
            'depreciation': np.random.normal(15, 5, n_samples),
            'dio': np.random.randint(30, 100, n_samples),
            'dso': np.random.randint(20, 80, n_samples),
            'dpo': np.random.randint(10, 50, n_samples),
            'market_debt': np.random.normal(200, 50, n_samples),
            'market_equity': np.random.normal(400, 100, n_samples),
            'total_debt': np.random.normal(250, 60, n_samples),
            'ebitda': np.random.normal(100, 30, n_samples),
            'interest_expense': np.random.normal(10, 3, n_samples),
            'op_cf': np.random.normal(40, 12, n_samples),
            'cash_eq': np.random.normal(60, 20, n_samples),
            'current_assets': np.random.normal(150, 40, n_samples),
            'inventory': np.random.normal(50, 15, n_samples),
            'current_liabilities': np.random.normal(100, 30, n_samples),
            'issuer_yield': np.random.normal(5, 1, n_samples),
            'benchmark_yield': np.random.normal(3, 0.5, n_samples),
            'beta': np.random.normal(1, 0.3, n_samples),
            'shares_short': np.random.randint(1000, 5000, n_samples),
            'avg_daily_volume': np.random.randint(10000, 50000, n_samples),
            'V': np.random.normal(1000, 200, n_samples),
            'B': np.random.normal(800, 150, n_samples),
        })
        X = compute_corporate_metrics(df)

        # structural PoD
        X['default_prob'] = [black_cox_pod(V, B, mu=0.05, sigma=0.25, T=1.0) for V, B in zip(df['V'], df['B'])]

        y = (X['default_prob'] > 0.3).astype(int)
        return X, y

    # -----------------------------------------------
    # Text Embeddings
    # -----------------------------------------------

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.bert_model.eval()
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.cpu().numpy()

    # -----------------------------------------------
    # Save and Load Model
    # -----------------------------------------------

    def save_model(self, path: str = "models/advanced_credit_model.joblib"):
        import joblib, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'catboost': self.catboost_model,
            'nn_state_dict': self.nn_model.state_dict() if self.nn_model else None,
            'risk_score_ann_state_dict': self.risk_score_ann.state_dict() if self.risk_score_ann else None,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)

    def load_model(self, path: str = "models/advanced_credit_model.joblib"):
        checkpoint = joblib.load(path)
        self.catboost_model = checkpoint.get('catboost', None)
        self.scaler = checkpoint.get('scaler', self.scaler)
        self.feature_names = checkpoint.get('feature_names', None)
        
        # ADD THIS BLOCK - Initialize SHAP explainer after loading CatBoost model
        if self.catboost_model is not None:
            self.shap_explainer = shap.TreeExplainer(self.catboost_model)
        else:
            self.shap_explainer = None
        
        input_dim = (len(self.feature_names) if self.feature_names else 0) + 16 + 768 + 1
        nn_state_dict = checkpoint.get('nn_state_dict', None)

        if nn_state_dict:
            self.nn_model = CreditNN(input_dim).to(self.device)
            self.nn_model.load_state_dict(nn_state_dict)
            self.nn_model.eval()

        if checkpoint.get('risk_score_ann_state_dict'):
            self.risk_score_ann = RiskScoreANN(len(self.feature_names)).to(self.device)
            self.risk_score_ann.load_state_dict(checkpoint['risk_score_ann_state_dict'])
            self.risk_score_ann.eval()

        self.is_trained = True


    
    # -----------------------------------------------
    # Train
    # -----------------------------------------------

    def train(self,
              X=None,
              y=None,
              graph_data: Optional[GraphData] = None,
              raw_texts: Optional[List[str]] = None) -> Dict:
        if X is None or y is None:
            X, y = self.generate_training_data()

        self.feature_names = X.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # RiskScore ANN
        self.risk_score_ann = RiskScoreANN(X_train_scaled.shape[1]).to(self.device)
        criterion_rs = nn.BCELoss()
        optimizer_rs = torch.optim.Adam(self.risk_score_ann.parameters(), lr=0.005)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(self.device)

        best_val_loss = float('inf')
        for epoch in range(30):
            self.risk_score_ann.train()
            optimizer_rs.zero_grad()
            out = self.risk_score_ann(X_train_tensor)
            loss = criterion_rs(out, y_train_tensor)
            loss.backward()
            optimizer_rs.step()

            self.risk_score_ann.eval()
            with torch.no_grad():
                val_out = self.risk_score_ann(X_val_tensor)
                val_loss = criterion_rs(val_out, y_val_tensor)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.risk_score_ann.state_dict(), "best_risk_ann.pth")

        self.risk_score_ann.load_state_dict(torch.load("best_risk_ann.pth"))
        self.risk_score_ann.eval()

        # Graph embeddings (if provided)
        if graph_data is not None:
            if self.gnn_model is None:
                self.gnn_model = GNN(graph_data.num_node_features).to(self.device)
            self.gnn_model.eval()
            with torch.no_grad():
                graph_embeds = self.gnn_model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
                train_graph_embeds = graph_embeds[:len(X_train)].cpu().numpy()
                val_graph_embeds = graph_embeds[len(X_train):len(X_train) + len(X_val)].cpu().numpy()
        else:
            train_graph_embeds = np.zeros((len(X_train), 16))
            val_graph_embeds = np.zeros((len(X_val), 16))

        # Text embeddings
        if raw_texts is not None:
            train_text_embeds = self.get_text_embeddings(raw_texts[:len(X_train)])
            val_text_embeds = self.get_text_embeddings(raw_texts[len(X_train):len(X_train) + len(X_val)])
        else:
            train_text_embeds = np.zeros((len(X_train), 768))
            val_text_embeds = np.zeros((len(X_val), 768))

        # Risk scores as feature
        with torch.no_grad():
            train_risk_scores = self.risk_score_ann(X_train_tensor).cpu().numpy()
            val_risk_scores = self.risk_score_ann(X_val_tensor).cpu().numpy()

        train_risk_scores = train_risk_scores.reshape(-1, 1)
        val_risk_scores = val_risk_scores.reshape(-1, 1)

        # Combine features for training and validation
        X_train_combined = np.hstack([X_train_scaled, train_graph_embeds, train_text_embeds, train_risk_scores])
        X_val_combined = np.hstack([X_val_scaled, val_graph_embeds, val_text_embeds, val_risk_scores])
        combined_input_dim = X_train_combined.shape[1]

        # CatBoost
        self.catboost_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            eval_metric='AUC',
            random_seed=42,
            verbose=0
        )
        self.catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=25)

        # Main NN
        self.nn_model = CreditNN(combined_input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)

        X_train_full = torch.tensor(X_train_combined, dtype=torch.float32).to(self.device)
        y_train_full = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_full = torch.tensor(X_val_combined, dtype=torch.float32).to(self.device)
        y_val_full = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(self.device)

        best_auc = 0
        for epoch in range(50):
            self.nn_model.train()
            optimizer.zero_grad()
            outputs = self.nn_model(X_train_full)
            loss = criterion(outputs, y_train_full)
            loss.backward()
            optimizer.step()

            self.nn_model.eval()
            with torch.no_grad():
                val_preds = self.nn_model(X_val_full).cpu().numpy()
                val_auc = roc_auc_score(y_val, val_preds)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.nn_model.state_dict(), "best_nn.pth")

        self.nn_model.load_state_dict(torch.load("best_nn.pth"))
        self.is_trained = True

        # Ensemble evaluation
        cat_preds_val = self.catboost_model.predict_proba(X_val)[:, 1]
        with torch.no_grad():
            nn_preds_val = self.nn_model(X_val_full).cpu().numpy().flatten()
        ensemble_preds = (cat_preds_val + nn_preds_val) / 2.0
        ensemble_auc = roc_auc_score(y_val, ensemble_preds)

        if self.catboost_model is not None:
            self.shap_explainer = shap.TreeExplainer(self.catboost_model)


        return {
            "catboost_val_auc": roc_auc_score(y_val, cat_preds_val),
            "nn_val_auc": best_auc,
            "ensemble_val_auc": ensemble_auc
        }

    # -----------------------------------------------
    # Predict
    # -----------------------------------------------

    def predict(self, X: pd.DataFrame,
                graph_features: np.ndarray = None,
                text_features: np.ndarray = None) -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        # Ensure all required features exist and are ordered correctly
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        if self.risk_score_ann is not None:
            with torch.no_grad():
                risk_score_feat = self.risk_score_ann(X_tensor).cpu().numpy()
        else:
            risk_score_feat = np.zeros((len(X), 1))

        if graph_features is None:
            graph_features = np.zeros((len(X), 16))
        if text_features is None:
            text_features = np.zeros((len(X), 768))

        combined_features = np.hstack([X_scaled, graph_features, text_features, risk_score_feat])
        combined_tensor = torch.tensor(combined_features, dtype=torch.float32).to(self.device)

        preds = []

        if self.catboost_model is not None:
            preds.append(self.catboost_model.predict_proba(X)[:, 1])

        if self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                preds.append(self.nn_model(combined_tensor).cpu().numpy().flatten())

        if not preds:
            raise ValueError("No prediction models available (both CatBoost and NN are None).")

        combined_pred = np.mean(preds, axis=0)
        risk_score_val = float(combined_pred[0] * 100)
        prob_high = float(combined_pred[0])

        if prob_high < 0.3:
            risk_level_val = "Low Risk"
        elif prob_high < 0.7:
            risk_level_val = "Medium Risk"
        else:
            risk_level_val = "High Risk"

        return {
            "risk_score": risk_score_val,
            "risk_level": risk_level_val,
            "probability_high_risk": prob_high
        }

    # -----------------------------------------------
    # Explainability
    # -----------------------------------------------

    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        if self.shap_explainer is None:
            return {"error": "SHAP explainer not available"}

        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.feature_names]

        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        if shap_values is None or len(shap_values) == 0:
            return {"error": "Invalid SHAP output"}

        shap_row = shap_values[0]
        contributions = {f: float(val) for f, val in zip(self.feature_names, shap_row)}

        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        positive_factors = [(n, v) for n, v in sorted_contrib if v > 0][:5]
        negative_factors = [(n, v) for n, v in sorted_contrib if v < 0][:5]

        return {
            "feature_contributions": contributions,
            "top_risk_factors": positive_factors,
            "top_protective_factors": negative_factors
        }

    # -----------------------------------------------
    # Fairness
    # -----------------------------------------------

    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray):
        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        return metric_frame.by_group

    # -----------------------------------------------

    def _get_risk_level(self, risk_score: float) -> str:
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 70:
            return "Medium Risk"
        else:
            return "High Risk"


# ---------------------------------------------------
# 5. Run Example
# ---------------------------------------------------

if __name__ == "__main__":
    model = AdvancedCreditScoringModel()
    metrics = model.train()
    print("Training Metrics:")
    for k, v in metrics.items():
        print(f" {k}: {v:.3f}")

    # Test prediction
    X_test, _ = model.generate_training_data(1)
    dummy_graph_feats = np.zeros((1, 16))
    dummy_text_feats = np.zeros((1, 768))
    prediction = model.predict(X_test, graph_features=dummy_graph_feats, text_features=dummy_text_feats)
    print(f"\nTest Prediction: {prediction}")

    explanation = model.explain_prediction(X_test)
    print(f"\nExplanation available: {'feature_contributions' in explanation}")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/advanced_credit_model.joblib")
    print("\nModel saved successfully!")