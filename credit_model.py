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
from typing import Dict, Tuple, List
from fairlearn.metrics import MetricFrame, selection_rate


# Neural Network for tabular + embeddings input
class CreditNN(nn.Module):
    def _init_(self, input_dim: int):
        super(CreditNN, self)._init_()
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

    def forward(self, x, dropout_enabled=False):
        if dropout_enabled:
            self.network.train()  # Enable dropout during inference
        else:
            self.network.eval()
        return self.network(x)


# Graph Neural Network for graph embeddings
class GNN(nn.Module):
    def _init_(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 16):
        super(GNN, self)._init_()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Dynamic risk score ANN replacing fixed coefficients
class RiskScoreANN(nn.Module):
    def _init_(self, n_features: int):
        super(RiskScoreANN, self)._init_()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )

    def forward(self, x):
        return self.network(x)


class AdvancedCreditScoringModel:
    def _init_(self):
        self.catboost_model = None
        self.nn_model = None
        self.gnn_model = None
        self.risk_score_ann = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shap_explainer = None

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def generate_training_data(self, n_samples=2000) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(42)
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
            'positive_ratio': np.random.beta(2, 2, n_samples)
        }
        X = pd.DataFrame(data)
        # No handcrafted risk score here, target uses real labels or synthetic as before
        # Use same generation as previous for labels for backward compatibility
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
        risk_score += np.random.normal(0, 0.2, n_samples)
        threshold = np.percentile(risk_score, 70)
        y = (risk_score > threshold).astype(int)
        return X, y

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.bert_model.eval()
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embeddings = outputs.pooler_output.cpu().numpy()
        return embeddings

    def train(self, X=None, y=None,
              graph_data: GraphData = None,
              raw_texts: List[str] = None) -> Dict:
        if X is None or y is None:
            X, y = self.generate_training_data()
        self.feature_names = X.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Scale tabular features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train RiskScoreANN on tabular only to learn dynamic weights
        self.risk_score_ann = RiskScoreANN(X_train_scaled.shape[1]).to(self.device)
        criterion_rs = nn.BCELoss()
        optimizer_rs = torch.optim.Adam(self.risk_score_ann.parameters(), lr=0.005)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(self.device)

        best_val_loss = float('inf')
        for epoch in range(50):
            self.risk_score_ann.train()
            optimizer_rs.zero_grad()
            outputs_rs = self.risk_score_ann(X_train_tensor)
            loss_rs = criterion_rs(outputs_rs, y_train_tensor)
            loss_rs.backward()
            optimizer_rs.step()

            self.risk_score_ann.eval()
            with torch.no_grad():
                val_outputs_rs = self.risk_score_ann(X_val_tensor)
                val_loss_rs = criterion_rs(val_outputs_rs, y_val_tensor)
            if val_loss_rs < best_val_loss:
                best_val_loss = val_loss_rs
                torch.save(self.risk_score_ann.state_dict(), "best_risk_ann.pth")

        self.risk_score_ann.load_state_dict(torch.load("best_risk_ann.pth"))
        self.risk_score_ann.eval()

        # Obtain graph embeddings if graph data provided
        if graph_data is not None:
            if self.gnn_model is None:
                self.gnn_model = GNN(graph_data.num_node_features).to(self.device)
                self.gnn_model.eval()
            with torch.no_grad():
                graph_embeds = self.gnn_model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            train_graph_embeds = graph_embeds[:len(X_train)].cpu().numpy()
            val_graph_embeds = graph_embeds[len(X_train):len(X_train)+len(X_val)].cpu().numpy()
        else:
            train_graph_embeds = val_graph_embeds = np.zeros((len(X_train), 16))

        # Text embeddings
        if raw_texts is not None:
            train_text_embeds = self.get_text_embeddings(raw_texts[:len(X_train)])
            val_text_embeds = self.get_text_embeddings(raw_texts[len(X_train):len(X_train)+len(X_val)])
        else:
            train_text_embeds = val_text_embeds = np.zeros((len(X_train), 768))

        # Get risk score ANN outputs as feature
        with torch.no_grad():
            train_risk_scores = self.risk_score_ann(X_train_tensor).cpu().numpy()
            val_risk_scores = self.risk_score_ann(X_val_tensor).cpu().numpy()

        # Combine inputs for final NN
        X_train_combined = np.hstack([X_train_scaled, train_graph_embeds, train_text_embeds, train_risk_scores])
        X_val_combined = np.hstack([X_val_scaled, val_graph_embeds, val_text_embeds, val_risk_scores])
        combined_input_dim = X_train_combined.shape[1]

        # Train CatBoost on tabular features only
        self.catboost_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            eval_metric='AUC',
            random_seed=42,
            verbose=0
        )
        self.catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=25)

        # Train main NN on combined features (including dynamic risk score output)
        self.nn_model = CreditNN(combined_input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)

        X_train_tensor_full = torch.tensor(X_train_combined, dtype=torch.float32).to(self.device)
        y_train_tensor_full = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_tensor_full = torch.tensor(X_val_combined, dtype=torch.float32).to(self.device)
        y_val_tensor_full = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(self.device)

        best_auc = 0
        for epoch in range(100):
            self.nn_model.train()
            optimizer.zero_grad()
            outputs = self.nn_model(X_train_tensor_full)
            loss = criterion(outputs, y_train_tensor_full)
            loss.backward()
            optimizer.step()

            self.nn_model.eval()
            with torch.no_grad():
                val_preds = self.nn_model(X_val_tensor_full).cpu().numpy()
                val_auc = roc_auc_score(y_val, val_preds)
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(self.nn_model.state_dict(), "best_nn.pth")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, NN Val AUC: {val_auc:.4f}")

        self.nn_model.load_state_dict(torch.load("best_nn.pth"))
        self.is_trained = True

        # Ensemble evaluation
        cat_preds_val = self.catboost_model.predict_proba(X_val)[:, 1]
        with torch.no_grad():
            nn_preds_val = self.nn_model(X_val_tensor_full).cpu().numpy().flatten()

        ensemble_preds = (cat_preds_val + nn_preds_val) / 2.0
        ensemble_auc = roc_auc_score(y_val, ensemble_preds)

        self.shap_explainer = shap.TreeExplainer(self.catboost_model)

        return {
            "catboost_val_auc": roc_auc_score(y_val, cat_preds_val),
            "nn_val_auc": best_auc,
            "ensemble_val_auc": ensemble_auc
        }

    def predict(self, X: pd.DataFrame,
                graph_features: np.ndarray = None,
                text_features: np.ndarray = None) -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            risk_score_feat = self.risk_score_ann(X_tensor).cpu().numpy()

        if graph_features is None:
            graph_features = np.zeros((len(X), 16))
        if text_features is None:
            text_features = np.zeros((len(X), 768))

        combined_features = np.hstack([X_scaled, graph_features, text_features, risk_score_feat])
        combined_tensor = torch.tensor(combined_features, dtype=torch.float32).to(self.device)

        cat_preds = self.catboost_model.predict_proba(X)[:, 1]

        self.nn_model.eval()
        with torch.no_grad():
            nn_preds = self.nn_model(combined_tensor).cpu().numpy().flatten()

        combined_pred = (cat_preds + nn_preds) / 2.0
        risk_score = float(combined_pred[0] * 100)
        prediction = 1 if combined_pred > 0.5 else 0

        return {
            "prediction": int(prediction),
            "probability_low_risk": float(1 - combined_pred),
            "probability_high_risk": float(combined_pred),
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score)
        }

    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        if self.shap_explainer is None:
            return {"error": "SHAP explainer not available"}
        for f in self.feature_names:
            if f not in X.columns:
                X[f] = 0.0
        X = X[self.feature_names]
        shap_values = self.shap_explainer.shap_values(X)
        feature_contributions = shap_values[0]
        contributions = {f: float(val) for f, val in zip(self.feature_names, feature_contributions)}
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x), reverse=True)
        positive_factors = [(n, v) for n, v in sorted_contrib if v > 0][:5]
        negative_factors = [(n, v) for n, v in sorted_contrib if v < 0][:5]
        return {
            "feature_contributions": contributions,
            "top_risk_factors": positive_factors,
            "top_protective_factors": negative_factors
        }

    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray):
        metric_frame = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        return metric_frame.by_group

    def _get_risk_level(self, risk_score: float) -> str:
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 70:
            return "Medium Risk"
        else:
            return "High Risk"


if _name_ == "_main_":
    model = AdvancedCreditScoringModel()
    metrics = model.train()

    print("Training Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    X_test, _ = model.generate_training_data(1)

    dummy_graph_feats = np.zeros((1, 16))
    dummy_text_feats = np.zeros((1, 768))

    prediction = model.predict(X_test, graph_features=dummy_graph_feats, text_features=dummy_text_feats)
    print(f"\nTest Prediction: {prediction}")

    explanation = model.explain_prediction(X_test)
    print(f"\nExplanation available: {'feature_contributions' in explanation}")

    # Save models
    joblib.dump({
        'catboost': model.catboost_model,
        'nn_state_dict': model.nn_model.state_dict(),
        'risk_score_ann_state_dict': model.risk_score_ann.state_dict(),
        'scaler': model.scaler,
        'feature_names': model.feature_names
    }, "models/advanced_credit_model.joblib")
    print("\nModel saved successfully!")