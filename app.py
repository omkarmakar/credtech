import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys
import umap

# Add src directory to path for imports if needed
sys.path.append('.')

alpha_vantage_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
news_api_key = st.secrets["NEWS_API_KEY"]
finnhub_api_key = st.secrets["FINNHUB_API_KEY"]
fmp_key = st.secrets["FMP_KEY"]
twelvedata_key = st.secrets["TWELVEDATA_API_KEY"]

from data_sources import AlphaVantageClient as AlphaVantageClient
from data_sources import NewsClient as NewsClient
from data_sources import FeatureEngineer as FeatureEngineer

from credit_model import AdvancedCreditScoringModel as CreditScoringModel

# Page configuration and styling
st.set_page_config(
    page_title="CredTech - Real-Time Credit Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.risk-high {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    border-left: 5px solid #ff0000;
    padding: 1rem;
    border-radius: 8px;
    color: white;
    margin: 1rem 0;
}

.risk-medium {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-left: 5px solid #ff9800;
    padding: 1rem;
    border-radius: 8px;
    color: white;
    margin: 1rem 0;
}

.risk-low {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-left: 5px solid #4caf50;
    padding: 1rem;
    border-radius: 8px;
    color: white;
    margin: 1rem 0;
}

.news-item {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
</style>
""", unsafe_allow_html=True)


def create_risk_gauge(risk_score, company_name):
    """Create a risk score gauge"""
    # Determine color based on risk level
    if risk_score < 30:
        color = "green"
        risk_level = "Low Risk"
    elif risk_score < 70:
        color = "orange" 
        risk_level = "Medium Risk"
    else:
        color = "red"
        risk_level = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{company_name} Credit Risk Score", 'font': {'size': 20}},
        delta = {'reference': 50, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e8'},
                {'range': [30, 70], 'color': '#fff8e1'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig, risk_level
def create_feature_vector_weight_chart(feature_vector):
    abs_values = feature_vector.abs()
    normalized = abs_values / abs_values.sum()
    labels = normalized.index.str.replace('_',' ').str.title()
    values = normalized.values

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent'
    ))
    fig.update_layout(
        title="Feature Vector Weight Distribution",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_shap_waterfall(explanation_data):
    """Create SHAP waterfall-style chart"""
    if 'feature_contributions' not in explanation_data:
        return None
    
    contributions = explanation_data['feature_contributions']
    base_value = explanation_data.get('base_value', 0)
    
    # Sort contributions by absolute value
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 10 features
    # top_features = sorted_contributions[:10]
    # Always include financial ratios of interest if present
    important_keys = ['roe', 'pe_ratio', 'peg_ratio', 'debt_to_capital_employed']
    top_features = [item for item in sorted_contributions if item[0] in important_keys]
    top_features += [item for item in sorted_contributions if item[0] not in important_keys][:10]

    
    features = [feature.replace('_', ' ').title() for feature, value in top_features]
    values = [value for feature, value in top_features]
    colors = ['red' if v > 0 else 'green' for v in values]
    
    fig = go.Figure(go.Bar(
        x=[abs(v) for v in values],
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='auto',
        textfont={'color': 'white', 'size': 12}
    ))
    
    fig.update_layout(
        title="Feature Impact on Credit Risk (SHAP Values)",
        xaxis_title="Impact on Risk Score",
        yaxis_title="Features",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial", 'size': 12}
    )
    
    return fig

def create_feature_importance_pie(explanation_data):
    """Create pie chart of feature importance"""
    if 'feature_contributions' not in explanation_data:
        return None
    
    contributions = explanation_data['feature_contributions']
    
    # Get absolute values and sort
    abs_contributions = {k: abs(v) for k, v in contributions.items()}
    sorted_contributions = sorted(abs_contributions.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 8 features
    top_features = sorted_contributions[:8]
    other_sum = sum([v for _, v in sorted_contributions[8:]])
    
    if other_sum > 0:
        top_features.append(('Others', other_sum))
    
    labels = [feature.replace('_', ' ').title() for feature, value in top_features]
    values = [value for feature, value in top_features]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Feature Importance Distribution",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True
    )
    
    return fig

def create_news_sentiment_chart(news_data):
    """Create news sentiment timeline"""
    if not news_data:
        return None
    
    # Extract sentiment data
    dates = []
    sentiments = []
    titles = []
    
    for item in news_data:
        try:
            date = pd.to_datetime(item.get('published_at', item.get('date', '2024-08-21')))
            sentiment = item['sentiment']['compound']
            title = item.get('title', item.get('headline', 'News'))[:50] + "..."
            
            dates.append(date)
            sentiments.append(sentiment)
            titles.append(title)
        except:
            continue
    
    if not dates:
        return None
    
    # Create timeline chart
    fig = go.Figure()
    
    colors = ['red' if s < -0.1 else 'green' if s > 0.1 else 'gray' for s in sentiments]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiments,
        mode='markers+lines',
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=1, color='black')
        ),
        text=titles,
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>',
        name='News Sentiment'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title="Recent News Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_fairness_bar_chart(fairness_df):
    fig = go.Figure(go.Bar(
        x=fairness_df.index.astype(str),
        y=fairness_df.values,
        marker_color='indigo'
    ))
    fig.update_layout(
        title="Selection Rate by Sensitive Groups",
        yaxis_title="Selection Rate",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def create_performance_comparison(metrics):
    # Expecting dictionary with keys like "catboost_val_auc", "nn_val_auc", "ensemble_val_auc"
    labels = ["CatBoost", "Neural Network", "Ensemble"]
    values = [metrics.get('catboost_val_auc', 0), metrics.get('nn_val_auc', 0), metrics.get('ensemble_val_auc', 0)]
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=['#008fd5', '#fc4f30', '#6d904f']
    ))
    fig.update_layout(
        title="Validation AUC of Models",
        yaxis=dict(range=[0, 1]),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def create_prediction_score_histogram(scores):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=30,
        marker_color='#4facfe'
    ))
    fig.update_layout(
        title="Prediction Score Distribution",
        xaxis_title="Predicted Risk Score",
        yaxis_title="Count",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def create_text_embedding_umap(embeddings, risk_levels, docs):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    # risk_levels assumed numeric or mapped to numeric
    fig = go.Figure(go.Scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=risk_levels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Risk Level")
        ),
        text=docs
    ))
    fig.update_layout(
        title="2D Projection of Text Embeddings (UMAP)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# Cache and load model function
@st.cache_resource
def load_model():
    model = CreditScoringModel()
    try:
        model.load_model('models/credit_model.joblib')
        st.success("✅ Pre-trained model loaded successfully!")
    except Exception:
        with st.spinner("🔄 Training new model... This may take a moment."):
            metrics = model.train()
            model.save_model('models/credit_model.joblib')
            st.success(f"✅ Model trained! ROC AUC: {metrics['ensemble_val_auc']:.3f}")
    return model

@st.cache_resource
def load_data_clients():
    return AlphaVantageClient(), NewsClient(), FeatureEngineer()


def main():
    st.markdown('<h1 class="main-header">🏦 CredTech Real-Time Credit Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;">
        🚀 <strong>Real-time credit scoring with explainable AI</strong> | 📊 Advanced analytics | 🎯 Instant insights
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    av_client, news_client, feature_engineer = load_data_clients()

    st.sidebar.header("🎛️ Credit Assessment Controls")
    company_symbol = st.sidebar.text_input("📈 Company Symbol", value="AAPL", help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL, TSLA)").upper()
    st.sidebar.subheader("🔧 Analysis Options")
    include_news = st.sidebar.checkbox("📰 Include News Sentiment", value=True)
    include_financials = st.sidebar.checkbox("💰 Include Financial Data", value=True)
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
        news_days = st.slider("News Lookback Days", 1, 30, 7)
    run_analysis = st.sidebar.button("🚀 Run Credit Analysis", type="primary")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        
        st.rerun()


    if run_analysis or 'analysis_data' not in st.session_state:
        with st.spinner(f"🔍 Analyzing {company_symbol}... Fetching data and computing risk score..."):
            company_data, news_data = {}, []
            if include_financials:
                company_data = av_client.get_company_overview(company_symbol)
            if include_news:
                company_name = company_data.get('Name', company_symbol)
                news_data = news_client.get_company_news(company_name)
            # financial_features = feature_engineer.extract_financial_features(company_data)
            # sentiment_features = feature_engineer.extract_sentiment_features(news_data)
            # feature_vector = feature_engineer.create_feature_vector(financial_features, sentiment_features)
            # print(model.feature_names)
            financial_features = feature_engineer.extract_financial_features(company_data)
            sentiment_features = feature_engineer.extract_sentiment_features(news_data)
            # feature_vector = feature_engineer.create_feature_vector(financial_features, sentiment_features, model.feature_names)
            feature_vector = feature_engineer.create_feature_vector(financial_features, sentiment_features, model.feature_names if model.feature_names else None)
            prediction = model.predict(feature_vector)
            explanation = model.explain_prediction(feature_vector)

            # Optional: generate mock fairness metrics and prediction scores for demo
            # In real use, replace with actual data arrays
            fairness_metrics = None
            prediction_scores = None
            text_embeddings = None
            risk_levels_numeric = None
            documents = None

            # Example: You may prepare these from your pipeline if available

            st.session_state['analysis_data'] = {
                'symbol': company_symbol,
                'company_data': company_data,
                'news_data': news_data,
                'prediction': prediction,
                'explanation': explanation,
                'feature_vector': feature_vector,
                'timestamp': datetime.now(),
                'metrics': model.train() if not model.is_trained else {},  # Or cache previous metrics properly
                'fairness_metrics': fairness_metrics,
                'prediction_scores': prediction_scores,
                'text_embeddings': text_embeddings,
                'risk_levels_numeric': risk_levels_numeric,
                'documents': documents,
            }

    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="metric-card"><h3>Risk Score</h3><h2>{data["prediction"]["risk_score"]:.1f}/100</h2></div>', unsafe_allow_html=True)
        with col2:
            conf = max(data["prediction"].get("probability_low_risk", 0), data["prediction"].get("probability_high_risk", 0))
            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><h2>{conf:.1%}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>News Articles</h3><h2>{len(data["news_data"])}</h2></div>', unsafe_allow_html=True)
        with col4:
            avg_sentiment = np.mean([item['sentiment']['compound'] for item in data['news_data']]) if data['news_data'] else 0
            emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
            st.markdown(f'<div class="metric-card"><h3>Avg Sentiment</h3><h2>{emoji} {avg_sentiment:.2f}</h2></div>', unsafe_allow_html=True)

        st.header("📊 Risk Assessment")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, level = create_risk_gauge(data['prediction']['risk_score'], data['company_data'].get('Name', data['symbol']))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            risk_class = level.lower().split(' ')[0]
            st.markdown(f'<div class="risk-{risk_class}"><h2>{level}</h2><p><strong>Risk Score:</strong> {data["prediction"]["risk_score"]:.1f}/100</p><p><strong>Model Confidence:</strong> {conf:.1%}</p><p><strong>Last Updated:</strong> {data["timestamp"].strftime("%H:%M:%S")}</p></div>', unsafe_allow_html=True)

        st.header("🔍 AI Explanation (SHAP Analysis)")
        if 'error' not in data['explanation']:
            col1, col2 = st.columns(2)
            with col1:
                shap_fig = create_shap_waterfall(data['explanation'])
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)
                if 'top_risk_factors' in data['explanation']:
                    st.subheader("⚠️ Top Risk Factors")
                    for f, v in data['explanation']['top_risk_factors']:
                        st.write(f"• **{f.replace('_', ' ').title()}**: {v:+.3f}")
            with col2:
                pie_fig = create_feature_importance_pie(data['explanation'])
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)
                if 'top_protective_factors' in data['explanation']:
                    st.subheader("✅ Protective Factors")
                    for f, v in data['explanation']['top_protective_factors']:
                        st.write(f"• **{f.replace('_', ' ').title()}**: {abs(v):.3f}")
        else:
            st.warning("⚠️ Explanations temporarily unavailable. Prediction is still valid.")

        # News Sentiment
        if data['news_data']:
            st.header("📰 News Sentiment Analysis")
            c1, c2 = st.columns([2, 1])
            with c1:
                sentiment_fig = create_news_sentiment_chart(data['news_data'])
                if sentiment_fig:
                    st.plotly_chart(sentiment_fig, use_container_width=True)
            with c2:
                st.subheader("Recent Headlines")
                for item in data['news_data'][:10]:
                    score = item['sentiment']['compound']
                    color = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "🟡"
                    st.markdown(f'<div class="news-item"><strong>{color} {item.get("title", item.get("headline", "News"))[:60]}</strong><br><small>Sentiment: {score:.2f} | {item.get("published_at", item.get("date", "Recent"))}</small></div>', unsafe_allow_html=True)

        # st.header("📊 Feature Vector Weights (Raw Input Importance)")
        # weight_fig = create_feature_vector_weight_chart(data['feature_vector'].T.squeeze())
        # st.plotly_chart(weight_fig, use_container_width=True)

        # Company fundamentals
        st.header("💼 Company Fundamentals")
        if data['company_data']:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("📈 Profitability")
                st.write(f"**Profit Margin:** {data['company_data'].get('ProfitMargin', 'N/A')}")
                st.write(f"**Return on Assets:** {data['company_data'].get('ReturnOnAssetsTTM', 'N/A')}")
                st.write(f"**Return on Equity:** {data['company_data'].get('ReturnOnEquityTTM', 'N/A')}")
            with c2:
                st.subheader("💰 Valuation")
                st.write(f"**P/E Ratio:** {data['company_data'].get('PERatio', 'N/A')}")
                st.write(f"**Book Value:** ${data['company_data'].get('BookValue', 'N/A')}")
                st.write(f"**Market Cap:** ${data['company_data'].get('MarketCapitalization', 'N/A')}")
            with c3:
                st.subheader("📊 Performance")
                st.write(f"**EPS:** ${data['company_data'].get('EPS','N/A')}")
                st.write(f"**Beta:** {data['company_data'].get('Beta','N/A')}")
                st.write(f"**Dividend Yield:** {data['company_data'].get('DividendYield','N/A')}")

        # === NEW: Fairness Metrics ===
        if data.get('fairness_metrics') is not None:
            st.header("⚖️ Fairness Metrics")
            fairness_df = data['fairness_metrics']
            st.dataframe(fairness_df)
            fig_fair = create_fairness_bar_chart(fairness_df)
            st.plotly_chart(fig_fair, use_container_width=True)

        # === NEW: Model Performance Comparison ===
        if data.get('metrics'):
            st.header("📈 Model Performance Comparison")
            perf_fig = create_performance_comparison(data['metrics'])
            st.plotly_chart(perf_fig, use_container_width=True)

        # === NEW: Prediction Score Distribution ===
        if data.get('prediction_scores') is not None:
            st.header("📊 Prediction Score Distribution")
            hist_fig = create_prediction_score_histogram(data['prediction_scores'])
            st.plotly_chart(hist_fig, use_container_width=True)

        # === NEW: Text Embeddings UMAP Projection ===
        # if data.get('text_embeddings') is not None and data.get('risk_levels_numeric') is not None:
        #     st.header("📝 Text Embeddings Projection")
        #     umap_fig = create_text_embedding_umap(data['text_embeddings'], data['risk_levels_numeric'], data.get('documents', []))
        #     st.plotly_chart(umap_fig, use_container_width=True)

        with st.expander("🔧 Technical Details & Raw Data"):
            st.subheader("Feature Vector")
            st.dataframe(data['feature_vector'].T, use_container_width=True)
            st.subheader("Model Prediction Details")
            st.json(data['prediction'])
            if 'error' not in data['explanation']:
                st.subheader("SHAP Explanation Data")
                filtered_exp = {k: v for k, v in data['explanation'].items() if k != 'feature_contributions'}
                st.json(filtered_exp)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🏆 <strong>CredTech Hackathon 2025</strong> | Built with ❤️ using Streamlit, CatBoost & SHAP<br>
        <small>⚡ Real-time credit intelligence powered by explainable AI</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
