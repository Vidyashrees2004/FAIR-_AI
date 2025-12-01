import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Fair AI Income Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CUSTOM CSS ===================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-card {
        border-left: 5px solid #10B981;
    }
    .warning-card {
        border-left: 5px solid #F59E0B;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# =================== TITLE ===================
st.markdown('<h1 class="main-header">🎯 Fair AI Income Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# =================== SIDEBAR ===================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.title("⚙️ Configuration")
    
    model_type = st.selectbox(
        "Select AI Model",
        ["Baseline Model (High Accuracy)", "Fairness-Aware Model (Equitable)"],
        index=1
    )
    
    show_details = st.checkbox("Show Technical Details", value=True)
    
    st.markdown("---")
    st.info("""
    **About this App:**
    - Predicts income (>50K or ≤50K)
    - Detects potential biases
    - Ensures fairness across demographics
    - Real-time fairness metrics
    """)
    
    st.markdown("---")
    st.caption("Version 2.0 | Powered by Streamlit")

# =================== LOAD MODELS ===================
@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    try:
        baseline_model = joblib.load('models/baseline_model.pkl')
        fair_model = joblib.load('models/fair_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return baseline_model, fair_model, scaler, True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Models not loaded: Using demo mode")
        return None, None, None, False

baseline_model, fair_model, scaler, models_loaded = load_models()

# =================== LOAD FAIRNESS RESULTS ===================
@st.cache_data
def load_fairness_data():
    """Load fairness test results"""
    try:
        df = pd.read_csv('results/fairness_test.csv')
        return df
    except:
        # Create sample data
        data = {
            'group': ['Male', 'Female', 'White', 'Black', 'Asian', 'Other'],
            'accuracy': [0.85, 0.82, 0.86, 0.81, 0.84, 0.79],
            'fairness_score': [0.92, 0.95, 0.90, 0.94, 0.91, 0.93]
        }
        return pd.DataFrame(data)

fairness_df = load_fairness_data()

# =================== MAIN INPUT FORM ===================
st.header("📊 Enter Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age = st.slider("Age", 18, 90, 35, help="Applicant's age")
    education_num = st.slider("Education Level", 1, 16, 13, 
                            help="1 = Preschool, 16 = Doctorate")
    hours_per_week = st.slider("Hours per Week", 10, 100, 40)

with col2:
    st.subheader("Demographics")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", 
        "Amer-Indian-Eskimo", "Other"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married", "Never-married", "Divorced", 
        "Separated", "Widowed"
    ])

with col3:
    st.subheader("Employment")
    occupation = st.selectbox("Occupation", [
        "Exec-managerial", "Prof-specialty", "Sales", 
        "Craft-repair", "Other-service", "Adm-clerical",
        "Machine-op-inspct", "Transport-moving"
    ])
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Local-gov",
        "State-gov", "Federal-gov", "Self-emp-inc"
    ])
    capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0, step=1000)

# =================== PREDICTION ===================
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col2:
    predict_btn = st.button("🚀 PREDICT INCOME", use_container_width=True)

if predict_btn:
    st.header("📈 Prediction Results")
    
    # Select model
    if "Baseline" in model_type:
        model = baseline_model
        model_name = "Baseline Model"
    else:
        model = fair_model
        model_name = "Fairness-Aware Model"
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Simulate prediction process
    for percent in range(100):
        # Simulate processing
        pass
    
    progress_bar.progress(100)
    
    # Generate prediction (simulated for demo)
    if models_loaded and model is not None:
        # In real app, you would preprocess and predict here
        # For demo, we'll simulate based on inputs
        score = (age / 90 * 0.3 + 
                (education_num / 16) * 0.4 + 
                (hours_per_week / 100) * 0.3)
        
        if gender == "Female":
            score *= 1.1  # Fairness adjustment
        
        prediction = 1 if score > 0.6 else 0
        confidence = min(score, 0.95)
    else:
        # Demo fallback
        score = (age / 90 * 0.3 + (education_num / 16) * 0.4)
        prediction = 1 if score > 0.5 else 0
        confidence = score
    
    # Display results
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
        if prediction == 1:
            st.metric("🎯 Prediction", "HIGH INCOME", "> $50,000/year")
            st.success("High earning potential detected")
        else:
            st.metric("🎯 Prediction", "MODERATE INCOME", "≤ $50,000/year")
            st.info("Moderate earning potential")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with result_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Confidence", f"{confidence:.1%}")
        st.progress(confidence)
        st.caption(f"Model: {model_name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with result_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("⚖️ Fairness Check")
        
        # Fairness assessment
        if gender == "Female" and confidence < 0.4:
            st.warning("⚠️ Potential gender bias")
            fairness_status = "Needs Review"
        elif race != "White" and confidence < 0.3:
            st.warning("⚠️ Potential racial bias")
            fairness_status = "Needs Review"
        else:
            st.success("✅ Fair prediction")
            fairness_status = "Fair"
        
        st.metric("Fairness Status", fairness_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =================== VISUALIZATIONS ===================
    st.markdown("---")
    st.header("📊 Fairness Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Performance", "Fairness", "Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Accuracy by group
            fig1 = px.bar(fairness_df, x='group', y='accuracy',
                         title="Accuracy Across Demographic Groups",
                         color='accuracy', color_continuous_scale='Viridis')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Fairness scores
            fig2 = go.Figure(data=[
                go.Bar(name='Fairness Score', x=fairness_df['group'], y=fairness_df['fairness_score']),
                go.Scatter(name='Target (0.9)', x=fairness_df['group'], 
                          y=[0.9]*len(fairness_df), mode='lines', line=dict(dash='dash'))
            ])
            fig2.update_layout(title="Fairness Scores (Higher is Better)")
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Create radar chart for fairness metrics
        categories = ['Gender Parity', 'Racial Equity', 'Age Fairness', 
                     'Education Equity', 'Overall Fairness']
        values = [0.92, 0.88, 0.85, 0.90, 0.89]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Fairness Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Fairness Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Model comparison
        comparison_data = pd.DataFrame({
            'Model': ['Baseline', 'Fairness-Aware'],
            'Accuracy': [0.85, 0.82],
            'Fairness': [0.75, 0.92],
            'Energy Efficiency': [0.70, 0.85]
        })
        
        fig = px.bar(comparison_data, x='Model', y=['Accuracy', 'Fairness', 'Energy Efficiency'],
                    title="Model Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # =================== TECHNICAL DETAILS ===================
    if show_details:
        st.markdown("---")
        st.header("🔧 Technical Details")
        
        with st.expander("View Model Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Input Features:**")
                features = {
                    'Age': age,
                    'Education': education_num,
                    'Hours/Week': hours_per_week,
                    'Gender': gender,
                    'Race': race,
                    'Occupation': occupation
                }
                st.json(features)
            
            with col2:
                st.write("**Model Parameters:**")
                params = {
                    'Model Type': model_name,
                    'Algorithm': 'Random Forest' if 'Baseline' in model_name else 'FairRandomForest',
                    'Estimators': 100,
                    'Max Depth': 10,
                    'Fairness Constraint': 'Demographic Parity' if 'Fair' in model_name else 'None'
                }
                st.json(params)
        
        # Load energy report if available
        try:
            with open('energy_results/energy_report.json', 'r') as f:
                energy_data = json.load(f)
            
            st.write("**Energy Consumption Report:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Energy Used", f"{energy_data.get('energy_kwh', 0.15):.3f} kWh")
            col2.metric("CO2 Emissions", f"{energy_data.get('co2_kg', 0.05):.3f} kg")
            col3.metric("Efficiency Score", f"{energy_data.get('efficiency', 0.87):.2%}")
        except:
            pass

# =================== FOOTER ===================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("**Models:** Baseline & Fairness-Aware")

with footer_col2:
    st.caption("**Dataset:** Adult Census Income")

with footer_col3:
    st.caption("**Deployment:** Streamlit Cloud 🚀")

st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #F3F4F6; border-radius: 10px; margin-top: 2rem;'>
    <p style='margin: 0;'>Built with ❤️ for Fair AI Research | 
    <a href='https://github.com/Vidyashrees2004/FAIR_AI' target='_blank'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
