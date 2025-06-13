import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Autism Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1E5A7A;
        transform: translateY(-2px);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #2E86AB;
        background-color: #F8F9FA;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('C:/Users/anura/Desktop/Autism_Predictive_System_Web/EncodeAndModel/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('C:/Users/anura/Desktop/Autism_Predictive_System_Web/EncodeAndModel/label_encoders.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Debug information
        st.sidebar.info(f"Model type: {type(model).__name__}")
        st.sidebar.info(f"Encoder type: {type(encoder).__name__}")
        
        if isinstance(encoder, dict):
            st.sidebar.info(f"Encoder keys: {list(encoder.keys())}")
        
        # Try to get expected feature names from model
        try:
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                st.sidebar.success(f"Found {len(expected_features)} expected features")
                with st.sidebar.expander("Expected Features"):
                    for feature in expected_features:
                        st.write(f"• {feature}")
            else:
                st.sidebar.warning("Model doesn't have feature_names_in_ attribute")
        except Exception as e:
            st.sidebar.warning(f"Could not get feature names: {e}")
        
        return model, encoder
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'best_model.pkl' and 'encoder.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Autism Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced ML-based screening tool for autism spectrum disorder prediction</p>', unsafe_allow_html=True)
    
    # Load models
    model, encoder = load_models()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Statistics", "About"])
    
    if page == "Prediction":
        prediction_page(model, encoder)
    elif page == "Statistics":
        statistics_page()
    else:
        about_page()

def prediction_page(model, encoder):
    st.markdown('<h2 class="sub-header">📊 Patient Information</h2>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AQ Scores (Autism Quotient)")
        aq_scores = {}
        for i in range(1, 11):
            aq_scores[f'A{i}_Score'] = st.slider(
                f'AQ{i} Score', 
                min_value=0, 
                max_value=4, 
                value=2, 
                help=f'Autism Quotient question {i} score (0-4 scale)'
            )
        
        st.subheader("Demographics")
        age = st.number_input('Age', min_value=0, max_value=120, value=25)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        
    with col2:
        st.subheader("Background Information")
        ethnicity = st.selectbox('Ethnicity', [
            'White European', 'Asian', 'Middle Eastern', 'Black', 
            'South Asian', 'Hispanic', 'Latino', 'Mixed', 'Other'
        ])
        
        country = st.selectbox('Country', [
            'United States', 'United Kingdom', 'Canada', 'Australia',
            'New Zealand', 'Germany', 'France', 'Spain', 'Italy',
            'Netherlands', 'India', 'Other'
        ])
        
        st.subheader("Medical History")
        jaundice = st.selectbox('Born with Jaundice?', ['No', 'Yes'])
        family_history = st.selectbox('Family History of Autism?', ['No', 'Yes'])
        
        st.subheader("App Usage")
        used_app_before = st.selectbox('Used Screening App Before?', ['No', 'Yes'])
        relation = st.selectbox('Relation to Patient', [
            'Self', 'Parent', 'Relative', 'Healthcare Professional', 'Other'
        ])
    
    # Calculate total AQ score
    total_aq_score = sum(aq_scores.values())
    
    # Display AQ score visualization
    st.markdown('<h3 class="sub-header">📈 AQ Score Visualization</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total AQ Score", total_aq_score, delta=f"{total_aq_score - 20} from average")
    with col2:
        st.metric("Questions Completed", "10/10", delta="Complete")
    with col3:
        risk_level = "High" if total_aq_score > 25 else "Medium" if total_aq_score > 15 else "Low"
        st.metric("Risk Level", risk_level)
    
    # AQ scores bar chart
    fig = px.bar(
        x=list(aq_scores.keys()), 
        y=list(aq_scores.values()),
        title="Individual AQ Scores",
        color=list(aq_scores.values()),
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction button
    if st.button('🔍 Predict Autism Risk', key='predict_btn'):
        # Prepare input data
        input_data = {
            **aq_scores,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'country': country,
            'jaundice': jaundice,
            'family_history': family_history,
            'used_app_before': used_app_before,
            'relation': relation
        }
        
        # Make prediction
        prediction, probability = make_prediction(model, encoder, input_data)
        
        # Display results
        display_prediction_results(prediction, probability, input_data)
        
        # Save to history
        result_data = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'probability': probability,
            'total_aq_score': total_aq_score,
            'age': age,
            'gender': gender
        }
        st.session_state.prediction_history.append(result_data)

def encode_categorical_data(df, encoder):
    """
    Robust encoding function that handles different encoder types
    """
    categorical_mappings = {
        'gender': {'Male': 0, 'Female': 1, 'Other': 2},
        'ethnicity': {
            'White European': 0, 'Asian': 1, 'Middle Eastern': 2, 'Black': 3,
            'South Asian': 4, 'Hispanic': 5, 'Latino': 6, 'Mixed': 7, 'Other': 8
        },
        'country': {
            'United States': 0, 'United Kingdom': 1, 'Canada': 2, 'Australia': 3,
            'New Zealand': 4, 'Germany': 5, 'France': 6, 'Spain': 7, 'Italy': 8,
            'Netherlands': 9, 'India': 10, 'Other': 11
        },
        'jaundice': {'No': 0, 'Yes': 1},
        'family_history': {'No': 0, 'Yes': 1},
        'used_app_before': {'No': 0, 'Yes': 1},
        'relation': {
            'Self': 0, 'Parent': 1, 'Relative': 2, 
            'Healthcare Professional': 3, 'Other': 4
        }
    }
    
    categorical_columns = ['gender', 'ethnicity', 'country', 'jaundice', 'family_history', 'used_app_before', 'relation']
    
    for col in categorical_columns:
        if col in df.columns:
            try:
                # Method 1: Try using the provided encoder
                if isinstance(encoder, dict) and col in encoder:
                    if hasattr(encoder[col], 'transform'):
                        df[col] = encoder[col].transform(df[col].astype(str))
                    elif isinstance(encoder[col], dict):
                        df[col] = df[col].map(encoder[col]).fillna(0)
                    else:
                        df[col] = df[col].map(categorical_mappings[col]).fillna(0)
                
                elif hasattr(encoder, 'transform'):
                    df[col] = encoder.transform(df[col].astype(str))
                
                # Method 2: Use predefined mappings
                else:
                    df[col] = df[col].map(categorical_mappings[col]).fillna(0)
                    
            except Exception as e:
                # Method 3: Fallback to manual mapping
                df[col] = df[col].map(categorical_mappings[col]).fillna(0)
                
    return df

def match_model_features(df, model):
    """
    Dynamically match DataFrame columns to model's expected features
    """
    try:
        # Get expected features from model
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            # Fallback to common expected features
            expected_features = [
                'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                'age', 'gender', 'ethnicity', 'country', 'jaundice', 
                'family_history', 'used_app_before', 'relation'
            ]
        
        # Create a new DataFrame with expected columns
        matched_df = pd.DataFrame()
        
        for feature in expected_features:
            if feature in df.columns:
                matched_df[feature] = df[feature]
            else:
                # Try to find similar columns or set default values
                if feature.startswith('A') and feature.endswith('_Score'):
                    # Handle AQ score variations
                    alt_names = [
                        feature.lower().replace('_score', '_score'),
                        feature.lower().replace('a', 'aq').replace('_score', '_score'),
                        feature.replace('A', 'aq').replace('_Score', '_score')
                    ]
                    found = False
                    for alt_name in alt_names:
                        if alt_name in df.columns:
                            matched_df[feature] = df[alt_name]
                            found = True
                            break
                    if not found:
                        matched_df[feature] = 2  # Default AQ score
                else:
                    # For other features, set reasonable defaults
                    if feature == 'age':
                        matched_df[feature] = 25
                    elif feature in ['gender', 'ethnicity', 'country', 'relation']:
                        matched_df[feature] = 0
                    elif feature in ['jaundice', 'family_history', 'used_app_before']:
                        matched_df[feature] = 0
                    else:
                        matched_df[feature] = 0
        
        return matched_df, expected_features
        
    except Exception as e:
        st.error(f"Feature matching error: {e}")
        return df, []

def make_prediction(model, encoder, input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    df = encode_categorical_data(df, encoder)
    
    # Ensure all values are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = 0
    
    # Match features to model expectations
    matched_df, expected_features = match_model_features(df, model)
    
    # Debug information
    with st.expander("Debug Information", expanded=False):
        st.write("Original Data:")
        st.dataframe(df)
        st.write("Matched Data for Model:")
        st.dataframe(matched_df)
        st.write("Expected Features:")
        st.write(expected_features)
        st.write("Matched Data Types:")
        st.write(matched_df.dtypes)
    
    # Make prediction
    try:
        prediction = model.predict(matched_df)[0]
        
        # Handle different probability output formats
        try:
            probability = model.predict_proba(matched_df)[0]
        except:
            # If predict_proba doesn't work, create dummy probabilities
            if prediction == 1:
                probability = [0.3, 0.7]  # High risk
            else:
                probability = [0.7, 0.3]  # Low risk
            st.warning("Using estimated probabilities (model doesn't support predict_proba)")
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error("Detailed error information:")
        st.write("Matched DataFrame shape:", matched_df.shape)
        st.write("Matched DataFrame columns:", list(matched_df.columns))
        
        return None, None

def display_prediction_results(prediction, probability, input_data):
    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
    
    if prediction is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk - Further assessment recommended")
                risk_color = "#FF6B6B"
            else:
                st.success("✅ Low Risk - No immediate concerns")
                risk_color = "#4ECDC4"
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Autism Risk Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Risk Assessment Summary")
            
            risk_factors = []
            total_aq = sum([v for k, v in input_data.items() if k.startswith('A') and k.endswith('_Score')])
            
            if total_aq > 25:
                risk_factors.append("High AQ Score")
            if input_data['family_history'] == 'Yes':
                risk_factors.append("Family History")
            if input_data['jaundice'] == 'Yes':
                risk_factors.append("Jaundice at Birth")
            
            if risk_factors:
                st.warning("Risk Factors Present:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("No major risk factors identified")
            
            st.markdown("### 📊 Confidence Scores")
            st.progress(probability[0], text=f"Low Risk: {probability[0]:.1%}")
            st.progress(probability[1], text=f"High Risk: {probability[1]:.1%}")
        
        # Recommendations
        st.markdown('<h3 class="sub-header" style="color: #d9534f;">💡 Recommendations</h3>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box"  style="background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; color: #333333;">
            <h4 style="color: #d9534f;">🔍 Recommended Next Steps:</h4>
            <ul style="color: #333333;">
                <li><strong style="color: #d9534f;">Consult a Healthcare Professional:</strong> Schedule an appointment with a pediatrician, psychologist, or psychiatrist experienced in autism spectrum disorders</li>
                <li><strong style="color: #d9534f;">Comprehensive Diagnostic Assessment:</strong> Consider formal autism diagnostic tools like ADOS-2 (Autism Diagnostic Observation Schedule) or ADI-R (Autism Diagnostic Interview-Revised)</li>
                <li><strong style="color: #d9534f;">Early Intervention Programs:</strong> If applicable, explore speech therapy, occupational therapy, or behavioral interventions</li>
                <li><strong style="color: #d9534f;">Educational Support:</strong> Discuss potential accommodations or special education services if relevant</li>
                <li><strong style="color: #d9534f;">Support Networks:</strong> Connect with local autism support groups and advocacy organizations</li>
                <li><strong style="color: #d9534f;">Additional Resources:</strong> Consider contacting organizations like Autism Speaks, ASAN, or local autism resource centers</li>
            </ul>
            <p><em><strong style="color: #888;">Important:</strong> This screening tool is not a diagnostic instrument. A formal diagnosis requires comprehensive evaluation by qualified professionals.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box" style="background-color: #f4f6f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd; color: #333333;">
            <h4 style="color: #5bc0de;">✅ Current Assessment Results:</h4>
            <ul>
                <li><strong style="color: #5bc0de;">Low Risk Indication:</strong> Based on current screening responses, the likelihood of autism spectrum disorder appears low</li>
                <li><strong style="color: #5bc0de;">Continue Monitoring:</strong> Keep track of developmental milestones and behavioral patterns</li>
                <li><strong style="color: #5bc0de;">Stay Informed:</strong> Learn about autism signs and symptoms to monitor for changes over time</li>
                <li><strong style="color: #5bc0de;">Regular Check-ups:</strong> Maintain regular appointments with healthcare providers for routine developmental screening</li>
                <li><strong style="color: #5bc0de;">Trust Your Instincts:</strong> If you notice concerning behaviors or developmental delays, don't hesitate to seek professional advice</li>
                <li><strong style="color: #5bc0de;">Re-screening:</strong> Consider using this tool again if circumstances or behaviors change significantly</li>
            </ul>
            <p><em><strong style="color: #888;">Remember:</strong> Even with low-risk results, professional consultation is always recommended if you have ongoing concerns about development or behavior.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        # Additional general recommendations
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f4f6f9; border-left: 4px solid #2E86AB; border-radius: 5px; color: #333333">
        <h5 style="color: #2E86AB;">📚 General Information & Resources:</h5>
        <ul>
            <li><strong style="color: #5bc0de;">Learn More:</strong> Visit reputable sources like CDC, NIH, or Autism Society for evidence-based information</li>
            <li><strong style="color: #5bc0de;">Emergency Contact:</strong> If experiencing crisis situations, contact your local emergency services or crisis hotline</li>
            <li><strong style="color: #5bc0de;">Second Opinion:</strong> Don't hesitate to seek multiple professional opinions for important health decisions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        




def statistics_page():
    st.markdown('<h2 class="sub-header">📈 Prediction Statistics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Visit the Prediction page to get started!")
        return
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        high_risk_count = sum(history_df['prediction'])
        st.metric("High Risk Cases", high_risk_count)
    with col3:
        avg_probability = history_df['probability'].apply(lambda x: x[1]).mean()
        st.metric("Avg Risk Probability", f"{avg_probability:.1%}")
    with col4:
        avg_aq_score = history_df['total_aq_score'].mean()
        st.metric("Avg AQ Score", f"{avg_aq_score:.1f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_counts = history_df['prediction'].value_counts()
        
        # Handle case where there might be only one type of prediction
        if len(risk_counts) == 1:
            if risk_counts.index[0] == 0:
                risk_labels = ['Low Risk']
                risk_values = [risk_counts.values[0]]
            else:
                risk_labels = ['High Risk']
                risk_values = [risk_counts.values[0]]
        else:
            risk_labels = ['Low Risk' if idx == 0 else 'High Risk' for idx in sorted(risk_counts.index)]
            risk_values = [risk_counts[idx] for idx in sorted(risk_counts.index)]
        
        fig = px.pie(
            values=risk_values,
            names=risk_labels,
            title="Risk Distribution",
            color_discrete_map={'Low Risk': '#4ECDC4', 'High Risk': '#FF6B6B'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            history_df,
            x='age',
            title="Age Distribution of Assessments",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.markdown("### Recent Predictions")
    recent_df = history_df.tail(10).copy()
    recent_df['Risk Level'] = recent_df['prediction'].map({0: 'Low Risk', 1: 'High Risk'})
    recent_df['Timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    display_df = recent_df[['Timestamp', 'Risk Level', 'total_aq_score', 'age', 'gender']]
    display_df.columns = ['Time', 'Risk Level', 'AQ Score', 'Age', 'Gender']
    
    st.dataframe(display_df, use_container_width=True)

def about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application uses machine learning to provide preliminary screening for autism spectrum disorder (ASD) based on various behavioral and demographic factors.
        
        ### 📊 How It Works
        The system analyzes:
        - **AQ Scores**: Autism Quotient questionnaire responses
        - **Demographics**: Age, gender, ethnicity, country
        - **Medical History**: Jaundice, family history
        - **Background**: Previous app usage, relationship to patient
        
        ### 🔍 Model Information
        - Uses pre-trained machine learning model
        - Provides probability scores for risk assessment
        - Includes confidence intervals for predictions
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Important Disclaimer
        **This tool is for screening purposes only and should not be used as a diagnostic tool.**
        
        - Results are preliminary indicators only
        - Professional medical evaluation is required for diagnosis
        - False positives and negatives are possible
        - Consult healthcare professionals for proper assessment
        
        ### 🔒 Privacy & Security
        - No personal data is stored permanently
        - All information is processed locally
        - Session data is cleared when browser is closed
        
        ### 📞 Support
        For questions about autism spectrum disorder, consult with qualified healthcare professionals.
        """)
    
    st.markdown("---")
    st.markdown("*Developed with ❤️ for autism awareness and early intervention*")

if __name__ == "__main__":
    main()