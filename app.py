import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(
    page_title="Term Deposit Subscription Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    div.stButton {
        text-align: center;
    }
    /* Enforce single line headers if possible, or reduce size */
    h2 {
        font-size: 1.8rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    return joblib.load('model/lightgbm_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. App Header
st.markdown('<div class="main-header">🏦 Bank Marketing Prediction App</div>', unsafe_allow_html=True)
st.markdown("---")

# 5. User Inputs - Layout with 3 Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📋 Customer Profile")
    # Demographics
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", 
        ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 
         'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox("Education", 
        ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
         'professional.course', 'university.degree', 'unknown'])
    
    st.markdown("#### Financial Products")
    default = st.selectbox("Credit in Default?", ['no', 'unknown'])
    housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

with col2:
    st.header("📞 Campaign Contact Info")
    # latest contact attributes
    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    # Last Contact month
    month_codes = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    month_full_map = {
        'jan': 'January', 'feb': 'February', 'mar': 'March',
        'apr': 'April', 'may': 'May', 'jun': 'June',
        'jul': 'July', 'aug': 'August', 'sep': 'September',
        'oct': 'October', 'nov': 'November', 'dec': 'December'
    }

    month = st.selectbox("Last Contact Month", 
        month_codes, index=4, format_func=lambda x: month_full_map[x])

    # Last Contact Day
    wod_codes = ['mon', 'tue', 'wed', 'thu', 'fri']

    wod_full_map = {
        'mon': 'Monday', 'tue': 'Tuesday', 'wed': 'Wednesday',
        'thu': 'Thursday', 'fri': 'Friday'
    }

    day_of_week = st.selectbox("Last Contact Day", 
        wod_codes, index = 0, format_func = lambda x: wod_full_map[x])
    duration = st.number_input("Duration (sec)", min_value=0, value=200, help="Last contact duration in seconds")
    # previous campaign attributes
    campaign = st.number_input("Total Number of Contacts (Current Campaign)", min_value=1, value=1, help="Number of contacts during this campaign (minimum 1)")
    # pdays = st.number_input("Days Since Last Contact (Last Campaign)", min_value=0, value=999, help="999 means never contacted")
    # previous = st.number_input("Total Number of Contacts (Previous Campaigns)", min_value=0, value=0, help="Number of contacts before this campaign")
    # poutcome = st.selectbox("Deposit Outcome of the Last Campaign", ['failure', 'nonexistent', 'success'], index = 1)

with col3:
    st.header("📊 Socio-Economic Indices")
    # Socio-Economic Indicators
    cons_price_idx = st.number_input("Consumer Price Index", value=92.89)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-46.2)
    euribor3m = st.number_input("Euribor (3 Month Maturity)", value=1.25)


# Prepare Data
data = {
    'age': age, 'job': job, 'marital': marital, 'education': education,
    'default': default, 'housing': housing, 'loan': loan,
    'contact': contact, 'month': month, 'day_of_week': day_of_week,
    'duration': duration, 'campaign': campaign, 
    # 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
    'cons_price_idx': cons_price_idx,
    'cons_conf_idx': cons_conf_idx, 'euribor3m': euribor3m
}
df = pd.DataFrame([data])


# 6. Prediction Area with Centered Button
st.markdown("---")

# Centering Button: Create 3 columns, put button in middle
b1, b2, b3 = st.columns([1, 1, 1])
if b2.button("🚀 Predict Subscription Probability"):
    
    with st.spinner("Calculating..."):
        try:
            prediction0 = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]
            threshold = 0.7544693069918953
            prediction = 1 if probability >= threshold else 0

            # Display Results (Centered below button)
            r1, r2, r3 = st.columns([1, 2, 1])
            
            with r2:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                if prediction == 1:
                    st.balloons()
                    st.success("## ✅ High Likelihood of Subscription")
                else:
                    st.error("## ❌ Low Likelihood of Subscription")
                    
                st.metric(label="Subscription Probability", value=f"{probability:.2%}", delta=None)
                st.caption(f"Decision threshold: {threshold:.2f}")
                
                st.progress(probability)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
