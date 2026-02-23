import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Telecom Insight", layout="wide")

@st.cache_resource
def load_assets():
    data = joblib.load("churn_model.pkl")
    return data["model"], data["features"], data["threshold"]

model, all_features, threshold = load_assets()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Modify values to see real-time risk updates.")

# Inputs with range disclaimers
rev = st.sidebar.slider("Monthly Revenue ($)", 0.0, 250.0, 58.0)
st.sidebar.caption("Typical: 30 - 120")

tenure = st.sidebar.slider("Months in Service", 1, 72, 18)
st.sidebar.caption("Typical: 6 - 36 months")

calls = st.sidebar.number_input("Total Calls", 0, 5000, 500)
st.sidebar.caption("Typical: 100 - 2,000")

dropped = st.sidebar.slider("Dropped Calls", 0, 50, 2)
st.sidebar.caption("Typical: 0 - 10")

price = st.sidebar.selectbox("Handset Price Tier", ['Unknown', '30', '80', '150', '250', '500'], index=3)

# --- LOGIC: DATA PREP ---
# Create a full 56-column row with defaults
input_dict = {feat: 0 for feat in all_features}
# Categorical defaults
for feat in all_features:
    if feat in ['ServiceArea', 'HandsetPrice', 'CreditRating', 'Occupation']:
        input_dict[feat] = 'Unknown'

# Update with our 5 sliders
input_dict.update({
    'MonthlyRevenue': rev,
    'MonthsInService': tenure,
    'TotalCalls': calls,
    'DroppedCalls': dropped,
    'HandsetPrice': price
})

# Get Prediction
input_df = pd.DataFrame([input_dict])
prob = model.predict_proba(input_df)[0, 1]

# --- MAIN DASHBOARD ---
st.title("Customer Churn Analysis Dashboard")
st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Predictive Risk Score")
    
    # Create the Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Churn Probability %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, threshold*100], 'color': "#00CC96"}, # Green
                {'range': [threshold*100, 100], 'color': "#EF553B"}  # Red
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("AI Verdict")
    risk_level = "🔴 HIGH RISK" if prob >= threshold else "🟢 LOW RISK"
    
    st.markdown(f"### Current Status: **{risk_level}**")
    
    if prob >= threshold:
        st.error(f"Customer is above the {threshold*100:.0f}% danger threshold.")
        st.info("💡 **Recommendation:** Offer a plan upgrade or loyalty discount immediately.")
    else:
        st.success(f"Customer is below the {threshold*100:.0f}% danger threshold.")
        st.info("💡 **Recommendation:** Maintain current engagement; customer is stable.")

    # Show a metric with delta
    st.metric("Risk Score", f"{prob*100:.1f}%", delta=f"{(prob-threshold)*100:.1f}% vs Threshold", delta_color="inverse")

# --- IMPACT SECTION ---
st.markdown("---")
st.subheader("Feature Impact Comparison")
st.write("How this customer compares to the average (baseline) customer:")

# Simple bar chart comparison
comp_data = pd.DataFrame({
    "Metric": ["Revenue", "Tenure", "Dropped Calls"],
    "Current Customer": [rev, tenure, dropped],
    "Average": [58.0, 18.0, 2.0]
})
st.bar_chart(comp_data.set_index("Metric"))

# --- ROI CALCULATOR SECTION ---
st.markdown("---")
st.subheader("💰 Business Impact Calculator")
st.write("Estimate the potential revenue saved by taking action on this customer.")

with st.expander("Configure Retention Economics"):
    # User can adjust these based on business assumptions
    success_rate = st.slider("Retention Success Rate (%)", 0, 100, 40, help="What % of high-risk customers actually stay when offered a discount?")
    discount_offered = st.number_input("Cost of Retention Offer ($)", value=20.0, help="The value of the discount/gift given to the customer.")

# Mathematical Calculation
# Potential Saved Revenue = (Monthly Revenue * Success Rate) - Cost of Offer
potential_savings = (rev * (success_rate / 100)) - discount_offered

if prob >= threshold:
    st.info(f"### Potential Monthly Saving: ${potential_savings:.2f}")
    st.write(f"**Explanation:** If we intervene now with a ${discount_offered} offer, and have a {success_rate}% chance of success, we preserve the revenue stream of ${rev}/month.")
    
    # Visualization of ROI
    if potential_savings > 0:
        st.success(f"**ROI Insight:** The intervention pays for itself. You save ${potential_savings} more than the cost of the offer.")
    else:
        st.warning("**ROI Insight:** The cost of this offer might be too high compared to the customer's monthly value.")
else:
    st.write("This customer is low risk. No retention spending is required at this time.")