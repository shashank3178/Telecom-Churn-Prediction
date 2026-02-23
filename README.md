##Telecom Churn Prediction Dashboard
This repository contains a machine learning application designed to predict the likelihood of customer churn in the telecommunications sector. It features a real-time interactive dashboard built with Streamlit and a high-performance XGBoost model.

#Project Overview
The goal of this project is to identify "High Risk" customers based on their usage patterns and profile. The app provides:

Risk Probability: Real-time inference using a calibrated XGBoost model.

Feature Interaction: A sidebar to adjust customer metrics (Revenue, Tenure, Calls, etc.).

ROI Calculator: A business tool to estimate the financial impact of retention offers.

#File Structure
app4-main.py: The main Streamlit application script.

churn_model.pkl: The serialized model object (contains the pipeline, feature names, and optimal threshold).

requirements.txt: List of necessary Python libraries.

cell2cell-xgb-model.ipynb: The original development notebook containing data analysis and training logic.

#Getting Started
Follow these steps to set up the environment on your local machine.

1. Clone or Download the Project
Ensure all files (app4-main.py, churn_model.pkl, and requirements.txt) are in the same folder.

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

Windows:

Bash
python -m venv venv
venv\Scripts\activate
macOS/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
With the virtual environment activated, install the required libraries:

Bash
pip install -r requirements.txt
4. Run the Application
Launch the dashboard using the Streamlit command:

Bash
streamlit run app4-main.py
After running this, a local URL (usually http://localhost:8501) will open in your web browser.

#Usage Guide
Adjusting Customer Data
Use the Sidebar to input customer specifics:

Monthly Revenue: The average billing amount.

Months in Service: How long the customer has been with the provider.

Dropped/Blocked Calls: Indicators of service quality issues.

Understanding the Results
Risk Gauge: Shows the probability of churn. If the value exceeds the model's threshold, the status will flip to High Risk.

Business Impact: Use the slider in the "Business Impact Calculator" to see if a retention discount is financially viable based on the customer's revenue.

#Model Details
Algorithm: XGBoost (Extreme Gradient Boosting).

Calibration: The model uses CalibratedClassifierCV to ensure that predicted probabilities reflect real-world frequencies.

Preprocessing: Includes a pipeline with SimpleImputer (median strategy), StandardScaler, and OneHotEncoder.