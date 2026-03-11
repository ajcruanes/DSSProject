import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

st.title("AI Job Automation Risk Predictor")
st.write("Predict whether a job has Low, Medium, or High automation risk using machine learning.")

# Load dataset
df = pd.read_csv("AI_DATASET.csv")

# Encode target
risk_map = {"Low":0, "Medium":1, "High":2}
label_map = {0:"Low",1:"Medium",2:"High"}

df["automation_risk_category_encoded"] = df["automation_risk_category"].map(risk_map)

# Features
features = [
    "automation_risk_percent",
    "ai_replacement_score",
    "skill_gap_index",
    "salary_before_usd",
    "salary_after_usd",
    "salary_change_percent",
    "skill_demand_growth_percent",
    "remote_feasibility_score",
    "ai_adoption_level",
    "education_requirement_level",
    "skill_transition_pressure",
    "wage_volatility_index",
    "reskilling_urgency_score",
    "ai_disruption_intensity"
]

X = df[features]
y = df["automation_risk_category_encoded"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# Sidebar input
st.sidebar.header("Enter Job Information")

automation_risk_percent = st.sidebar.slider("Automation Risk Percent",0,100,50)
ai_replacement_score = st.sidebar.slider("AI Replacement Score",0,100,50)
skill_gap_index = st.sidebar.slider("Skill Gap Index",0,100,50)
salary_before_usd = st.sidebar.number_input("Salary Before (USD)",30000,200000,50000)
salary_after_usd = st.sidebar.number_input("Salary After (USD)",30000,200000,55000)
salary_change_percent = st.sidebar.slider("Salary Change Percent",-50,50,0)
skill_demand_growth_percent = st.sidebar.slider("Skill Demand Growth Percent",0,20,5)
remote_feasibility_score = st.sidebar.slider("Remote Feasibility Score",0,100,50)
ai_adoption_level = st.sidebar.slider("AI Adoption Level",0,100,50)
education_requirement_level = st.sidebar.slider("Education Requirement Level",1,5,3)
skill_transition_pressure = st.sidebar.slider("Skill Transition Pressure",0,100,50)
wage_volatility_index = st.sidebar.slider("Wage Volatility Index",0,100,50)
reskilling_urgency_score = st.sidebar.slider("Reskilling Urgency Score",0,100,50)
ai_disruption_intensity = st.sidebar.slider("AI Disruption Intensity",0,100,50)

# Prediction input
input_data = np.array([[

automation_risk_percent,
ai_replacement_score,
skill_gap_index,
salary_before_usd,
salary_after_usd,
salary_change_percent,
skill_demand_growth_percent,
remote_feasibility_score,
ai_adoption_level,
education_requirement_level,
skill_transition_pressure,
wage_volatility_index,
reskilling_urgency_score,
ai_disruption_intensity

]])

if st.sidebar.button("Predict Automation Risk"):
    
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")
    st.success(f"Predicted Automation Risk: {label_map[prediction]}")

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Confusion matrix
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)
