import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Job Risk Predictor", layout="wide")

st.title("AI Job Automation Risk Predictor")
st.write("Predict whether a job has Low, Medium, or High automation risk using machine learning.")

# Load dataset
df = pd.read_csv("AI_DATASET.csv")

# Encode target variable
risk_map = {"Low": 0, "Medium": 1, "High": 2}
label_map = {0: "Low", 1: "Medium", 2: "High"}

df["automation_risk_category_encoded"] = df["automation_risk_category"].map(risk_map)

# Features for the model
features = ["job_role", "industry", "country", "year"]
X = df[features]
y = df["automation_risk_category_encoded"]

# Encode categorical variables
le_job = LabelEncoder()
le_industry = LabelEncoder()
le_country = LabelEncoder()

X["job_role"] = le_job.fit_transform(X["job_role"])
X["industry"] = le_industry.fit_transform(X["industry"])
X["country"] = le_country.fit_transform(X["country"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter Job Information")

job_role_input = st.sidebar.selectbox(
    "Job Role", sorted(df["job_role"].unique())
)

industry_input = st.sidebar.selectbox(
    "Industry", sorted(df["industry"].unique())
)

country_input = st.sidebar.selectbox(
    "Country", sorted(df["country"].unique())
)

year_input = st.sidebar.slider(
    "Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median())
)

# Prediction
if st.sidebar.button("Predict Automation Risk"):

    input_data = pd.DataFrame({
        "job_role": [le_job.transform([job_role_input])[0]],
        "industry": [le_industry.transform([industry_input])[0]],
        "country": [le_country.transform([country_input])[0]],
        "year": [year_input]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Automation Risk: {label_map[prediction]}")

    # Additional information
    similar_jobs = df[df["job_role"] == job_role_input]
    avg_risk = similar_jobs["automation_risk_percent"].mean()

    st.write("Average Automation Risk Percent for this job role:")
    st.write(f"{avg_risk:.2f}%")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Model evaluation
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.write("Confusion Matrix Values:")
st.write(cm)
