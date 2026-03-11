import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Job Risk Predictor", layout="wide")

st.title("AI Job Automation Risk Predictor")
st.write("Select a job role and the system will automatically predict its automation risk category.")

# Load dataset
df = pd.read_csv("AI_DATASET.csv")

# Encode target variable
risk_map = {"Low": 0, "Medium": 1, "High": 2}
label_map = {0: "Low", 1: "Medium", 2: "High"}

df["automation_risk_category_encoded"] = df["automation_risk_category"].map(risk_map)

# Use only job_role as input
X = df[["job_role"]].copy()
y = df["automation_risk_category_encoded"]

# Encode job_role
le_job = LabelEncoder()
X["job_role"] = le_job.fit_transform(X["job_role"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("Job Role Input")
job_role_input = st.sidebar.selectbox(
    "Select Job Role",
    sorted(df["job_role"].dropna().unique())
)

# Automatically predict when a role is selected
encoded_role = le_job.transform([job_role_input])[0]
input_data = pd.DataFrame({"job_role": [encoded_role]})
prediction = model.predict(input_data)[0]

# Main result
st.subheader("Prediction Result")
st.success(f"Predicted Automation Risk: {label_map[prediction]}")

# Show supporting info from dataset
role_data = df[df["job_role"] == job_role_input]

avg_risk = role_data["automation_risk_percent"].mean()
common_category = role_data["automation_risk_category"].mode()[0]
common_industry = role_data["industry"].mode()[0] if not role_data["industry"].mode().empty else "N/A"
common_country = role_data["country"].mode()[0] if not role_data["country"].mode().empty else "N/A"
record_count = len(role_data)

st.subheader("Job Role Summary")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Selected Job Role:** {job_role_input}")
    st.write(f"**Most Common Risk Category in Dataset:** {common_category}")
    st.write(f"**Average Automation Risk Percent:** {avg_risk:.2f}%")

with col2:
    st.write(f"**Number of Records for this Role:** {record_count}")
    st.write(f"**Most Common Industry:** {common_industry}")
    st.write(f"**Most Common Country:** {common_country}")

# Dataset information
st.subheader("Dataset Information")
st.write(f"**Total Records:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")
st.write(f"**Unique Job Roles:** {df['job_role'].nunique()}")

# Sample preview
st.subheader("Sample Dataset Preview")
st.dataframe(df.sample(min(20, len(df)), random_state=42))

# Model evaluation
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"**Model Accuracy:** {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_yticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
st.pyplot(fig)

# Explanation
st.subheader("How the Prediction Works")
st.write(
    "This version of the app uses job role as the only input feature. "
    "When a user selects a job role, the trained Decision Tree model predicts whether that role "
    "has Low, Medium, or High automation risk based on patterns learned from the dataset."
)
