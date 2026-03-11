import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(page_title="AI Job Automation Risk Finder", layout="wide")

st.title("AI Job Automation Risk Finder")
st.write(
    "Search for a job role and view its automation risk category, model prediction, "
    "and evaluation metrics."
)

# -------------------------
# Load and prepare dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AI_DATASET.csv")
    return df

df = load_data()

# Clean text columns
for col in ["job_role", "industry", "country", "automation_risk_category"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Encode target
risk_map = {"Low": 0, "Medium": 1, "High": 2}
label_map = {0: "Low", 1: "Medium", 2: "High"}

df["automation_risk_category_encoded"] = df["automation_risk_category"].map(risk_map)

# -------------------------
# Train model
# -------------------------
X = df[["job_role"]].copy()
y = df["automation_risk_category_encoded"]

le_job = LabelEncoder()
X["job_role"] = le_job.fit_transform(X["job_role"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# -------------------------
# Sidebar search
# -------------------------
st.sidebar.header("Search Job Role")

search_term = st.sidebar.text_input("Type a job role", "")

all_job_roles = sorted(df["job_role"].dropna().unique())

if search_term:
    filtered_roles = [
        role for role in all_job_roles
        if search_term.lower() in role.lower()
    ]
else:
    filtered_roles = all_job_roles

selected_role = None
if len(filtered_roles) > 0:
    selected_role = st.sidebar.selectbox("Matching Job Roles", filtered_roles)
else:
    st.sidebar.warning("No matching job roles found.")

# -------------------------
# Dataset background
# -------------------------
st.subheader("📈 Background of the Data Used")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", df.shape[0])
with col2:
    st.metric("Total Features", df.shape[1])
with col3:
    st.metric("Unique Job Roles", df["job_role"].nunique())

st.write(
    "This dataset contains job-related records with features related to AI adoption, "
    "automation, salaries, skills, and workforce disruption. "
    "The target variable is **automation_risk_category**, which classifies jobs into "
    "**Low**, **Medium**, or **High** automation risk."
)

# -------------------------
# Prediction result
# -------------------------
st.subheader("🗳️ Results")

if selected_role:
    encoded_role = le_job.transform([selected_role])[0]
    input_data = pd.DataFrame({"job_role": [encoded_role]})
    prediction = model.predict(input_data)[0]
    predicted_label = label_map[prediction]

    role_data = df[df["job_role"] == selected_role]

    avg_risk_percent = role_data["automation_risk_percent"].mean()
    most_common_risk = role_data["automation_risk_category"].mode()[0]
    common_industry = role_data["industry"].mode()[0] if not role_data["industry"].mode().empty else "N/A"
    common_country = role_data["country"].mode()[0] if not role_data["country"].mode().empty else "N/A"
    total_records = len(role_data)

    st.success(f"Selected Job Role: {selected_role}")
    st.info(f"Predicted Automation Risk Category: {predicted_label}")

    colA, colB = st.columns(2)
    with colA:
        st.write(f"**Most Common Risk Category in Dataset:** {most_common_risk}")
        st.write(f"**Average Automation Risk Percent:** {avg_risk_percent:.2f}%")
    with colB:
        st.write(f"**Most Common Industry:** {common_industry}")
        st.write(f"**Most Common Country:** {common_country}")
        st.write(f"**Number of Records for this Role:** {total_records}")

    st.markdown("**Note:**")
    st.write(
        "- **Predicted Automation Risk Category** is the model's output for the selected job role."
    )
    st.write(
        "- **Most Common Risk Category in Dataset** is the category most frequently associated with that job role in the CSV."
    )
    st.write(
        "- **Average Automation Risk Percent** shows the average percentage risk for that role based on the dataset."
    )
else:
    st.write("Search and select a job role from the sidebar to see its automation risk.")

# -------------------------
# Models used
# -------------------------
st.subheader("⚙️ Model Used")

st.write("**Model:** Decision Tree Classifier")
st.write(
    "We used a Decision Tree Classifier because it is easy to interpret and can classify "
    "job roles into Low, Medium, or High automation risk."
)

st.markdown("**Note:**")
st.write(
    "- The model learns patterns from the dataset and uses job role as the input feature"
    "to predict the automation risk category."
)

# -------------------------
# Model evaluation
# -------------------------
st.subheader("🔨 Model Evaluation")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("Accuracy", f"{acc:.4f}")
with metric_col2:
    st.metric("Precision", f"{prec:.4f}")
with metric_col3:
    st.metric("Recall", f"{rec:.4f}")
with metric_col4:
    st.metric("F1 Score", f"{f1:.4f}")

st.markdown("**Note:**")
st.write("- **Accuracy**: the overall percentage of correct predictions.")
st.write("- **Precision**: how reliable the model’s predicted classes are.")
st.write("- **Recall**: how well the model finds the actual classes.")
st.write("- **F1 Score**: the balanced measure of precision and recall.")

# -------------------------
# Confusion matrix
# -------------------------
st.subheader("📋 Confusion Matrix")

fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_yticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
st.pyplot(fig)

st.markdown("**Note:**")
st.write(
    "- The **rows** represent the **actual or true labels**."
)
st.write(
    "- The **columns** represent the **predicted labels**."
)
st.write(
    "- Values on the **diagonal** are correct predictions."
)
st.write(
    "- Values outside the diagonal are incorrect predictions."
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low", "Actual Medium", "Actual High"],
    columns=["Predicted Low", "Predicted Medium", "Predicted High"]
)

st.write("📏 Confusion Matrix Table:")
st.dataframe(cm_df)

# -------------------------
# Sample dataset
# -------------------------
st.subheader("6. Sample Dataset Preview")
st.write("This is only a sample preview of the dataset for demonstration purposes.")
st.dataframe(df.sample(min(20, len(df)), random_state=42).reset_index(drop=True))
