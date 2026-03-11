import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Job Automation Risk Finder", layout="wide")

st.title("AI Job Automation Risk Finder")
st.write("Search for a job role and view its automation risk based on the dataset.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AI_DATASET.csv")
    return df

df = load_data()

# Clean text columns just in case
for col in ["job_role", "industry", "country", "automation_risk_category"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Sidebar info
st.sidebar.header("Search Job Role")
search_term = st.sidebar.text_input("Type a job role", "")

# Get all unique job roles
all_job_roles = sorted(df["job_role"].dropna().unique())

# Filter based on search term
if search_term:
    filtered_roles = [
        role for role in all_job_roles
        if search_term.lower() in role.lower()
    ]
else:
    filtered_roles = all_job_roles

# Show matching roles
selected_role = None

if len(filtered_roles) > 0:
    selected_role = st.sidebar.selectbox(
        "Matching Job Roles",
        filtered_roles
    )
else:
    st.sidebar.warning("No matching job roles found.")

# Main content
st.subheader("Dataset Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])

with col3:
    st.metric("Unique Job Roles", df["job_role"].nunique())

st.divider()

if selected_role:
    role_data = df[df["job_role"] == selected_role]

    # Compute summary stats
    avg_risk_percent = role_data["automation_risk_percent"].mean()
    most_common_risk = role_data["automation_risk_category"].mode()[0]
    common_industry = role_data["industry"].mode()[0] if not role_data["industry"].mode().empty else "N/A"
    common_country = role_data["country"].mode()[0] if not role_data["country"].mode().empty else "N/A"
    avg_ai_adoption = role_data["ai_adoption_level"].mean() if "ai_adoption_level" in role_data.columns else None
    avg_reskilling = role_data["reskilling_urgency_score"].mean() if "reskilling_urgency_score" in role_data.columns else None
    total_records = len(role_data)

    st.subheader("Automation Risk Result")
    st.success(f"Job Role: {selected_role}")
    st.info(f"Most Common Automation Risk Category: {most_common_risk}")

    colA, colB = st.columns(2)

    with colA:
        st.write(f"**Average Automation Risk Percent:** {avg_risk_percent:.2f}%")
        st.write(f"**Most Common Industry:** {common_industry}")
        st.write(f"**Most Common Country:** {common_country}")

    with colB:
        st.write(f"**Number of Records for this Role:** {total_records}")
        if avg_ai_adoption is not None:
            st.write(f"**Average AI Adoption Level:** {avg_ai_adoption:.2f}")
        if avg_reskilling is not None:
            st.write(f"**Average Reskilling Urgency Score:** {avg_reskilling:.2f}")

    # Explanation text
    st.subheader("Interpretation")
    if most_common_risk.lower() == "high":
        st.write(
            "This job role is commonly classified as **High Risk**, which means it is more vulnerable "
            "to automation and AI-driven replacement compared to other roles in the dataset."
        )
    elif most_common_risk.lower() == "medium":
        st.write(
            "This job role is commonly classified as **Medium Risk**, which means it may be partially "
            "affected by automation, but not fully replaced in the near term."
        )
    else:
        st.write(
            "This job role is commonly classified as **Low Risk**, which means it is less vulnerable "
            "to automation compared to other roles in the dataset."
        )

    st.divider()

    st.subheader("Matching Records")
    display_columns = [
        "job_role",
        "industry",
        "country",
        "year",
        "automation_risk_percent",
        "automation_risk_category",
        "ai_adoption_level",
        "reskilling_urgency_score"
    ]

    available_columns = [col for col in display_columns if col in role_data.columns]
    st.dataframe(role_data[available_columns].reset_index(drop=True))

else:
    st.subheader("How to Use")
    st.write(
        "Type a job role in the search box on the left sidebar. "
        "Then select one of the matching job roles to view its automation risk."
    )

    st.subheader("Sample Job Roles")
    st.dataframe(pd.DataFrame({"job_role": all_job_roles[:20]}))

st.divider()

st.subheader("About This App")
st.write(
    "This app reads all job roles from the uploaded CSV dataset. "
    "When a user searches for a job role, the app retrieves matching entries and shows "
    "the most common automation risk category along with related job statistics."
)
