import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import speech_recognition as sr
import numpy as np
import pyttsx3
import torch
import datetime
import joblib
from fpdf import FPDF
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🌐 Page Config
st.set_page_config(page_title="AI Academic Dashboard", layout="wide")
st.title("🎓 AI-Powered Academic Performance Dashboard")

# 📥 Load Data
@st.cache_data
def load_data():
    df_mat = pd.read_csv("student-mat.csv", sep=';')
    df_por = pd.read_csv("student-por.csv", sep=';')
    return pd.concat([df_mat, df_por], axis=0).drop_duplicates().reset_index(drop=True)

df = load_data()

import io

# Load dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment this in your project

# Rename columns to full names for better readability
readable_df = df.rename(columns={
    "school": "School",
    "sex": "Gender",
    "age": "Age",
    "address": "Address Type",
    "famsize": "Family Size",
    "Pstatus": "Parent Status",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "Mjob": "Mother's Job",
    "Fjob": "Father's Job",
    "reason": "Reason for School",
    "guardian": "Guardian",
    "traveltime": "Travel Time",
    "studytime": "Study Time",
    "failures": "Past Failures",
    "schoolsup": "School Support",
    "famsup": "Family Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery",
    "higher": "Wants Higher Education",
    "internet": "Internet Access",
    "romantic": "Romantic Relationship",
    "famrel": "Family Relationship",
    "freetime": "Free Time",
    "goout": "Going Out Frequency",
    "Dalc": "Weekday Alcohol Use",
    "Walc": "Weekend Alcohol Use",
    "health": "Health Status",
    "absences": "Absences",
    "G1": "Grade 1",
    "G2": "Grade 2",
    "G3": "Final Grade"
})

# 📊 Enhanced Dataset Preview with Multi-Filters, Charts, Summary & Download
st.subheader("1. 📋 Dataset Preview with Filters, Charts & Download")

# Rename confusing column names
column_rename_map = {
    'Medu': "Mother's Education",
    'Fedu': "Father's Education",
    'G1': "Grade Period 1",
    'G2': "Grade Period 2",
    'G3': "Final Grade",
    'traveltime': "Travel Time",
    'studytime': "Weekly Study Time",
    'failures': "Past Failures",
    'absences': "Total Absences",
    'age': "Student Age",
    'schoolsup': "School Support",
    'famsup': "Family Support",
    'sex': "Gender",
    'school': "School"
}
readable_df = df.rename(columns=column_rename_map)

# Fix serialization warning
readable_df = readable_df.astype({col: 'string' for col in readable_df.select_dtypes('object').columns})

# --- Multi-column Filtering ---
with st.expander("🔍 Apply Filters"):
    filter_cols = st.multiselect("Select columns to filter", readable_df.columns)
    filtered_df = readable_df.copy()

    for col in filter_cols:
        unique_vals = filtered_df[col].dropna().unique()
        selected_vals = st.multiselect(f"Filter {col}", unique_vals, key=col)
        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Show filtered dataset
st.dataframe(filtered_df.head(50), use_container_width=True)
st.markdown(f"📌 Showing {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")

# --- Summary & Description ---
with st.expander("📈 View Data Summary & Description"):
    st.write("🔢 **Statistical Summary (Numeric Columns)**")
    st.dataframe(filtered_df.describe())

    st.write("📋 **Column Info (Non-numeric)**")
    st.dataframe(filtered_df.select_dtypes(include='string').nunique().to_frame(name="Unique Values"))

# --- Charts: Grade & Absences Distribution ---
if 'Final Grade' in filtered_df.columns:
    st.subheader("📊 Grade Distribution")
    fig, ax = plt.subplots()
    filtered_df['Final Grade'].astype(float).hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

if 'Total Absences' in filtered_df.columns:
    st.subheader("📊 Absences Distribution")
    fig2, ax2 = plt.subplots()
    filtered_df['Total Absences'].astype(float).hist(bins=20, color='orange', edgecolor='black', ax=ax2)
    ax2.set_xlabel("Total Absences")
    ax2.set_ylabel("Number of Students")
    st.pyplot(fig2)

# --- Download Button ---
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Dataset as CSV",
    data=csv,
    file_name="filtered_student_data.csv",
    mime="text/csv"
)

# --- Data summary ---
st.subheader("📈 2. Dataset Summary Statistics")
st.dataframe(filtered_df.describe(include='all'))

# --- Quick visual summary ---
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📊 3. Grade Distribution Plot")
fig, ax = plt.subplots()
sns.histplot(filtered_df["Final Grade"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("🧠 Correlation with Final Grade")
correlation = filtered_df.corr(numeric_only=True)["Final Grade"].sort_values(ascending=False)
st.bar_chart(correlation)

# 🔎 Data Quality Insights
st.subheader("2. Data Quality Overview")

# --- Missing Data Overview ---
missing = df.isnull().mean().sort_values(ascending=False)

if missing.any():
    st.warning("⚠️ Missing Data Detected")
    st.dataframe(missing[missing > 0].apply(lambda x: f"{x*100:.2f}%", axis=0))

    # 📊 Missing Data Bar Chart
    st.bar_chart(missing[missing > 0])
    
    # Optional: Allow user to drop columns with too much missing data
    if st.checkbox("🧹 Drop columns with more than 30% missing values"):
        drop_cols = missing[missing > 0.3].index.tolist()
        df.drop(columns=drop_cols, inplace=True)
        st.success(f"Dropped columns: {', '.join(drop_cols)}")
else:
    st.success("✅ No missing values in the dataset.")

# --- Unique Values Overview ---
with st.expander("🔍 Unique Values per Feature"):
    st.dataframe(df.nunique().sort_values(ascending=False))

# --- Low Variance Features ---
st.markdown("📉 **Low Variance Features (May be less useful in ML models)**")
low_variance = df.nunique() <= 1
if low_variance.any():
    st.warning("⚠️ Columns with only one unique value:")
    st.write(df.columns[low_variance].tolist())
else:
    st.success("✅ All columns have more than one unique value.")


# 📊 Section 3: Attribute Comparison with Final Grade (G3)
st.header("📊 3. Attribute Comparison with Final Grade (G3)")

# 📦 Data Type Grouping
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# 🎛️ Tabs for plot types
tabs = st.tabs(["🔢 Scatter Plot", "🧰 Box Plot", "📊 Bar Plot", "📈 Histogram", "🎻 Violin Plot", "🥧 Pie Chart", "🔥 Correlation Heatmap"])

# 1️⃣ Scatter Plot Tab
with tabs[0]:
    st.subheader("🔢 Scatter Plot: Numeric Features vs Final Grade (G3)")
    selected_numeric = st.multiselect("Select numeric features", options=numeric_cols, default=['studytime', 'absences'])
    for col in selected_numeric:
        if col != 'G3':
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col, y='G3', ax=ax)
            ax.set_title(f"{col} vs G3")
            st.pyplot(fig)

# 2️⃣ Box Plot Tab
with tabs[1]:
    st.subheader("🧰 Box Plot: Categorical Features vs Final Grade (G3)")
    selected_box = st.multiselect("Select categorical features", options=categorical_cols, default=['sex', 'school'])
    for col in selected_box:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col, y='G3', ax=ax)
        ax.set_title(f"{col} vs G3")
        st.pyplot(fig)

# 3️⃣ Bar Plot Tab
with tabs[2]:
    st.subheader("📊 Bar Plot: Average Final Grade by Category")
    selected_bar = st.selectbox("Choose a categorical feature", options=categorical_cols)
    avg_data = df.groupby(selected_bar)['G3'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_data, x=selected_bar, y='G3', ax=ax)
    ax.set_title(f"Average G3 by {selected_bar}")
    st.pyplot(fig)

# 4️⃣ Histogram Tab
with tabs[3]:
    st.subheader("📈 Histogram: Distribution of Final Grades (G3)")
    fig, ax = plt.subplots()
    sns.histplot(df['G3'], bins=10, kde=True, ax=ax)
    ax.set_title("Distribution of G3")
    st.pyplot(fig)

# 5️⃣ Violin Plot Tab
with tabs[4]:
    st.subheader("🎻 Violin Plot: G3 Distribution by Category")
    selected_violin = st.selectbox("Select a categorical feature", options=categorical_cols, key='violin')
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x=selected_violin, y='G3', ax=ax)
    ax.set_title(f"Violin Plot of G3 by {selected_violin}")
    st.pyplot(fig)

# 6️⃣ Pie Chart Tab
with tabs[5]:
    st.subheader("🥧 Pie Chart: Distribution of Categorical Feature")
    selected_pie = st.selectbox("Choose a feature for pie chart", options=categorical_cols, key='pie')
    pie_data = df[selected_pie].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f"Distribution of {selected_pie}")
    st.pyplot(fig)

# 7️⃣ Heatmap Tab
with tabs[6]:
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    st.pyplot(fig)

# 4. 📊 Model Accuracy Comparison (R² Score for Final Grade Prediction)
st.subheader("📊 4. Model Accuracy Comparison")

from sklearn.metrics import r2_score

# 🔁 Function to train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "Bayesian Ridge": BayesianRidge()
    }
    scores = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        scores[name] = score
        trained_models[name] = model
    return scores, trained_models

# ⚙️ Prepare Data
features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📈 Train and get scores
scores, trained_models = train_models(X_train, X_test, y_train, y_test)

# 🔍 Best model detection
best_model_name = max(scores, key=scores.get)
best_score = scores[best_model_name]

# 📊 Plot Horizontal Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['green' if m == best_model_name else 'skyblue' for m in scores.keys()]
bars = ax.barh(list(scores.keys()), list(scores.values()), color=colors)

for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.2f}', va='center', fontsize=10)

ax.set_xlim(0, 1)
ax.set_xlabel("R² Score")
ax.set_title("🔍 Model Comparison: R² Score for Predicting Final Grade (G3)")

st.pyplot(fig)

# ✅ Best model output
st.success(f"🏆 Best Model: **{best_model_name}** with R² Score of **{best_score:.2f}**")

# 📌 Optional: Feature importance if applicable
if best_model_name == "Random Forest":
    st.markdown("### 📌 Top Contributing Features (Random Forest)")
    feature_imp = trained_models[best_model_name].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_imp
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='crest', ax=ax2)
    ax2.set_title("Feature Importance from Random Forest")
    st.pyplot(fig2)

#st.title("🎓 Predict Final Grade & Get Personalized Learning Path")

# Load dataset
df_mat = pd.read_csv("student-mat.csv", sep=';')
df_por = pd.read_csv("student-por.csv", sep=';')
df = pd.concat([df_mat, df_por], axis=0).drop_duplicates().reset_index(drop=True)

# Feature selection
features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
target = 'G3'
X = df[features]
y = df[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
br_model = BayesianRidge()

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
br_model.fit(X_train, y_train)

# Evaluate
model_scores = {
    "Random Forest": r2_score(y_test, rf_model.predict(X_test)),
    "XGBoost": r2_score(y_test, xgb_model.predict(X_test)),
    "Bayesian Ridge": r2_score(y_test, br_model.predict(X_test))
}

# Store trained models
trained_models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Bayesian Ridge": br_model
}

# Sidebar model selector or auto-best
st.sidebar.markdown("### Select a Model (or leave for Auto Best)")
user_model_choice = st.sidebar.selectbox("Choose Model", ["Auto Best"] + list(model_scores.keys()))

if user_model_choice == "Auto Best":
    best_model_name = max(model_scores, key=model_scores.get)
else:
    best_model_name = user_model_choice

st.sidebar.success(f"✅ Using: {best_model_name}")

# Form for input
with st.form("prediction_form"):
    st.markdown("📌 Enter student academic and socio-economic details:")
    age = st.slider("Age", 15, 22, 17)
    Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
    traveltime = st.slider("Travel Time (1=short <15min - 4=long >1hr)", 1, 4, 1)
    studytime = st.slider("Weekly Study Time (1=<2hrs - 4=>10hrs)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.slider("Total Absences", 0, 100, 5)
    G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
    submitted = st.form_submit_button("🎓 Predict & Recommend")

if submitted:
    input_df = pd.DataFrame([[age, Medu, Fedu, traveltime, studytime, failures, absences, G1, G2]], columns=features)
    input_scaled = scaler.transform(input_df)
    model = trained_models[best_model_name]
    G3_pred = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Final Grade (G3): {G3_pred:.2f}")

    st.markdown("### 🧑‍🏫 Personalized Academic Recommendations")
    recs = []

    if G3_pred < 10:
        recs.append("🔴 At-Risk Student: Consider personalized tutoring and learning plans.")
        if failures > 0:
            recs.append("❌ Prior failures detected. Academic counseling is recommended.")
        if studytime <= 2:
            recs.append("📘 Low study time. Encourage time management and productivity techniques.")
        if absences > 10:
            recs.append("🏫 High absenteeism. Parental engagement and hybrid learning options suggested.")

    elif G3_pred < 14:
        recs.append("🟡 Average Performer: Promote structured self-learning modules.")
        if absences > 5:
            recs.append("🕒 Moderate absenteeism. Suggest attendance tracking systems.")

    else:
        recs.append("🟢 High Performer: Recommend enrichment activities and advanced programs.")
        if studytime > 3:
            recs.append("🚀 Eligible for competitive programs and MOOCs.")

    for r in recs:
        st.info(r)

    st.markdown("### 📚 Suggested Resources")
    if G3_pred < 10:
        st.markdown("- [Time Management Course](https://www.coursera.org/learn/work-smarter-not-harder)")
        st.markdown("- [Khan Academy](https://www.khanacademy.org)")
    elif G3_pred < 14:
        st.markdown("- [Focus Techniques - Todoist Blog](https://blog.todoist.com/productivity-methods/)")
    else:
        st.markdown("- [Advanced MOOCs on edX](https://www.edx.org/learn/study-skills)")

# ✅ Ensure recommendations exist in session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []


# 🤖 Section 8: Chatbot Assistant
st.subheader("6.Chatbot Assistant")
user_question = st.text_input("Ask anything about student performance...")

def ai_bot_response(query):
    query = query.lower()
    
    if "parent" in query:
        return (
            "👩‍🎓 Higher parental education, especially mothers' education, "
            "has a strong positive impact on student grades. Community-based "
            "parental literacy programs can help bridge this gap."
        )
    elif "fail" in query or "failure" in query:
        return (
            "📉 Frequent failures indicate students need timely intervention. "
            "Tutoring, mentoring, and academic support programs are effective "
            "in reducing future failure rates and boosting confidence."
        )
    elif "study" in query or "study time" in query:
        return (
            "📖 Increased study time often correlates with better academic performance. "
            "Encourage consistent daily study routines and focused learning strategies "
            "like Pomodoro or spaced repetition."
        )
    elif "absent" in query or "absences" in query:
        return (
            "🏫 High absences can hurt learning progress. Schools should promote attendance "
            "through counseling, parental involvement, and incentives for regular attendance."
        )
    elif "travel" in query:
        return (
            "🚌 Longer travel time might reduce study hours. Providing access to nearby learning "
            "centers or online learning options can help."
        )
    elif "improve" in query or "academic performance" in query:
        return (
            "🚀 To improve academic performance:\n"
            "- Support family education (especially mothers)\n"
            "- Offer personalized tutoring for struggling students\n"
            "- Ensure daily study routines\n"
            "- Reduce student absenteeism\n"
            "- Provide socio-emotional support\n\n"
            "✨ Use the dashboard's AI predictions to identify at-risk students early!"
        )
    else:
        return (
            "🤖 I'm here to help! You can ask about factors like parental education, failures, "
            "study time, or how to improve student grades."
        )

if user_question:
    st.write(ai_bot_response(user_question))
