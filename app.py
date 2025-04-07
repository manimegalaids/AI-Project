# AI-Augmented Socioeconomic Impact Analysis for Academic Performance
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

 🎯 Dashboard Settings
st.set_page_config(layout="wide")
st.title("AI-Powered Socio-Economic Dashboard")


# 🏁 Load and Merge Dataset
@st.cache_data
def load_data():
    df_mat = pd.read_csv("student-mat.csv", sep=';')
    df_por = pd.read_csv("student-por.csv", sep=';')
    df_combined = pd.concat([df_mat, df_por], axis=0).drop_duplicates().reset_index(drop=True)
    return df_combined

df = load_data()

# 📊 Dataset Overview
st.subheader("1. Dataset Overview")
st.dataframe(df.head())
st.markdown(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# 🔍 Socioeconomic Features vs. Performance
st.subheader("2. Socioeconomic Features vs. Academic Performance")
features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 🧠 Model Training
st.subheader("3. Train AI Model to Predict Final Grade (G3)")
model_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[model_features]
y = df['G3']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = BayesianRidge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
st.markdown(f"**RMSE:** {rmse:.2f} | **R² Score:** {r2:.2f}")

# 🔮 Real-Time Prediction
st.subheader("4. Real-Time Prediction Interface")
with st.form("prediction_form"):
    input_data = {
        'age': st.slider("Age", 10, 22, 17),
        'Medu': st.selectbox("Mother's Education", [0, 1, 2, 3, 4]),
        'Fedu': st.selectbox("Father's Education", [0, 1, 2, 3, 4]),
        'traveltime': st.slider("Travel Time (1-4)", 1, 4, 1),
        'studytime': st.slider("Study Time (1-4)", 1, 4, 2),
        'failures': st.slider("Failures", 0, 3, 0),
        'absences': st.slider("Absences", 0, 30, 4),
        'G1': st.slider("Grade G1", 0, 20, 10),
        'G2': st.slider("Grade G2", 0, 20, 10)
    }
    submit = st.form_submit_button("Predict G3")
    if submit:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 Predicted Final Grade (G3): **{prediction:.2f}**")

# 🧩 Socioeconomic Intervention Insight
st.subheader("5. Policy Recommendations Based on AI Insights")
if corr['G3']['Medu'] > 0.2 or corr['G3']['Fedu'] > 0.2:
    st.info("- Strong correlation between parental education and final grades.\n- Recommend community education programs to support families.")
if corr['G3']['failures'] < -0.3:
    st.info("- Negative correlation between failures and performance.\n- Suggest early intervention and tutoring programs.")

# 📝 Feedback
st.sidebar.header("📬 Feedback")
st.sidebar.text_area("Your suggestions")
if st.sidebar.button("Submit"):
    st.sidebar.success("Thank you for your feedback!")
