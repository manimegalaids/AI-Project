import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

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

# 📊 Dataset Preview
st.subheader("1. Dataset Preview")
st.dataframe(df.head())
st.markdown(f"📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# 🔎 Data Quality Insights
st.subheader("2. Data Quality Overview")
missing = df.isnull().mean().sort_values(ascending=False)
if missing.any():
    st.warning("⚠️ Missing Data Detected")
    st.dataframe(missing[missing > 0])
else:
    st.success("✅ No missing values in the dataset.")
st.write("🔍 Unique Values per Feature:")
st.dataframe(df.nunique().sort_values(ascending=False))

# 📈 Correlation with G3
st.subheader("3. Attribute Comparison with Final Grade (G3)")
selected_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
corr = df[selected_cols].corr()['G3'].sort_values(ascending=False)
st.bar_chart(corr.drop("G3"))

fig, ax = plt.subplots()
sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 🧠 Model Comparison
st.subheader("4. Model Accuracy Comparison")
features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "Bayesian Ridge": BayesianRidge()
}
scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    scores[name] = r2_score(y_test, preds)

st.write("🔍 **R² Scores:**")
st.json(scores)
best_model = max(scores, key=scores.get)
st.success(f"✅ Best Model: {best_model} (R²: {scores[best_model]:.2f})")

# 🔍 Feature Importance for Random Forest
if best_model == "Random Forest":
    st.subheader("📊 Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": models[best_model].feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

# 📌 AI-Driven Socioeconomic Recommendations
st.subheader("5. AI-Driven Recommendations for Academic Support")

recommendations = []

if corr['Medu'] > 0.2 or corr['Fedu'] > 0.2:
    recommendations.append("📚 **Parental Education**: Students with better-educated parents (especially mothers) tend to perform better. Promote adult education and family engagement programs.")

if corr['failures'] < -0.3:
    recommendations.append("⏱️ **Failures**: Past failures significantly lower future academic performance. Implement early warning systems, mentoring, and after-school tutoring.")

if corr['studytime'] > 0.2:
    recommendations.append("📖 **Study Time**: More time spent studying correlates with better grades. Encourage structured study routines and productivity workshops.")

if corr['absences'] < -0.2:
    recommendations.append("🏫 **Absenteeism**: Higher absenteeism negatively impacts performance. Consider attendance incentives and parental counseling.")

if corr['traveltime'] < -0.1:
    recommendations.append("🚌 **Travel Time**: Long commute times reduce study opportunities. Suggest community learning centers or hybrid/online classes.")

if not recommendations:
    st.warning("No strong actionable insights found. Consider checking more features or using feature engineering.")
else:
    for rec in recommendations:
        st.info(rec)

# 📥 Downloadable Recommendation Report
st.subheader("6. Downloadable Report")
if st.button("📤 Download AI Recommendations"):
    rec_text = "\n\n".join(recommendations)
    st.download_button("📄 Download", rec_text, file_name="ai_recommendations.txt")

# 🎯 Predict Final Grade
st.subheader("7. Predict a Student's Final Grade (G3)")
with st.form("prediction_form"):
    age = st.slider("Age", 15, 22, 17)
    Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
    traveltime = st.slider("Travel Time (1=short <15min - 4=long >1hr)", 1, 4, 1)
    studytime = st.slider("Study Time (1=<2hrs - 4=>10hrs)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.slider("Absences", 0, 100, 5)
    G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([[age, Medu, Fedu, traveltime, studytime, failures, absences, G1, G2]], columns=features)
    input_scaled = scaler.transform(input_data)
    pred_g3 = models[best_model].predict(input_scaled)[0]
    st.success(f"🎓 Predicted Final Grade (G3): {pred_g3:.2f}")

# 💬 Chatbot
st.subheader("8. Ask an AI Bot")
user_question = st.text_input("Ask anything about student performance...")

def ai_bot_response(query):
    query = query.lower()
    if "parent" in query:
        return "👩‍🎓 Higher parental education, especially mothers' education, improves student performance."
    elif "fail" in query or "failure" in query:
        return "📉 Frequent failures need tutoring, mentoring, and academic support."
    elif "study" in query or "study time" in query:
        return "📖 Study time matters! Encourage structured routines like Pomodoro or spaced repetition."
    elif "absent" in query or "absences" in query:
        return "🏫 High absences impact performance. Use attendance incentives and parental engagement."
    elif "travel" in query:
        return "🚌 Long commutes reduce study time. Promote online/hybrid options or community centers."
    elif "improve" in query or "academic performance" in query:
        return (
            "🚀 To improve academic performance:\n"
            "- Support family education\n"
            "- Offer personalized tutoring\n"
            "- Ensure study routines\n"
            "- Reduce absenteeism\n"
            "- Provide emotional and academic support"
        )
    else:
        return "🤖 I'm here to help! Ask about parental education, failures, study time, or improving performance."

if user_question:
    st.write(ai_bot_response(user_question))
