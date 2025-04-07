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

# 🌐 Page config
st.set_page_config(page_title="AI Academic Dashboard", layout="wide")

# 📥 Load Data
@st.cache_data
def load_data():
    df_mat = pd.read_csv("student-mat.csv", sep=';')
    df_por = pd.read_csv("student-por.csv", sep=';')
    return pd.concat([df_mat, df_por], axis=0).drop_duplicates().reset_index(drop=True)

df = load_data()
st.title("🎓 AI-Powered Academic Performance Dashboard")

# 📊 Dataset Preview
st.subheader("1. Dataset Preview")
st.dataframe(df.head())
st.markdown(f"📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# 📈 Correlation with G3
st.subheader("2. Attribute Comparison with Final Grade (G3)")
selected_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
corr = df[selected_cols].corr()['G3'].sort_values(ascending=False)
st.bar_chart(corr.drop("G3"))

fig, ax = plt.subplots()
sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 🧠 Model Comparison
st.subheader("3. Model Accuracy Comparison")
features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# 📌 Recommendation
st.subheader("4. AI-Driven Recommendation")
if corr['Medu'] > 0.2:
    st.info("📚 Parental education positively affects grades. Recommend promoting adult literacy.")
if corr['failures'] < -0.3:
    st.info("⏱️ Failures hurt performance. Recommend early academic intervention programs.")
if corr['studytime'] > 0.2:
    st.info("🕓 More study time helps. Recommend study-hour boosting strategies.")

# 💬 Chatbot
st.subheader("5. Ask an AI Bot")
user_question = st.text_input("Ask anything about student performance...")
if user_question:
    if "parent" in user_question.lower():
        st.write("👩‍🎓 Higher parental education often leads to better student outcomes.")
    elif "fail" in user_question.lower():
        st.write("📉 Students with more failures tend to perform worse. Early intervention helps.")
    elif "study" in user_question.lower():
        st.write("📖 More study time usually improves final grades.")
    else:
        st.write("🤖 I recommend exploring the dashboard insights for more details.")
