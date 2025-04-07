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

# Section 3: Attribute Comparison with Final Grade (G3)
st.header("📊 3. Attribute Comparison with Final Grade (G3)")
st.markdown("""
This section shows the relationship between each feature and the final student grade (G3) using:
- A **Lollipop Chart** to visualize correlation values.
- A **Random Forest** model to extract and show feature importances.
""")

# 🔧 Define selected_cols to avoid NameError
selected_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 1. Lollipop Chart for Correlation with G3
st.subheader("🔗 Correlation of Attributes with Final Grade (G3)")

# Calculate correlations
corr = df[selected_cols].corr()['G3'].drop('G3').sort_values()

# Plot Lollipop Chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.hlines(y=corr.index, xmin=0, xmax=corr.values, color='skyblue', linewidth=2)
ax.plot(corr.values, corr.index, "o", color='blue')
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Correlation with Final Grade (G3)")
ax.set_title("Lollipop Chart: Feature Correlation with Final Grade (G3)")
st.pyplot(fig)

# 2. Feature Importance from Random Forest
st.subheader("🌲 Feature Importance from Random Forest Model")

# Prepare data
X = df[selected_cols].drop(columns=['G3'])
y = df['G3']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importances
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()

# Plot Horizontal Bar Chart
fig2, ax2 = plt.subplots(figsize=(10, 6))
importances.plot(kind='barh', color='teal', ax=ax2)
ax2.set_xlabel("Importance Score")
ax2.set_title("Random Forest: Feature Importance for Final Grade (G3)")
st.pyplot(fig2)

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

# 🎯 Predict Final Grade + Recommend Learning Path
st.subheader("7. Predict Final Grade & Get Personalized Learning Path")

with st.form("combined_prediction_form"):
    st.markdown("📌 Enter student academic and socio-economic details:")
    age = st.slider("Age", 15, 22, 17)
   Medu = st.selectbox("Mother's Education Level", options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {
                        0: "0 - None",
                        1: "1 - Primary (up to 4th grade)",
                        2: "2 - 5th to 9th grade",
                        3: "3 - Secondary (High School)",
                        4: "4 - Higher (University)"
                    }[x])

    Fedu = st.selectbox("Father's Education Level", options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {
                        0: "0 - None",
                        1: "1 - Primary (up to 4th grade)",
                        2: "2 - 5th to 9th grade",
                        3: "3 - Secondary (High School)",
                        4: "4 - Higher (University)"
                    }[x])

    traveltime = st.slider("Travel Time (1=short <15min - 4=long >1hr)", 1, 4, 1)
    studytime = st.slider("Weekly Study Time (1=<2hrs - 4=>10hrs)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    absences = st.slider("Total Absences", 0, 100, 5)
    G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button("🎓 Predict & Recommend")

if submitted:
    input_data = pd.DataFrame([[age, Medu, Fedu, traveltime, studytime, failures, absences, G1, G2]], columns=features)
    input_scaled = scaler.transform(input_data)
    G3_pred = models[best_model].predict(input_scaled)[0]

    st.success(f"🎓 Predicted Final Grade (G3): {G3_pred:.2f}")

    # 📌 Personalized Academic Recommendations
    recommendations = []

    if G3_pred < 10:
        recommendations.append("🔴 **At-Risk Student**: Personalized tutoring sessions needed with focus on weak concepts from G1 & G2.")
        if failures > 0:
            recommendations.append("❌ Prior failures detected. Recommend academic counseling and regular progress tracking.")
        if studytime <= 2:
            recommendations.append("⏱️ Study time is low. Suggest time management coaching and digital learning planners.")
        if absences > 10:
            recommendations.append("🏫 High absenteeism. Engage with guardians and consider blended/remote learning models.")

    elif G3_pred < 14:
        recommendations.append("🟡 **Average Performer**: Recommend structured self-paced modules and performance goals.")
        if studytime <= 2:
            recommendations.append("📘 Boost study hours using techniques like Pomodoro and spaced repetition.")
        if absences > 5:
            recommendations.append("🕒 Reduce missed classes by sending automated alerts and reminders.")

    else:
        recommendations.append("🟢 **High Performer**: Recommend advanced learning paths or gifted programs.")
        if studytime > 3:
            recommendations.append("🚀 Encourage participation in competitions or online MOOCs (Coursera, edX).")

    st.markdown("### 🧑‍🏫 Recommended Actions:")
    for rec in recommendations:
        st.info(rec)

    # 📚 AI-Powered Learning Resource Recommender
    st.markdown("### 📚 Tailored Learning Resources")

    if G3_pred < 10:
        st.markdown("- [🎥 How to Study Effectively – Science-Based Tips (YouTube)](https://youtu.be/p60rN9JEapg)")
        st.markdown("- [⏱️ Pomodoro Timer Web Tool](https://pomofocus.io/)")
        st.markdown("- [📘 Time Management Course – Coursera](https://www.coursera.org/learn/work-smarter-not-harder)")
        st.markdown("- [📗 Khan Academy – Foundational Skills](https://www.khanacademy.org)")
        st.markdown("- [🧠 Motivation for Students – TEDx Talk](https://youtu.be/O96fE1E-rf8)")

    elif G3_pred < 14:
        st.markdown("- [🎓 Study Skills for High School & College – YouTube](https://youtu.be/CPxSzxylRCI)")
        st.markdown("- [📈 Focus & Productivity Guide – Todoist Blog](https://blog.todoist.com/productivity-methods/)")
        st.markdown("- [📚 Self-Paced Learning: Study Smarter](https://www.opencollege.info/self-paced-learning/)")

    else:
        st.markdown("- [🏆 Advanced MOOC: edX – Academic Excellence Courses](https://www.edx.org/learn/study-skills)")
        st.markdown("- [🎖️ Olympiad/Competition Preparation – Learn More](https://artofproblemsolving.com/)")
        st.markdown("- [🚀 Research Basics for Students – Google Scholar Guide](https://scholar.google.com/)")

# 💬 Chatbot
st.subheader("8. Chat bot")
import streamlit as st
import openai
import speech_recognition as sr
import pyttsx3

# Initialize OpenAI (Replace with your own key)
openai.api_key = "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"

# Set Streamlit page configuration
st.set_page_config(page_title="AI Dashboard", layout="wide")

# Sidebar page selector
page = st.sidebar.selectbox("Choose a page", [
    "Home",
    "Dataset Preview",
    "Attribute Analysis",
    "Model Accuracy Comparison",
    "AI-Based Recommendations",
    "Ask AI Bot"
])

# Chatbot: Session state for conversation memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Voice Recognition Function
def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand audio.")
    except sr.RequestError:
        st.error("API unavailable")
    return None

# Text-to-speech function
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Get AI response
def get_ai_response(prompt):
    messages = [{"role": "system", "content": "You are an AI academic assistant helping analyze student data and give insights."}]
    for user, ai in st.session_state.chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": prompt})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or gpt-4 if available
        messages=messages,
        max_tokens=200
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

# -------------------------------
# 🧠 ASK AI BOT PAGE
# -------------------------------
if page == "Ask AI Bot":
    st.title("🤖 Ask the AI Bot")
    st.write("Get insights and explanations about student academic performance, AI predictions, or socio-economic factors.")

    # Suggested Questions
    st.markdown("💡 **Try asking:**")
    st.markdown("- What factors most affect student performance?")
    st.markdown("- How does parental education influence grades?")
    st.markdown("- Can AI suggest how to improve a student's performance?")
    
    # Input options
    input_mode = st.radio("Choose input mode:", ["Text", "Voice"])
    user_input = ""

    if input_mode == "Text":
        user_input = st.text_input("Type your question:")
    else:
        if st.button("🎤 Start Voice Input"):
            user_input = recognize_voice()

    if user_input:
        with st.spinner("Thinking..."):
            response = get_ai_response(user_input)
            st.session_state.chat_history.append((user_input, response))
            st.success("Answer:")
            st.write(response)
            if st.checkbox("🔊 Read aloud"):
                speak(response)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## 🗂️ Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
