# 📌 AI-Driven Socioeconomic Recommendations
st.subheader("4. AI-Driven Recommendations for Academic Support")

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
