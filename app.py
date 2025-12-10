import streamlit as st
import pandas as pd
from predict import predict_url

st.set_page_config(page_title="🔐 URL Security Analyzer", layout="wide")

st.title("🔐 URL Malware & Phishing Detector")
st.write("نظام كشف الروابط الخبيثة باستخدام الذكاء الاصطناعي (Machine Learning)")

tab1, tab2 = st.tabs(["🔍 فحص رابط", "📂 رفع ملف CSV"])

with tab1:
    url = st.text_input("أدخل الرابط هنا:")
    if st.button("فحص الرابط"):
        label, prob = predict_url(url)
        st.subheader("🔎 النتيجة:")

        # ✅ هنا التصحيح الحقيقي
      if label == 1:
    st.error(f"⚠️ الرابط خبيث بنسبة {prob*100:.2f}%")
    st.progress(prob)
else:
    safe_prob = 1 - prob
    st.success(f"✔️ الرابط سليم بنسبة {safe_prob*100:.2f}%")
    st.progress(safe_prob)


with tab2:
    file = st.file_uploader("ارفع ملف CSV يحتوي على عمود url", type=["csv"])
    if file:
        df = pd.read_csv(file)
        results = []
        for u in df["url"]:
            pred, prob = predict_url(u)

            # الحساب الصحيح
            safe_prob = (1 - prob) if pred == 0 else prob

            results.append([u, pred, safe_prob])

        output_df = pd.DataFrame(results, columns=["url", "prediction", "probability"])
        st.dataframe(output_df)

        st.download_button(
            label="⬇️ تحميل النتائج",
            data=output_df.to_csv(index=False),
            file_name="scan_results.csv",
            mime="text/csv"
        )
