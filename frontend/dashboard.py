# pyrefly: ignore [missing-import]
import streamlit as st
import requests

st.set_page_config(page_title="Kenyan Sentiment API", page_icon="🇰🇪")

st.title("🇰🇪 Kenyan Code-Switched Sentiment Analyzer")
st.markdown("### Multilingual AI for Sheng, Swahili, and English")
st.write("This dashboard connects to our custom XLM-RoBERTa FastAPI backend.")

user_input = st.text_area("Enter customer feedback (e.g., 'hii app inanisumbua sana'):", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post("http://localhost:8000/predict", json={"text": user_input})
                if response.status_code == 200:
                    data = response.json()
                    st.divider()
                    st.subheader("Prediction Results:")
                    st.error(f"**Sentiment:** {data.get('sentiment')} 🔴")
                    st.write(f"**Confidence Score:** {data.get('confidence') * 100:.1f}%")
                else:
                    st.error("API Error")
            except:
                st.error("🚨 Could not connect to the backend. Ensure FastAPI is running on port 8000.")
