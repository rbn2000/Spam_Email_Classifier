import streamlit as st
import pickle
import sys

from preprocess import clean_text

# If the script is executed with `python app.py` instead of
# `streamlit run app.py`, Streamlit's bootstrap module won't be
# present in `sys.modules`. Detect that case and exit with a
# helpful message to avoid the many runtime warnings.
import sys as _sys
if "streamlit.web.bootstrap" not in _sys.modules:
    print("Please run this app with: streamlit run app.py")
    _sys.exit(1)

# -----------------------------------------
# Load Saved Model
# -----------------------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# -----------------------------------------
# Title
# -----------------------------------------

st.title("📧 Spam Email Classifier")

st.write(
    """
This application uses **Machine Learning** to predict whether
an Email/SMS is **Spam** or **Ham (Not Spam)**.
"""
)

st.write("---")

# -----------------------------------------
# User Input
# -----------------------------------------

message = st.text_area(
    "Enter your Email/SMS Message",
    height=180
)

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        cleaned = clean_text(message)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        spam_probability = probability[0][1] * 100
        ham_probability = probability[0][0] * 100

        st.write("---")

        if prediction == 1:

            st.error("🚨 This is a SPAM message.")

        else:

            st.success("✅ This is NOT SPAM (HAM).")

        st.subheader("Prediction Confidence")

        st.write(f"Spam Probability : **{spam_probability:.2f}%**")

        st.write(f"Ham Probability : **{ham_probability:.2f}%**")