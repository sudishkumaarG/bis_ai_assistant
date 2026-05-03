import streamlit as st
from app import BISRecommendationEngine

# --- Page Config ---
st.set_page_config(
    page_title="BIS AI Assistant",
    page_icon="🏗️",
    layout="centered"
)

# --- Title ---
st.title("🏗️ BIS AI Assistant")
st.markdown("### 🔍 Find the correct BIS standards instantly")

# --- Load Engine ---
@st.cache_resource
def load_engine():
    return BISRecommendationEngine()

engine = load_engine()

# --- Input ---
query = st.text_input(
    "Enter product description",
    placeholder="e.g. Ordinary Portland Cement 33 grade"
)

# --- Button ---
if st.button("🚀 Get Standards"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Analyzing BIS standards..."):
            codes, rationale, latency = engine.get_recommendation(query)

        st.divider()

        # --- Results ---
        st.subheader("📌 Recommended Standards")
        if codes:
            for i, code in enumerate(codes, 1):
                st.success(f"{i}. {code}")
        else:
            st.error("No standards found")

        # --- Rationale ---
        st.subheader("🧠 Explanation")
        st.info(rationale)

        # --- Confidence ---
        st.subheader("📊 Confidence Level")
        confidence = min(1.0, 0.6 + 0.1 * len(codes))
        st.progress(confidence)
        st.caption(f"Confidence Score: {int(confidence*100)}%")

        # --- Compliance Checker ---
        st.subheader("✔️ Compliance Check")
        user_standard = st.text_input("Check a standard (e.g. IS 269)")

        if user_standard:
            if user_standard.strip().lower().replace(" ", "") in [
                c.lower().replace(" ", "") for c in codes
            ]:
                st.success("✅ This standard is relevant")
            else:
                st.error("❌ Not found in recommendations")

        # --- Latency ---
        st.divider()
        st.caption(f"⚡ Response time: {round(latency, 3)} seconds")