import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Bati Bank Risk Radar", page_icon="🏦")
st.title("🏦 Bati Bank Credit Risk Radar")

# 1. Main Input Section
st.header("1. Input Customer Data")
col1, col2 = st.columns(2)

with col1:
    amount_mean = st.number_input("Average Transaction Amount", min_value=0.0, value=500.0)
    max_velocity = st.number_input("Max Velocity (Spending Speed)", min_value=0.0, value=1500.0)

with col2:
    avg_momentum = st.slider("Transaction Momentum", 0.0, 1.0, 0.5)
    # MOVED: Threshold is now here in the main view instead of the sidebar!
    threshold_value = st.number_input("Risk Threshold (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.markdown("---")

# 2. Prediction Section
if st.button("Evaluate Credit Risk", use_container_width=True):
    payload = {
        "amount_mean": amount_mean,
        "max_velocity": max_velocity,
        "avg_momentum": avg_momentum,
        "threshold": threshold_value
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        proba = result["probability"]
        
        # 3. Dynamic Results Display
        st.subheader("Assessment Result")
        
        # Visualize the probability vs threshold
        st.write(f"**Model Probability:** {proba:.2%} | **Your Threshold:** {threshold_value:.2%}")
        
        if result["is_high_risk"] == 1:
            st.error(f"🚨 **HIGH RISK ALERT**")
            st.progress(proba)
            st.write("Outcome: **LOAN DENIED** (Probability exceeds threshold)")
        else:
            st.success(f"✅ **LOW RISK APPROVED**")
            st.progress(proba)
            st.write("Outcome: **LOAN GRANTED** (Probability within threshold)")
        
        if "explanation" in result and result["explanation"]:
            st.markdown("---")
            st.subheader("🧠 Risk Factor Analysis (SHAP)")
            
            # Convert explanation dict to DataFrame for plotting
            expl_data = result["explanation"]
            expl_df = pd.DataFrame({
                "Feature": list(expl_data.keys()),
                "Influence": list(expl_data.values())
            }).sort_values(by="Influence")

            # Color code: Red for increasing risk, Green for decreasing
            expl_df["Color"] = ["#ff4b4b" if x > 0 else "#28a745" for x in expl_df["Influence"]]
            
            st.write("How much each feature contributed to this specific decision:")
            st.bar_chart(data=expl_df, x="Feature", y="Influence", color="Color")
            
            st.info("💡 **Positive (Red):** Pushes toward rejection. **Negative (Green):** Pushes toward approval.")
    except Exception as e:
        st.error(f"Error: API is likely offline. {e}")