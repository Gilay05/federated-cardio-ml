import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SERVER = "https://federated-central-server.onrender.com"

st.set_page_config(page_title="Federated Machine Learning", layout="wide")

st.title("Federated Machine Learning")
st.markdown("---")

# ============================
# DATASET INFORMATION
# ============================

st.header("Dataset Information")

st.markdown("""
**Cardiovascular Disease Dataset**

• 70,000 instances  
• 12 attributes  

Binary Target Variable:

• **Class 0 — No Cardiovascular Disease:** 35,021  
• **Class 1 — Has Cardiovascular Disease:** 34,979  

Source:  
https://www.kaggle.com/datasets/salomemweluscherer/cardiovascular-disease-dataset
""")

st.markdown("---")

# ============================
# RUN PIPELINE
# ============================

if st.button("Run Federated Learning Pipeline"):

    with st.spinner("Running pipeline..."):
        pipeline = requests.get(f"{SERVER}/run_full_pipeline").json()

    baseline = requests.get(f"{SERVER}/baseline_metrics").json()

    st.success("Pipeline Completed")

    # ============================
    # BASELINE MODEL PERFORMANCE
    # ============================

    st.header("Baseline Model Performance")

    df_base = pd.DataFrame([baseline])
    st.dataframe(df_base)

    # Confusion Matrix (estimated from metrics)

    st.subheader("Confusion Matrix (Baseline Model)")

    tp = baseline["recall"] * 35000
    fn = 35000 - tp
    fp = (tp / baseline["precision"]) - tp
    tn = 35000 - fp

    cm = pd.DataFrame(
        [[tn, fp],
         [fn, tp]],
        columns=["Predicted 0", "Predicted 1"],
        index=["Actual 0", "Actual 1"]
    )

    fig_cm = px.imshow(cm,
                       text_auto=True,
                       color_continuous_scale="Blues")

    st.plotly_chart(fig_cm)

    st.markdown("---")

    # ============================
    # HOSPITAL PERFORMANCE
    # ============================

    st.header("Hospital Performance")

    h1 = pipeline["hospital1_logic"]
    h2 = pipeline["hospital2_logic"]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Hospital 1 Analysis")

        st.write("Decision:", h1["decision"])

        df_h1 = pd.DataFrame(
            [baseline, h1["initial"], h1["final_full"]],
            index=["Baseline", "Initial Test", "Final Test"]
        )

        st.dataframe(df_h1)

        fig1 = px.bar(
            df_h1.reset_index(),
            x="index",
            y="accuracy",
            color="index",
            color_discrete_map={
                "Baseline": "white",
                "Initial Test": "light blue",
                "Final Test": "blue"
            },
            title="Hospital 1 Accuracy vs Baseline"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.subheader("Hospital 2 Analysis")

        st.write("Decision:", h2["decision"])

        df_h2 = pd.DataFrame(
            [baseline, h2["initial"], h2["final_full"]],
            index=["Baseline", "Initial Test", "Final Test"]
        )

        st.dataframe(df_h2)

        fig2 = px.bar(
            df_h2.reset_index(),
            x="index",
            y="accuracy",
            color="index",
            color_discrete_map={
                "Baseline": "white",
                "Initial Test": "light blue",
                "Final Test": "blue"
            },
            title="Hospital 2 Accuracy vs Baseline"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ============================
    # GLOBAL MODEL PERFORMANCE
    # ============================

    st.header("Global Model Performance (main_model_v2)")

    g = pipeline["global_test"]

    g1 = g["hospital1_global_test"]
    g2 = g["hospital2_global_test"]

    df_global = pd.DataFrame([g1, g2], index=["Hospital 1", "Hospital 2"])

    st.dataframe(df_global)

    fig3 = px.bar(
        df_global.reset_index(),
        x="index",
        y="accuracy",
        color="index",
        color_discrete_sequence=["white", "light blue"],
        title="Global Model Accuracy Across Hospitals"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ============================
    # ACCURACY COMPARISON
    # ============================

    st.header("Accuracy Comparison")

    comparison = pd.DataFrame({
        "Stage": [
            "Baseline",
            "Hospital1 Initial",
            "Hospital1 Final",
            "Hospital2 Initial",
            "Hospital2 Final",
            "Global Model (Hospital1)",
            "Global Model (Hospital2)"
        ],
        "Accuracy": [
            baseline["accuracy"],
            h1["initial"]["accuracy"],
            h1["final_full"]["accuracy"],
            h2["initial"]["accuracy"],
            h2["final_full"]["accuracy"],
            g1["accuracy"],
            g2["accuracy"]
        ]
    })

    st.dataframe(comparison)

    fig4 = px.bar(
        comparison,
        x="Stage",
        y="Accuracy",
        color="Stage",
        title="Accuracy Comparison Across Federated Learning Stages"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ============================
    # FEDERATED ARCHITECTURE
    # ============================

    st.header("Federated Architecture (flow)")

    labels = ["Hospital 1 (local training)", "Hospital 2 (local training)", "Central Server (aggregation)",
              "Global model → hospitals"]
    # Sankey nodes indexes
    label_idx = {l: i for i, l in enumerate(labels)}
    # create flows:
    # hospitals -> central, central -> hospitals
    source = [label_idx["Hospital 1 (local training)"], label_idx["Hospital 2 (local training)"],
              label_idx["Central Server (aggregation)"], label_idx["Central Server (aggregation)"]]
    target = [label_idx["Central Server (aggregation)"], label_idx["Central Server (aggregation)"],
              label_idx["Global model → hospitals"], label_idx["Global model → hospitals"]]
    # values are arbitrary just for visualization
    value = [1, 1, 1, 1]

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=labels, color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]),
        link=dict(source=source, target=target, value=value)
    )])
    fig_sankey.update_layout(title_text="Federated Learning Data / Weights Flow (Sankey)", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown("---")

    # ============================
    # SHAP EXPLAINABILITY
    # ============================
    st.header("Explainability (SHAP)")

    shap_data = pipeline["shap"]["shap_feature_importance"]

    df_shap = pd.DataFrame({
        "Feature": shap_data.keys(),
        "Importance": shap_data.values()
    }).sort_values("Importance", ascending=False)

    fig5 = px.bar(
        df_shap,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance for Cardiovascular Risk"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.dataframe(df_shap)


else:

    st.info("Click the button to run the Federated Learning Pipeline.")