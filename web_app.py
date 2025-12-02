import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# import your existing engine functions from main.py
from main import generate_fraud_data, inject_noise, add_rule_score


def run_app():
    st.title("Fraud Pattern Generator for Model Training")
    st.markdown(
        "Generate synthetic fraud data, train models, and explore fraud detection patterns."
    )

    # ------------------------------------------------------------------
    # SIDEBAR: user controls
    # ------------------------------------------------------------------
    st.sidebar.header("Simulation Settings")

    samples = st.sidebar.selectbox(
        "Number of transactions",
        [1000, 2000, 5000, 10000],
        index=1,  # default 2000
    )

    scenario = st.sidebar.selectbox(
        "Fraud scenario",
        ["Low fraud (~5%)", "Medium fraud (~15%)", "High fraud (~25%)"],
        index=1,
    )

    model_type = st.sidebar.selectbox(
        "Model type",
        ["RandomForest", "Logistic Regression"],
        index=0,
    )

    iso_contamination = st.sidebar.slider(
        "Anomaly detection sensitivity (IsolationForest contamination)",
        min_value=0.01,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Higher = more transactions flagged as suspicious",
    )

    seed = st.sidebar.number_input(
        "Random seed (for reproducible results)",
        value=42,
        step=1,
    )

    show_corr = st.sidebar.checkbox("Show correlation heatmap (Graphs tab)", value=True)

    # map scenario → fraud_ratio
    if "Low" in scenario:
        fraud_ratio = 0.05
    elif "High" in scenario:
        fraud_ratio = 0.25
    else:
        fraud_ratio = 0.15

    st.info(
        f"Scenario: **{scenario}** → using fraud ratio ≈ **{fraud_ratio:.0%}** "
        f"with **{samples}** transactions."
    )

    # ------------------------------------------------------------------
    # STEP 1: generate data (engine from main.py)
    # ------------------------------------------------------------------
    st.write("### Step 1: Generating synthetic data...")
    df = generate_fraud_data(
        n_samples=samples,
        fraud_ratio=fraud_ratio,
        random_state=seed,
    )
    df = inject_noise(df, missing_frac=0.02, outlier_frac=0.01, random_state=seed)
    df = add_rule_score(df)

    total_rows = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = total_rows - fraud_count

    # prepare features/target once, reused in tabs
    features = df[
        ["amount", "hour", "is_international", "country_risk", "rule_score"]
    ].copy()
    features["amount"] = features["amount"].fillna(features["amount"].median())
    target = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        stratify=target,
        random_state=seed,
    )

    # choose & train model
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # IsolationForest anomaly detection
    iso = IsolationForest(contamination=iso_contamination, random_state=seed)
    iso.fit(features)
    df["anomaly_score"] = -iso.decision_function(features)
    df_sorted = df.sort_values("anomaly_score", ascending=False)

    # ------------------------------------------------------------------
    # TABS LAYOUT
    # ------------------------------------------------------------------
    tab_overview, tab_model, tab_alerts, tab_graphs = st.tabs(
        ["Overview", "Model Performance", "Fraud Alerts", "Graphs"]
    )

    # ------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ------------------------------------------------------------------
    with tab_overview:
        st.subheader("Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total rows", total_rows)
        col2.metric("Fraud rows", fraud_count)
        col3.metric("Non-fraud rows", legit_count)
        col4.metric("Fraud ratio", f"{fraud_count/total_rows:.2%}")

        # download dataset
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download generated dataset (CSV)",
            data=csv_data,
            file_name="synthetic_fraud_data_web.csv",
            mime="text/csv",
        )

        st.write("#### Sample of generated data")
        st.dataframe(df.head(20))

        st.markdown(
            "- `rule_score` combines simple business rules.\n"
            "- `anomaly_score` (later) comes from an unsupervised model."
        )

    # ------------------------------------------------------------------
    # TAB 2: MODEL PERFORMANCE + CUSTOM TESTER
    # ------------------------------------------------------------------
    with tab_model:
        st.subheader(f"Model Performance ({model_type})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")

        st.write("#### Confusion Matrix (numbers)")
        st.write(cm)

        # confusion matrix heatmap
        fig_cm, ax_cm = plt.subplots()
        im_cm = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred: Legit", "Pred: Fraud"], rotation=45, ha="right")
        ax_cm.set_yticklabels(["True: Legit", "True: Fraud"])
        ax_cm.set_title("Confusion Matrix Heatmap")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig_cm.colorbar(im_cm, ax=ax_cm)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

        # feature importances (only for RF)
        if model_type == "RandomForest":
            importances = model.feature_importances_
            names = features.columns
            sorted_idx = np.argsort(importances)

            fig1, ax1 = plt.subplots()
            ax1.barh(names[sorted_idx], importances[sorted_idx])
            ax1.set_title("Feature Importances (RandomForest)")
            fig1.tight_layout()
            st.subheader("Feature Importances")
            st.pyplot(fig1)

        # -------------- Custom Transaction Tester --------------
        st.subheader("Try a Custom Transaction")

        st.markdown(
            "Enter a hypothetical transaction and see how the trained model scores it."
        )

        with st.form("custom_txn_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                amount_input = st.number_input(
                    "Transaction amount",
                    min_value=0.0,
                    value=1500.0,
                    step=100.0,
                )
                hour_input = st.slider(
                    "Hour of day (0–23)",
                    min_value=0,
                    max_value=23,
                    value=10,
                )
            with col_b:
                intl_choice = st.selectbox(
                    "Is international?",
                    ["No", "Yes"],
                )
                country_choice = st.selectbox(
                    "Country risk",
                    ["Low", "Medium", "High"],
                )

            submitted = st.form_submit_button("Score this transaction")

        if submitted:
            is_international_val = 1 if intl_choice == "Yes" else 0
            risk_map = {"Low": 0, "Medium": 1, "High": 2}
            country_risk_val = risk_map[country_choice]

            # compute rule_score with same logic as backend
            rule_score_val = 0
            if amount_input > 2500:
                rule_score_val += 1
            if hour_input < 6:
                rule_score_val += 1
            rule_score_val += 2 * is_international_val
            if country_risk_val == 2:
                rule_score_val += 1

            single_features = pd.DataFrame(
                [
                    {
                        "amount": amount_input,
                        "hour": hour_input,
                        "is_international": is_international_val,
                        "country_risk": country_risk_val,
                        "rule_score": rule_score_val,
                    }
                ]
            )

            prob_fraud = model.predict_proba(single_features)[0, 1]
            if prob_fraud < 0.3:
                risk_level = "LOW"
                color = "green"
            elif prob_fraud < 0.7:
                risk_level = "MEDIUM"
                color = "orange"
            else:
                risk_level = "HIGH"
                color = "red"

            st.markdown(
                f"**Fraud probability:** `{prob_fraud:.2f}` → "
                f"**<span style='color:{color}'>{risk_level} RISK</span>**",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # TAB 3: FRAUD ALERTS (IsolationForest)
    # ------------------------------------------------------------------
    with tab_alerts:
        st.subheader("Top Fraud Alerts (IsolationForest)")

        st.markdown(
            "IsolationForest is an unsupervised model. "
            "It only looks for unusual patterns, not the true labels."
        )

        alert_cols = [
            "amount",
            "hour",
            "is_international",
            "country_risk",
            "rule_score",
            "is_fraud",
            "anomaly_score",
        ]
        st.dataframe(df_sorted[alert_cols].head(10))

        st.info(
            "Compare `rule_score`, `anomaly_score`, and `is_fraud` to see how "
            "rules and ML agree or disagree."
        )

    # ------------------------------------------------------------------
    # TAB 4: GRAPHS (distribution + anomaly + correlation)
    # ------------------------------------------------------------------
    with tab_graphs:
        st.subheader("Class Distribution (Fraud vs Non-Fraud)")
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(["Non-Fraud", "Fraud"], [legit_count, fraud_count])
        ax_bar.set_ylabel("Count")
        fig_bar.tight_layout()
        st.pyplot(fig_bar)

        st.subheader("Anomaly Score Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["anomaly_score"], bins=30)
        ax2.set_title("Anomaly Score Distribution")
        ax2.set_xlabel("Anomaly score (higher = more suspicious)")
        ax2.set_ylabel("Frequency")
        fig2.tight_layout()
        st.pyplot(fig2)

        if show_corr:
            st.subheader("Correlation Heatmap (features vs fraud)")
            numeric_cols = [
                "amount",
                "hour",
                "is_international",
                "country_risk",
                "rule_score",
                "is_fraud",
            ]
            corr = df[numeric_cols].corr()

            fig_corr, ax_corr = plt.subplots()
            im = ax_corr.imshow(corr, interpolation="nearest")
            ax_corr.set_xticks(range(len(numeric_cols)))
            ax_corr.set_yticks(range(len(numeric_cols)))
            ax_corr.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax_corr.set_yticklabels(numeric_cols)
            ax_corr.set_title("Correlation Heatmap")
            fig_corr.colorbar(im, ax=ax_corr)
            fig_corr.tight_layout()
            st.pyplot(fig_corr)

    st.success("Run complete! Change settings in the sidebar to explore more scenarios.")


if __name__ == "__main__":
    run_app()
