import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)

import matplotlib.pyplot as plt
from fpdf import FPDF

from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()

# ===================== 1) DATA GENERATION =====================

def generate_fraud_data(
    n_samples: int = 2000,
    fraud_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic fraud data with identities and timestamps."""
    if fraud_ratio <= 0 or fraud_ratio >= 1:
        raise ValueError("fraud_ratio must be between 0 and 1 (e.g., 0.1).")

    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # identities
    n_users = max(50, n_samples // 20)
    n_merchants = max(20, n_samples // 50)
    user_ids = rng.integers(1, n_users + 1, size=n_samples)
    merchant_ids = rng.integers(1, n_merchants + 1, size=n_samples)

    # timestamps over 7 days
    start_ts = np.datetime64("2024-01-01T00:00")
    minutes_offsets = rng.integers(0, 7 * 24 * 60, size=n_samples)
    timestamps = start_ts + minutes_offsets.astype("timedelta64[m]")

    # base features
    legit_amount = rng.normal(500, 150, n_legit)
    fraud_amount = rng.normal(3500, 900, n_fraud)

    legit_hour = rng.integers(7, 23, n_legit)
    fraud_hour = rng.integers(0, 24, n_fraud)

    legit_international = rng.binomial(1, 0.05, n_legit)
    fraud_international = rng.binomial(1, 0.4, n_fraud)

    # country risk: 0 = low risk, 1 = medium, 2 = high
    legit_country_risk = rng.choice([0, 1], size=n_legit, p=[0.7, 0.3])
    fraud_country_risk = rng.choice([1, 2], size=n_fraud, p=[0.3, 0.7])

    amount = np.concatenate([legit_amount, fraud_amount])
    hour = np.concatenate([legit_hour, fraud_hour])
    is_international = np.concatenate([legit_international, fraud_international])
    country_risk = np.concatenate([legit_country_risk, fraud_country_risk])

    labels = np.concatenate(
        [np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]
    )

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "merchant_id": merchant_ids,
            "timestamp": timestamps,
            "amount": amount,
            "hour": hour,
            "is_international": is_international,
            "country_risk": country_risk,
            "is_fraud": labels,
        }
    )

    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

# ===================== 2) NOISE + RULE ENGINE =====================

def inject_noise(
    df: pd.DataFrame,
    missing_frac: float = 0.02,
    outlier_frac: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inject missing values and outliers to mimic real-world dirty data."""
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    n_rows = len(df_noisy)

    # missing in amount
    n_missing = int(n_rows * missing_frac)
    if n_missing > 0:
        idx = rng.choice(df_noisy.index, size=n_missing, replace=False)
        df_noisy.loc[idx, "amount"] = np.nan

    # outliers in amount
    n_outliers = int(n_rows * outlier_frac)
    if n_outliers > 0:
        idx = rng.choice(df_noisy.index, size=n_outliers, replace=False)
        df_noisy.loc[idx, "amount"] *= rng.uniform(3, 8, size=n_outliers)

    return df_noisy


def add_rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple rule-based risk score."""
    df = df.copy()
    score = np.zeros(len(df))

    score += (df["amount"] > 2500).astype(int)
    score += (df["hour"] < 6).astype(int)
    score += df["is_international"] * 2
    score += (df["country_risk"] == 2).astype(int)

    df["rule_score"] = score
    return df

# ===================== 3) CROSS-VALIDATION + TUNING =====================

def cross_validate_random_forest(X, y, random_state=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    prec_scores = cross_val_score(model, X, y, cv=skf, scoring="precision")
    rec_scores = cross_val_score(model, X, y, cv=skf, scoring="recall")

    return {
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "cv_precision_mean": float(prec_scores.mean()),
        "cv_precision_std": float(prec_scores.std()),
        "cv_recall_mean": float(rec_scores.mean()),
        "cv_recall_std": float(rec_scores.std()),
    }


def tune_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["auto", "sqrt", 0.5],
        "min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="recall",
        n_jobs=-1,
        cv=3,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    table = Table(title="Model Comparison", box=box.SIMPLE)
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")

    best_model = None
    best_recall = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        table.add_row(name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}")
        if rec > best_recall:
            best_recall = rec
            best_model = (name, model)

    console.print(table)
    return best_model

# ===================== 4) VISUALS + REAL-TIME =====================

def show_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "amount",
        "hour",
        "is_international",
        "country_risk",
        "rule_score",
        "is_fraud",
    ]
    corr = df[numeric_cols].corr()

    plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Feature Correlation Heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.close()


def simulate_realtime(model, df_features: pd.DataFrame, n_events: int = 10):
    console.rule("[bold magenta]Real-Time Fraud Simulation[/bold magenta]")
    sample = df_features.sample(n=min(n_events, len(df_features)), random_state=42)
    for _, row in sample.iterrows():
        # keep feature names to avoid warning
        x = row.to_frame().T
        prob = model.predict_proba(x)[0, 1]
        risk_level = "LOW"
        if prob > 0.8:
            risk_level = "HIGH"
        elif prob > 0.5:
            risk_level = "MEDIUM"

        console.print(
            f"[cyan]Tx[/cyan] amount={row['amount']:.2f}, hour={int(row['hour'])}, "
            f"intl={int(row['is_international'])} -> "
            f"fraud_prob={prob:.2f} ([bold]{risk_level} RISK[/bold])"
        )
        time.sleep(0.2)

# ===================== 5) PDF REPORT =====================

def generate_pdf_report(
    filename: str,
    total_rows: int,
    fraud_count: int,
    cv_metrics: dict,
    tuned_metrics: dict,
):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Fraud Pattern Generator - Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Total Transactions: {total_rows}", ln=True)
    pdf.cell(0, 8, f"Fraudulent Transactions: {fraud_count}", ln=True)
    pdf.cell(0, 8, f"Fraud Ratio: {fraud_count/total_rows:.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Cross-Validation (RandomForest):", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Accuracy mean: {cv_metrics['cv_accuracy_mean']:.3f}", ln=True)
    pdf.cell(0, 8, f"Recall   mean: {cv_metrics['cv_recall_mean']:.3f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Tuned RandomForest (Hold-out Test):", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Accuracy: {tuned_metrics['accuracy']:.3f}", ln=True)
    pdf.cell(0, 8, f"Precision: {tuned_metrics['precision']:.3f}", ln=True)
    pdf.cell(0, 8, f"Recall: {tuned_metrics['recall']:.3f}", ln=True)

    pdf.output(filename)

# ===================== 6) MAIN PIPELINE =====================

def main(n_samples=2000, fraud_ratio=0.15, random_state=42):
    console.rule("[bold green]Fraud Pattern Generator – Full System[/bold green]")
    console.print(
        f"[cyan]samples={n_samples}, fraud_ratio={fraud_ratio}, seed={random_state}[/cyan]"
    )

    # STEP 1: Data generation + noise + rule score
    console.rule("[bold green]STEP 1: Generate Synthetic Data[/bold green]")
    for _ in track(range(60), description="Generating data..."):
        pass

    df = generate_fraud_data(n_samples, fraud_ratio, random_state)
    df = inject_noise(df, missing_frac=0.02, outlier_frac=0.01, random_state=random_state)
    df = add_rule_score(df)

    df.to_csv("synthetic_fraud_data_full.csv", index=False)

    total_rows = len(df)
    fraud_count = int(df["is_fraud"].sum())
    legit_count = total_rows - fraud_count

    summary = Table(title="Dataset Summary", box=box.DOUBLE_EDGE)
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Total Rows", str(total_rows))
    summary.add_row("Fraud Count", str(fraud_count))
    summary.add_row("Non-Fraud Count", str(legit_count))
    summary.add_row("Fraud Ratio", f"{fraud_count/total_rows:.2f}")
    console.print(summary)

    # bar chart (only show, not save)
    plt.figure()
    plt.bar(["Non-Fraud", "Fraud"], [legit_count, fraud_count])
    plt.title("Fraud vs Non-Fraud")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    plt.close()

    # STEP 2: Correlation heatmap
    console.rule("[bold blue]STEP 2: Correlation Analysis[/bold blue]")
    show_correlation_heatmap(df)

    # Prepare data for ML
    features = df[
        ["amount", "hour", "is_international", "country_risk", "rule_score"]
    ].copy()
    features["amount"] = features["amount"].fillna(features["amount"].median())
    target = df["is_fraud"]

    # STEP 3: Cross-validation
    console.rule("[bold blue]STEP 3: Cross-Validation[/bold blue]")
    cv = cross_validate_random_forest(features, target, random_state)
    console.print(f"[yellow]CV Accuracy mean: {cv['cv_accuracy_mean']:.3f}[/yellow]")
    console.print(f"[yellow]CV Recall   mean: {cv['cv_recall_mean']:.3f}[/yellow]")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, stratify=target, random_state=random_state
    )

    # STEP 4: Hyperparameter tuning
    console.rule("[bold yellow]STEP 4: Hyperparameter Tuning (RandomForest)[/bold yellow]")
    best_rf, best_params = tune_random_forest(X_train, y_train, random_state)
    console.print(f"[green]Best RF params: {best_params}[/green]")

    # STEP 5: Model comparison
    console.rule("[bold yellow]STEP 5: Model Comparison[/bold yellow]")
    (best_name, best_model) = compare_models(X_train, X_test, y_train, y_test)
    console.print(f"[bold green]Best model by recall: {best_name}[/bold green]")

    # STEP 6: Evaluate tuned RF
    console.rule("[bold yellow]STEP 6: Evaluate Tuned RandomForest[/bold yellow]")
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    console.print(f"[green]Tuned RF Accuracy: {acc:.3f}[/green]")
    console.print(f"[green]Tuned RF Precision: {prec:.3f}[/green]")
    console.print(f"[green]Tuned RF Recall: {rec:.3f}[/green]")

    tuned_metrics = {"accuracy": acc, "precision": prec, "recall": rec}

    cm = confusion_matrix(y_test, y_pred)
    console.print("[bold]Confusion Matrix:[/bold]")
    console.print(cm)
    console.print("[bold]Classification Report:[/bold]")
    console.print(classification_report(y_test, y_pred, digits=3))

    # feature importance plot (only show)
    importances = best_rf.feature_importances_
    names = features.columns
    sorted_idx = np.argsort(importances)

    plt.figure()
    plt.barh(names[sorted_idx], importances[sorted_idx])
    plt.title("Feature Importances (Tuned RandomForest)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # STEP 7: IsolationForest anomaly detection
    console.rule("[bold red]STEP 7: Unsupervised Anomaly Detection (IsolationForest)[/bold red]")
    iso = IsolationForest(contamination=fraud_ratio, random_state=random_state)
    iso.fit(features)
    df["anomaly_score"] = -iso.decision_function(features)

    df_sorted = df.sort_values("anomaly_score", ascending=False)

    alert = Table(title="🚨 Top 10 Fraud Alerts (IsolationForest) 🚨", box=box.ROUNDED)
    alert.add_column("Amount")
    alert.add_column("Hour")
    alert.add_column("Intl")
    alert.add_column("True Fraud")
    alert.add_column("Anomaly Score")

    for _, row in df_sorted.head(10).iterrows():
        alert.add_row(
            f"{row['amount']:.2f}",
            str(int(row["hour"])),
            str(int(row["is_international"])),
            str(int(row["is_fraud"])),
            f"{row['anomaly_score']:.3f}",
        )
    console.print(alert)

    plt.figure()
    plt.hist(df["anomaly_score"], bins=30)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    plt.close()

    # STEP 8: Real-time simulation with tuned RF
    simulate_realtime(best_rf, features)

    # STEP 9: PDF report
    console.rule("[bold magenta]STEP 9: Generating PDF Report[/bold magenta]")
    generate_pdf_report(
        "fraud_report.pdf",
        total_rows=total_rows,
        fraud_count=fraud_count,
        cv_metrics=cv,
        tuned_metrics=tuned_metrics,
    )
    console.print("[green]PDF report saved as fraud_report.pdf[/green]")

    console.rule("[bold green]✔ FULL RUN COMPLETE[/bold green]")
    console.print("[bold green]All modules executed successfully.[/bold green]")

    # close any leftover figures cleanly
    plt.close("all")

# ===================== 7) CLI ARGS =====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--fraud_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.samples, args.fraud_ratio, args.seed)
