"""
Smart Waste Category Classifier - Training & Evaluation Pipeline
Author: Sheikh Kamran Yasir Hussain (ID: 123626 | Batch: B1 | 2026)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Set clean academic plotting style
plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", palette="muted")


def load_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def build_pipeline(max_depth: int = 6) -> Pipeline:
    categorical_features = ["item_type", "material", "is_recyclable"]
    numeric_features = ["weight_grams"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=2,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_visualizations(
    df: pd.DataFrame,
    model: Pipeline,
    y_test: pd.Series,
    y_pred: np.ndarray,
    figures_dir: str,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    classes = sorted(df["waste_category"].unique())

    # 1. Dataset Class Distribution Chart
    plt.figure(figsize=(8, 4.5))
    order = df["waste_category"].value_counts().index
    ax = sns.countplot(data=df, x="waste_category", order=order, palette="viridis")
    plt.title("Waste Dataset Class Distribution (N = 150)", fontsize=13, fontweight="bold")
    plt.xlabel("Waste Category", fontsize=11)
    plt.ylabel("Item Count", fontsize=11)
    plt.xticks(rotation=15)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"))
    plt.close()

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        linewidths=0.8,
    )
    plt.title("Decision Tree Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("Actual Category", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    plt.close()

    # 3. Decision Tree Architecture Visualization
    clf = model.named_steps["classifier"]
    prep = model.named_steps["preprocessor"]
    feature_names = prep.get_feature_names_out()

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=4,
    )
    plt.title("Decision Tree Splitting Logic (Top 4 Levels)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "decision_tree_diagram.png"))
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "smart_waste_dataset.csv")
    model_path = os.path.join(base_dir, "models", "waste_classifier_dt.pkl")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    metrics_path = os.path.join(base_dir, "reports", "evaluation_metrics.json")

    df = load_dataset(data_path)
    X = df[["item_type", "material", "weight_grams", "is_recyclable"]]
    y = df["waste_category"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5-Fold Stratified Cross-Validation
    model = build_pipeline(max_depth=6)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Fit on training split & evaluate on hold-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("SMART WASTE CATEGORY CLASSIFIER - MODEL EVALUATION")
    print("=" * 60)
    print(f"Dataset Size       : {len(df)} records ({len(X_train)} train / {len(X_test)} test)")
    print(f"5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print(f"Hold-Out Accuracy  : {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save trained pipeline
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Save visual figures
    save_visualizations(df, model, y_test, y_pred, figures_dir)

    # Save JSON metrics summary
    metrics_summary = {
        "student": "Sheikh Kamran Yasir Hussain",
        "student_id": "123626",
        "batch": "B1",
        "year": 2026,
        "dataset_rows": int(len(df)),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(test_acc), 4),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Export human-readable tree rules
    clf = model.named_steps["classifier"]
    prep = model.named_steps["preprocessor"]
    rules_txt = export_text(clf, feature_names=list(prep.get_feature_names_out()))
    with open(os.path.join(base_dir, "reports", "decision_tree_rules.txt"), "w") as f:
        f.write(rules_txt)

    print(f"Saved model   -> {model_path}")
    print(f"Saved figures -> {figures_dir}/")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
