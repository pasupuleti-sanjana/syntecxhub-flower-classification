"""
Project 1: Flower Classification (Iris Dataset)
Syntecxhub Machine Learning Internship - Week 2 Task

This script:
1. Loads the Iris dataset
2. Performs basic EDA
3. Visualizes feature pairs
4. Trains Logistic Regression and Decision Tree classifiers
5. Compares accuracy
6. Plots confusion matrices
7. Provides a CLI prediction option for new flower measurements
"""

import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_iris_model.joblib")


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris


def perform_eda(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n--- Dataset Preview ---")
    print(df.head())

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    print("\n--- Species Count ---")
    print(df["species"].value_counts())

    df.describe().to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    df["species"].value_counts().to_csv(os.path.join(OUTPUT_DIR, "species_count.csv"))


def visualize_feature_pairs(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features = [col for col in df.columns if col != "species"]
    species_list = df["species"].unique()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            plt.figure(figsize=(8, 6))
            for species in species_list:
                subset = df[df["species"] == species]
                plt.scatter(subset[features[i]], subset[features[j]], label=species)
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.title(f"{features[i]} vs {features[j]}")
            plt.legend()
            plt.tight_layout()
            filename = f"{features[i].replace(' ', '_').replace('(', '').replace(')', '')}_vs_{features[j].replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
            plt.close()


def train_and_compare_models(iris):
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=3)
    }

    results = []
    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": accuracy})

        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=iris.target_names, cmap="Blues"
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"), dpi=200)
        plt.close()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_accuracy_comparison.csv"), index=False)

    joblib.dump({"model": best_model, "target_names": iris.target_names, "feature_names": iris.feature_names}, MODEL_PATH)
    print("\n--- Final Comparison ---")
    print(results_df)
    print(f"\nBest model saved: {best_name} with accuracy {best_accuracy:.4f}")

    return results_df


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    if not os.path.exists(MODEL_PATH):
        _, iris = load_data()
        train_and_compare_models(iris)

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    target_names = saved["target_names"]

    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

    print("\nPredicted Iris Species:", target_names[prediction])
    if probabilities is not None:
        print("\nPrediction Probabilities:")
        for species, prob in zip(target_names, probabilities):
            print(f"{species}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Iris Flower Classification Project")
    parser.add_argument("--predict", action="store_true", help="Predict species using input measurements")
    parser.add_argument("--sepal_length", type=float, help="Sepal length in cm")
    parser.add_argument("--sepal_width", type=float, help="Sepal width in cm")
    parser.add_argument("--petal_length", type=float, help="Petal length in cm")
    parser.add_argument("--petal_width", type=float, help="Petal width in cm")
    args = parser.parse_args()

    if args.predict:
        required = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
        if any(value is None for value in required):
            parser.error("For prediction, provide --sepal_length --sepal_width --petal_length --petal_width")
        predict_species(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width)
    else:
        df, iris = load_data()
        perform_eda(df)
        visualize_feature_pairs(df)
        train_and_compare_models(iris)
        print("\nProject completed. Check the outputs folder for graphs, CSV files, and saved model.")


if __name__ == "__main__":
    main()
