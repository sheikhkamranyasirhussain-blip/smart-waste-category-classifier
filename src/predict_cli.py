"""
Interactive CLI Predictor for Smart Waste Category Classifier
Author: Sheikh Kamran Yasir Hussain (ID: 123626 | Batch: B1 | 2026)
"""

import os
import joblib
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "waste_classifier_dt.pkl")

    if not os.path.exists(model_path):
        print("Model not found. Please run: python src/train_model.py")
        return

    model = joblib.load(model_path)
    print("=== Smart Waste Category Classifier (CLI) ===")
    print("Materials: Plastic, Metal, Paper, Cardboard, Glass, Organic, Electronic, Chemical, Multi-layer Plastic, Ceramic")

    while True:
        item_type = input("\nEnter Item Name (or 'q' to quit): ").strip()
        if item_type.lower() == "q":
            break
        material = input("Enter Material: ").strip()
        try:
            weight = float(input("Enter Weight (grams): ").strip())
        except ValueError:
            print("Invalid number. Please enter weight in grams.")
            continue
        recyclable = input("Is it recyclable? (Yes/No): ").strip().capitalize()

        df_in = pd.DataFrame(
            [
                {
                    "item_type": item_type,
                    "material": material,
                    "weight_grams": weight,
                    "is_recyclable": recyclable,
                }
            ]
        )
        pred = model.predict(df_in)[0]
        conf = max(model.predict_proba(df_in)[0]) * 100
        print(f"--> Recommended Bin: [{pred}] (Confidence: {conf:.1f}%)")


if __name__ == "__main__":
    main()
