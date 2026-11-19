"""
Automated Unit Tests for Smart Waste Category Classifier
Author: Sheikh Kamran Yasir Hussain (ID: 123626 | Batch: B1 | 2026)
Run: pytest
"""

import os
import unittest
import joblib
import pandas as pd


class TestWasteClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_path = os.path.join(base_dir, "data", "smart_waste_dataset.csv")
        cls.model_path = os.path.join(base_dir, "models", "waste_classifier_dt.pkl")
        cls.df = pd.read_csv(cls.data_path)
        cls.model = joblib.load(cls.model_path)

    def test_dataset_schema_and_size(self):
        expected_cols = ["item_type", "material", "weight_grams", "is_recyclable", "waste_category"]
        self.assertEqual(list(self.df.columns), expected_cols)
        self.assertGreaterEqual(len(self.df), 100)
        self.assertFalse(self.df.isnull().any().any())

    def test_known_waste_predictions(self):
        samples = pd.DataFrame(
            [
                {"item_type": "Aluminum Foil", "material": "Metal", "weight_grams": 18, "is_recyclable": "Yes"},
                {"item_type": "Banana Peel", "material": "Organic", "weight_grams": 100, "is_recyclable": "No"},
                {"item_type": "USB Cable", "material": "Electronic", "weight_grams": 45, "is_recyclable": "Yes"},
                {"item_type": "Chip Packet", "material": "Multi-layer Plastic", "weight_grams": 10, "is_recyclable": "No"},
                {"item_type": "AA Battery", "material": "Metal", "weight_grams": 24, "is_recyclable": "No"},
            ]
        )
        preds = list(self.model.predict(samples))
        expected = [
            "Dry/Recyclable",
            "Wet/Organic",
            "E-Waste",
            "General/Landfill",
            "Hazardous",
        ]
        self.assertEqual(preds, expected)


if __name__ == "__main__":
    unittest.main()
