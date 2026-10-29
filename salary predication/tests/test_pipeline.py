"""
Pipeline Unit & Integration Tests
=================================
Verifies data ingestion integrity, model training accuracy,
inference precision, and model serialization.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from src.data_loader import DataLoader, DataValidationError
from src.preprocessor import DataPreprocessor
from src.model import SalaryModelTrainer
from src.predictor import SalaryPredictor


class TestSalaryPredictionPipeline(unittest.TestCase):
    """Test suite covering the end-to-end salary prediction pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test instances and common data."""
        cls.data_loader = DataLoader(DEFAULT_DATA_PATH)
        cls.df = cls.data_loader.load_data()

    def test_01_dataset_schema_and_values(self) -> None:
        """Verify the sample database is correctly parsed with 10 rows and correct schema."""
        self.assertEqual(len(self.df), 10, "Default dataset must contain 10 rows.")
        self.assertIn("Experience (Years)", self.df.columns)
        self.assertIn("Salary (USD)", self.df.columns)
        self.assertEqual(self.df["Experience (Years)"].min(), 1.0)
        self.assertEqual(self.df["Experience (Years)"].max(), 10.0)
        self.assertEqual(self.df["Salary (USD)"].min(), 50000.0)
        self.assertEqual(self.df["Salary (USD)"].max(), 140000.0)

    def test_02_data_validation_boundaries(self) -> None:
        """Verify input boundary checks (non-negative, realistic career span)."""
        # Valid input
        valid_val = DataLoader.validate_experience_input(5.5)
        self.assertAlmostEqual(valid_val, 5.5)

        # Negative experience rejected
        with self.assertRaises(ValueError):
            DataLoader.validate_experience_input(-2.0)

        # Impossible career span (> 50 years) rejected
        with self.assertRaises(ValueError):
            DataLoader.validate_experience_input(75.0)

        # Non-numeric string rejected
        with self.assertRaises(ValueError):
            DataLoader.validate_experience_input("senior-dev")

    def test_03_preprocessor_formatting(self) -> None:
        """Verify feature extraction dimensions and shapes."""
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        X, y, _, _ = preprocessor.prepare_data(self.df, split=False)

        self.assertEqual(X.ndim, 2, "Feature matrix X must be 2-dimensional.")
        self.assertEqual(X.shape, (10, 1))
        self.assertEqual(y.ndim, 1, "Target vector y must be 1-dimensional.")
        self.assertEqual(len(y), 10)

    def test_04_linear_model_training(self) -> None:
        """
        Verify that Linear Regression perfectly fits the sample dataset:
        Salary = 40,000 + 10,000 * Experience (R² = 1.0).
        """
        preprocessor = DataPreprocessor()
        X, y, _, _ = preprocessor.prepare_data(self.df, split=False)

        trainer = SalaryModelTrainer(model_type="linear")
        metrics = trainer.fit(X, y)

        # In the provided dataset, the relationship is perfectly linear
        self.assertAlmostEqual(metrics["r2_score"], 1.0, places=4)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=2)
        self.assertAlmostEqual(trainer.get_intercept(), 40000.0, places=2)
        self.assertAlmostEqual(trainer.get_coefficients()[0], 10000.0, places=2)

    def test_05_prediction_inference(self) -> None:
        """Verify precise predictions for custom experience levels."""
        preprocessor = DataPreprocessor()
        X, y, _, _ = preprocessor.prepare_data(self.df, split=False)

        trainer = SalaryModelTrainer(model_type="linear")
        trainer.fit(X, y)

        predictor = SalaryPredictor(trainer)

        # Test at 1.0 year
        res_1 = predictor.predict_single(1.0)
        self.assertAlmostEqual(res_1.predicted_salary, 50000.0, places=2)
        self.assertEqual(res_1.formatted_annual, "$50,000.00")

        # Test at 4.5 years (mid-point interpolation: 40,000 + 10,000*4.5 = 85,000)
        res_4_5 = predictor.predict_single(4.5)
        self.assertAlmostEqual(res_4_5.predicted_salary, 85000.0, places=2)
        self.assertAlmostEqual(res_4_5.monthly_salary, 85000.0 / 12.0, places=2)

        # Test at 12 years (extrapolation: 40,000 + 10,000*12 = 160,000)
        res_12 = predictor.predict_single(12.0)
        self.assertAlmostEqual(res_12.predicted_salary, 160000.0, places=2)

    def test_06_batch_prediction(self) -> None:
        """Verify batch inference pipeline produces structured table."""
        preprocessor = DataPreprocessor()
        X, y, _, _ = preprocessor.prepare_data(self.df, split=False)

        trainer = SalaryModelTrainer(model_type="linear")
        trainer.fit(X, y)

        predictor = SalaryPredictor(trainer)
        batch_df = predictor.predict_batch([2.0, 5.0, 7.5])

        self.assertEqual(len(batch_df), 3)
        self.assertListEqual(
            list(batch_df["Salary (USD)"]),
            [60000.0, 90000.0, 115000.0]
        )

    def test_07_model_persistence(self) -> None:
        """Verify saving and reloading trained model from disk."""
        preprocessor = DataPreprocessor()
        X, y, _, _ = preprocessor.prepare_data(self.df, split=False)

        trainer = SalaryModelTrainer(model_type="linear")
        trainer.fit(X, y)

        test_model_file = Path("models/test_temp_model.joblib")
        trainer.save_model(test_model_file)
        self.assertTrue(test_model_file.exists())

        loaded_trainer = SalaryModelTrainer.load_model(test_model_file)
        self.assertTrue(loaded_trainer.is_fitted)
        self.assertAlmostEqual(loaded_trainer.get_intercept(), 40000.0, places=2)

        # Clean up temporary test file
        if test_model_file.exists():
            test_model_file.unlink()


if __name__ == "__main__":
    unittest.main()
