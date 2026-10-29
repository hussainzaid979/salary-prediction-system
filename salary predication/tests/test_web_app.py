"""
Web Application Unit & Integration Tests
========================================
Verifies HTTP endpoints, JSON serialization, and error handling for the Flask web application.
"""

import unittest
from app import app


class TestWebApplication(unittest.TestCase):
    """Test suite covering the Flask web application and REST endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize Flask test client."""
        cls.client = app.test_client()

    def test_01_index_page(self) -> None:
        """Verify the main single-page web dashboard renders with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Salary Prediction System", response.data)
        self.assertIn(b"Predict Salary", response.data)

    def test_02_predict_endpoint_valid(self) -> None:
        """Verify POST /api/predict calculates precise salary and returns structured JSON."""
        response = self.client.post("/api/predict", json={"experience": 4.5})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["experience"], 4.5)
        self.assertEqual(data["predicted_salary"], 85000.0)
        self.assertEqual(data["formatted_annual"], "$85,000.00")
        self.assertEqual(data["formatted_monthly"], "$7,083.33")
        self.assertEqual(data["seniority_tier"], "Mid-Level Professional")

    def test_03_predict_endpoint_invalid_input(self) -> None:
        """Verify input validation catches negative numbers and invalid data types."""
        # Negative experience
        response_neg = self.client.post("/api/predict", json={"experience": -3.0})
        self.assertEqual(response_neg.status_code, 422)
        self.assertFalse(response_neg.get_json()["success"])

        # Non-numeric input
        response_str = self.client.post("/api/predict", json={"experience": "expert"})
        self.assertEqual(response_str.status_code, 422)

        # Missing payload
        response_empty = self.client.post("/api/predict", json={})
        self.assertEqual(response_empty.status_code, 400)

    def test_04_dataset_endpoint(self) -> None:
        """Verify GET /api/dataset returns dataset records and regression line coordinates."""
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["record_count"], 10)
        self.assertEqual(len(data["data_points"]), 10)
        self.assertGreater(len(data["trend_line"]), 5)

    def test_05_batch_endpoint(self) -> None:
        """Verify POST /api/batch computes multi-milestone predictions."""
        response = self.client.post("/api/batch", json={"experiences": [1.0, 5.0, 10.0]})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["salary"], 50000.0)
        self.assertEqual(data["results"][1]["salary"], 90000.0)
        self.assertEqual(data["results"][2]["salary"], 140000.0)

    def test_06_model_info_and_retrain(self) -> None:
        """Verify GET /api/model_info and POST /api/retrain functionality."""
        info = self.client.get("/api/model_info").get_json()
        self.assertTrue(info["success"])
        self.assertEqual(info["model_type"], "linear")

        # Retrain on market dataset
        retrain_res = self.client.post("/api/retrain", json={"dataset": "market", "model_type": "linear"})
        self.assertEqual(retrain_res.status_code, 200)
        retrain_data = retrain_res.get_json()
        self.assertTrue(retrain_data["success"])

        # Reset back to default dataset
        self.client.post("/api/retrain", json={"dataset": "default", "model_type": "linear"})


if __name__ == "__main__":
    unittest.main()
