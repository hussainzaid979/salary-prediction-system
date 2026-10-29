"""
Salary Prediction Web Application
=================================
Flask web server powering the localhost browser interface for
the Salary Prediction Based on Experience System.
"""

import os
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_DATA_PATH,
    MARKET_DATA_PATH,
    DEFAULT_MODEL_PATH,
    FIGURE_PATH,
    WEB_FIGURE_PATH,
    EXPERIENCE_COLUMN,
    SALARY_COLUMN,
)
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import SalaryModelTrainer
from src.predictor import SalaryPredictor
from src.visualizer import SalaryVisualizer

# Initialize Flask application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "salary-prediction-ai-secret-key-2026"


class WebServiceState:
    """Encapsulates active dataset, trained model, and visualization state."""

    def __init__(self, dataset_path: Path = DEFAULT_DATA_PATH) -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset_id = "default" if self.dataset_path == DEFAULT_DATA_PATH else "market"
        self.model_type = "linear"
        self.poly_degree = 2

        self.df: pd.DataFrame = pd.DataFrame()
        self.trainer: SalaryModelTrainer = SalaryModelTrainer(model_type=self.model_type)
        self.predictor: SalaryPredictor = SalaryPredictor(self.trainer)

        # Initial bootstrap
        self.load_and_train()

    def load_and_train(self, dataset_path: Path = None, model_type: str = "linear", degree: int = 2) -> None:
        """Loads dataset, trains model, and updates visual diagnostics."""
        if dataset_path:
            self.dataset_path = Path(dataset_path)
            self.dataset_id = "default" if self.dataset_path == DEFAULT_DATA_PATH else "market"

        self.model_type = model_type.lower()
        self.poly_degree = degree

        loader = DataLoader(self.dataset_path)
        self.df = loader.load_data()

        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(self.df, split=len(self.df) >= 8)

        self.trainer = SalaryModelTrainer(model_type=self.model_type, degree=self.poly_degree)
        self.trainer.fit(X_train, y_train, X_test, y_test)
        self.trainer.save_model(DEFAULT_MODEL_PATH)

        self.predictor = SalaryPredictor(self.trainer)

        # Generate diagnostic figure for web consumption
        visualizer = SalaryVisualizer(self.trainer, self.df)
        visualizer.generate_dashboard(WEB_FIGURE_PATH, highlight_points=[3.5, 7.5])
        visualizer.generate_dashboard(FIGURE_PATH, highlight_points=[3.5, 7.5])


# Global service instance
service = WebServiceState()


@app.route("/")
def index():
    """Serves the primary interactive single-page application."""
    summary_stats = DataLoader.get_summary_statistics(service.df)
    return render_template(
        "index.html",
        dataset_name=service.dataset_path.name,
        dataset_id=service.dataset_id,
        record_count=len(service.df),
        model_type=service.model_type,
        equation=service.trainer.get_equation_string(),
        metrics=service.trainer.metrics,
        summary_stats=summary_stats,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Computes real-time salary prediction for a given experience input.
    Payload: {"experience": 4.5}
    """
    data = request.get_json(force=True, silent=True) or {}
    exp_val = data.get("experience")

    if exp_val is None:
        return jsonify({"success": False, "error": "Experience value is required."}), 400

    try:
        result = service.predictor.predict_single(exp_val)
        return jsonify({
            "success": True,
            "experience": result.experience_years,
            "predicted_salary": round(result.predicted_salary, 2),
            "formatted_annual": result.formatted_annual,
            "formatted_monthly": result.formatted_monthly,
            "formatted_weekly": result.formatted_weekly,
            "formatted_hourly": result.formatted_hourly,
            "formatted_interval": result.formatted_interval,
            "seniority_tier": result.seniority_tier,
            "lower_bound": round(result.lower_bound_95, 2),
            "upper_bound": round(result.upper_bound_95, 2),
            "equation": result.equation,
        })
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 422
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction error: {str(e)}"}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    Computes batch predictions for an array of experience inputs.
    Payload: {"experiences": [1, 2.5, 5, 8.5]}
    """
    data = request.get_json(force=True, silent=True) or {}
    experiences = data.get("experiences", [])

    if not isinstance(experiences, list) or len(experiences) == 0:
        return jsonify({"success": False, "error": "Please provide a non-empty array of experience values."}), 400

    batch_df = service.predictor.predict_batch(experiences)
    records = []
    for _, row in batch_df.iterrows():
        records.append({
            "experience": row[EXPERIENCE_COLUMN],
            "salary": row[SALARY_COLUMN],
            "formatted_annual": row["Formatted Annual"],
            "formatted_monthly": row["Formatted Monthly"],
            "interval": row["95% Prediction Range"],
        })

    return jsonify({"success": True, "results": records})


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    """Returns the records of the active dataset for tabular and chart rendering."""
    records = []
    for _, row in service.df.iterrows():
        records.append({
            "experience": float(row[EXPERIENCE_COLUMN]),
            "salary": float(row[SALARY_COLUMN]),
        })

    # Also compute linear trend line points for client-side plotting
    x_min = float(service.df[EXPERIENCE_COLUMN].min())
    x_max = float(service.df[EXPERIENCE_COLUMN].max())
    x_points = np.linspace(max(0, x_min - 0.5), x_max + 1.0, 30)
    y_preds = service.trainer.predict(x_points.reshape(-1, 1))

    trend_line = [
        {"experience": round(float(x), 2), "salary": round(float(y), 2)}
        for x, y in zip(x_points, y_preds)
    ]

    return jsonify({
        "success": True,
        "dataset_name": service.dataset_path.name,
        "dataset_id": service.dataset_id,
        "record_count": len(service.df),
        "data_points": records,
        "trend_line": trend_line,
    })


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """Returns active model metadata, parameters, and diagnostic metrics."""
    return jsonify({
        "success": True,
        "model_type": service.model_type,
        "degree": service.poly_degree,
        "equation": service.trainer.get_equation_string(),
        "metrics": service.trainer.metrics,
        "dataset_name": service.dataset_path.name,
        "record_count": len(service.df),
    })


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """
    Retrains the system with a designated model type and dataset.
    Payload: {"dataset": "default"|"market", "model_type": "linear"|"polynomial", "degree": 2}
    """
    data = request.get_json(force=True, silent=True) or {}
    dataset_choice = data.get("dataset", "default")
    model_type = data.get("model_type", "linear")
    degree = int(data.get("degree", 2))

    target_path = DEFAULT_DATA_PATH if dataset_choice == "default" else MARKET_DATA_PATH

    try:
        service.load_and_train(dataset_path=target_path, model_type=model_type, degree=degree)
        return jsonify({
            "success": True,
            "message": f"Successfully retrained {model_type.upper()} model on {target_path.name}",
            "equation": service.trainer.get_equation_string(),
            "metrics": service.trainer.metrics,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/plot", methods=["GET"])
def api_plot():
    """Serves the generated high-resolution diagnostic plot."""
    if WEB_FIGURE_PATH.exists():
        return send_file(WEB_FIGURE_PATH, mimetype="image/png")
    elif FIGURE_PATH.exists():
        return send_file(FIGURE_PATH, mimetype="image/png")
    return jsonify({"error": "Plot not yet generated"}), 404


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Launches the localhost web server."""
    print("\n" + "=" * 70)
    print(f"🚀 Salary Prediction System Web Server running at:")
    print(f"👉 Local URL: http://{host}:{port}")
    print("=" * 70 + "\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(host="127.0.0.1", port=5000, debug=False)
