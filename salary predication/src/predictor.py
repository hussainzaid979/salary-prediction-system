"""
Predictor Module
================
Inference engine delivering single and batch salary predictions, complete with
statistical confidence intervals and formatted currency outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.model import SalaryModelTrainer
from src.config import DEFAULT_MODEL_PATH, EXPERIENCE_COLUMN, SALARY_COLUMN


@dataclass
class PredictionResult:
    """Encapsulates prediction metrics and confidence intervals for a single input."""
    experience_years: float
    predicted_salary: float
    monthly_salary: float
    lower_bound_95: float
    upper_bound_95: float
    equation: str

    @property
    def formatted_annual(self) -> str:
        """Annual salary formatted as USD currency."""
        return f"${self.predicted_salary:,.2f}"

    @property
    def formatted_monthly(self) -> str:
        """Estimated monthly salary formatted as USD currency."""
        return f"${self.monthly_salary:,.2f}"

    @property
    def weekly_salary(self) -> float:
        """Estimated weekly salary (52 weeks)."""
        return self.predicted_salary / 52.0

    @property
    def hourly_salary(self) -> float:
        """Estimated hourly wage (assuming 40 hrs/week, 2,080 hrs/yr)."""
        return self.predicted_salary / 2080.0

    @property
    def seniority_tier(self) -> str:
        """Categorical career tier based on experience."""
        if self.experience_years < 2.0:
            return "Associate / Junior Level"
        elif self.experience_years < 5.0:
            return "Mid-Level Professional"
        elif self.experience_years < 8.0:
            return "Senior Professional"
        elif self.experience_years < 12.0:
            return "Lead / Staff Professional"
        return "Principal / Executive Level"

    @property
    def formatted_weekly(self) -> str:
        """Weekly compensation formatted as USD."""
        return f"${self.weekly_salary:,.2f}"

    @property
    def formatted_hourly(self) -> str:
        """Hourly rate formatted as USD."""
        return f"${self.hourly_salary:,.2f}"

    @property
    def formatted_interval(self) -> str:
        """95% Prediction interval formatted as a range."""
        return f"${self.lower_bound_95:,.2f} – ${self.upper_bound_95:,.2f}"


class SalaryPredictor:
    """
    User-facing predictor orchestrating validation, inference, and uncertainty estimation.
    """

    def __init__(self, model_trainer: Optional[SalaryModelTrainer] = None) -> None:
        """
        Initialize predictor with an in-memory model or prepare to load from disk.

        :param model_trainer: Optional pre-fitted SalaryModelTrainer.
        """
        self.trainer = model_trainer

    def load_model(self, filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
        """Loads a persisted model from disk."""
        self.trainer = SalaryModelTrainer.load_model(filepath)

    def predict_single(self, experience: Union[int, float, str]) -> PredictionResult:
        """
        Validates input and predicts salary with statistical 95% prediction interval.

        :param experience: Years of experience (e.g. 3.5).
        :return: Structured PredictionResult dataclass.
        """
        if self.trainer is None or not self.trainer.is_fitted:
            raise RuntimeError("Predictor does not have an active fitted model.")

        # Sanitize and validate input
        exp_clean = DataLoader.validate_experience_input(experience)

        # Reshape to (1, 1) matrix for scikit-learn
        X_input = np.array([[exp_clean]])
        pred_salary = float(self.trainer.predict(X_input)[0])

        # Monthly approximation
        monthly_salary = pred_salary / 12.0

        # Calculate 95% Prediction Interval:
        # Standard Error of Prediction = RSE * sqrt(1 + 1/n + (x0 - x_bar)^2 / sum_sq_diff_x)
        # Margin = t_critical (approx 2.0 - 2.2 for small n) * SE
        summary = self.trainer.training_summary
        rse = self.trainer.metrics.get("residual_std_error", 0.0)
        n = summary.get("n_train_samples", 10)
        x_bar = summary.get("x_train_mean", 5.5)
        sum_sq = summary.get("sum_sq_diff_x", 82.5)

        if rse > 0.0 and sum_sq > 0.0:
            # 95% critical value approximation (t-dist with df ~ n-2)
            t_crit = 2.10 if n > 15 else 2.30
            se_pred = rse * np.sqrt(1.0 + (1.0 / n) + (((exp_clean - x_bar) ** 2) / sum_sq))
            margin = t_crit * se_pred
            lower_bound = max(0.0, pred_salary - margin)
            upper_bound = pred_salary + margin
        else:
            # Exact deterministic fit (e.g. sample dataset with R² = 1.0)
            lower_bound = pred_salary
            upper_bound = pred_salary

        return PredictionResult(
            experience_years=exp_clean,
            predicted_salary=pred_salary,
            monthly_salary=monthly_salary,
            lower_bound_95=lower_bound,
            upper_bound_95=upper_bound,
            equation=self.trainer.get_equation_string(),
        )

    def predict_batch(self, experiences: Union[List[float], np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Performs batch inference for a collection of experience figures.

        :param experiences: Iterable of numeric experience values.
        :return: DataFrame of formatted predictions.
        """
        results = []
        for exp in experiences:
            try:
                res = self.predict_single(exp)
                results.append({
                    EXPERIENCE_COLUMN: res.experience_years,
                    SALARY_COLUMN: round(res.predicted_salary, 2),
                    "Formatted Annual": res.formatted_annual,
                    "Formatted Monthly": res.formatted_monthly,
                    "95% Prediction Range": res.formatted_interval,
                })
            except Exception as e:
                results.append({
                    EXPERIENCE_COLUMN: exp,
                    SALARY_COLUMN: np.nan,
                    "Formatted Annual": f"Error: {e}",
                    "Formatted Monthly": "N/A",
                    "95% Prediction Range": "N/A",
                })

        return pd.DataFrame(results)
