"""
Model Module
============
Constructs, trains, evaluates, and serializes machine learning regression models
for salary prediction based on professional experience.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score, KFold
import joblib

from src.config import DEFAULT_MODEL_PATH


class SalaryModelTrainer:
    """
    Supervises training, evaluation, comparison, and persistence of regression models.
    """

    def __init__(self, model_type: str = "linear", degree: int = 2) -> None:
        """
        Initialize trainer with selected regression model type.

        :param model_type: 'linear' for Simple Linear Regression, 'polynomial' for Polynomial Regression.
        :param degree: Polynomial degree if model_type is 'polynomial'.
        """
        self.model_type = model_type.lower()
        self.degree = degree

        if self.model_type == "polynomial":
            self.model: Union[LinearRegression, Pipeline] = Pipeline([
                ("poly_features", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("linear_regression", LinearRegression())
            ])
        else:
            self.model = LinearRegression()

        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self.training_summary: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fits the regression model to training data and computes evaluation metrics.

        :param X_train: Training features matrix (n_samples, 1).
        :param y_train: Training target array (n_samples,).
        :param X_test: Optional test feature matrix.
        :param y_test: Optional test target array.
        :return: Dictionary of evaluation metrics.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Use test set for metrics if available; otherwise evaluate on training set
        eval_X = X_test if X_test is not None and len(X_test) > 0 else X_train
        eval_y = y_test if y_test is not None and len(y_test) > 0 else y_train

        y_pred = self.model.predict(eval_X)

        # Basic regression metrics
        r2 = float(r2_score(eval_y, y_pred))
        mae = float(mean_absolute_error(eval_y, y_pred))
        mse = float(mean_squared_error(eval_y, y_pred))
        rmse = float(np.sqrt(mse))
        mape = float(mean_absolute_percentage_error(eval_y, y_pred)) * 100.0

        # Adjusted R²: 1 - [(1 - R²)(n - 1) / (n - p - 1)]
        n = len(eval_y)
        p = 1 if self.model_type == "linear" else self.degree
        if n > p + 1:
            adj_r2 = float(1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1)))
        else:
            adj_r2 = r2

        # Residual Standard Error (RSE) for inference confidence bands
        residuals = eval_y - y_pred
        degrees_of_freedom = max(1, n - (p + 1))
        residual_std_error = float(np.sqrt(np.sum(residuals ** 2) / degrees_of_freedom))

        self.metrics = {
            "r2_score": r2,
            "adjusted_r2": adj_r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape_percent": mape,
            "residual_std_error": residual_std_error,
            "sample_size": n,
        }

        # Cross-validation if sample size permits (at least 2 samples per test fold required for valid R²)
        if len(X_train) >= 10:
            n_splits = min(5, len(X_train) // 2)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="r2")
            # Filter out any non-finite scores if edge cases arise
            valid_cv = cv_scores[np.isfinite(cv_scores)]
            if len(valid_cv) > 0:
                self.metrics["cv_r2_mean"] = float(np.mean(valid_cv))
                self.metrics["cv_r2_std"] = float(np.std(valid_cv))
            else:
                self.metrics["cv_r2_mean"] = r2
                self.metrics["cv_r2_std"] = 0.0
        else:
            self.metrics["cv_r2_mean"] = r2
            self.metrics["cv_r2_std"] = 0.0

        # Save training summary details
        self.training_summary = {
            "model_type": self.model_type,
            "degree": self.degree if self.model_type == "polynomial" else 1,
            "intercept": self.get_intercept(),
            "coefficients": self.get_coefficients(),
            "equation": self.get_equation_string(),
            "x_train_mean": float(np.mean(X_train)),
            "sum_sq_diff_x": float(np.sum((X_train - np.mean(X_train)) ** 2)),
            "n_train_samples": len(X_train),
        }

        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for an array of feature inputs.

        :param X: Feature matrix of shape (n_samples, 1).
        :return: Array of predicted salaries.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        return self.model.predict(X)

    def get_intercept(self) -> float:
        """Extracts the base intercept (beta_0)."""
        if not self.is_fitted:
            return 0.0
        if self.model_type == "polynomial":
            return float(self.model.named_steps["linear_regression"].intercept_)
        return float(self.model.intercept_)

    def get_coefficients(self) -> list:
        """Extracts slope coefficients (beta_1, ...)."""
        if not self.is_fitted:
            return []
        if self.model_type == "polynomial":
            return [float(c) for c in self.model.named_steps["linear_regression"].coef_]
        return [float(c) for c in self.model.coef_]

    def get_equation_string(self) -> str:
        """
        Constructs an expressive, human-readable mathematical equation for the fitted model.

        :return: Mathematical formula representation.
        """
        if not self.is_fitted:
            return "Model not trained."

        intercept = self.get_intercept()
        coefs = self.get_coefficients()

        if self.model_type == "linear":
            slope = coefs[0] if len(coefs) > 0 else 0.0
            sign = "+" if slope >= 0 else "-"
            return f"Salary = ${intercept:,.2f} {sign} (${abs(slope):,.2f} × Experience)"

        # Polynomial format
        terms = [f"${intercept:,.2f}"]
        for idx, coef in enumerate(coefs, start=1):
            sign = "+" if coef >= 0 else "-"
            power = f"Experience^{idx}" if idx > 1 else "Experience"
            terms.append(f"{sign} (${abs(coef):,.2f} × {power})")
        return "Salary = " + " ".join(terms)

    def save_model(self, filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
        """
        Serializes and saves the trained model bundle to disk.

        :param filepath: Destination file path.
        :return: Path to saved file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "degree": self.degree,
            "metrics": self.metrics,
            "training_summary": self.training_summary,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load_model(cls, filepath: Union[str, Path] = DEFAULT_MODEL_PATH) -> "SalaryModelTrainer":
        """
        Deserializes a saved model bundle from disk.

        :param filepath: Path to saved model file.
        :return: Initialized and fitted SalaryModelTrainer instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        payload = joblib.load(path)
        trainer = cls(model_type=payload["model_type"], degree=payload.get("degree", 2))
        trainer.model = payload["model"]
        trainer.is_fitted = payload["is_fitted"]
        trainer.metrics = payload["metrics"]
        trainer.training_summary = payload["training_summary"]
        return trainer
