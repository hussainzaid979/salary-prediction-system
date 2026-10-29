"""
Data Loader Module
==================
Handles secure ingestion, validation, and schema verification for salary datasets.
"""

from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from src.config import (
    EXPERIENCE_COLUMN,
    SALARY_COLUMN,
    MIN_EXPERIENCE_YEARS,
    MAX_EXPERIENCE_YEARS,
    DEFAULT_DATA_PATH,
)


class DataValidationError(Exception):
    """Custom exception raised when dataset schema or contents are invalid."""
    pass


class DataLoader:
    """
    Ingests, validates, and summarizes salary and experience data.
    """

    def __init__(self, filepath: Union[str, Path] = DEFAULT_DATA_PATH) -> None:
        """
        Initialize DataLoader with a target filepath.

        :param filepath: Path to CSV dataset.
        """
        self.filepath = Path(filepath)

    def load_data(self) -> pd.DataFrame:
        """
        Loads and validates dataset from the specified CSV file.

        :return: Cleaned and verified pandas DataFrame.
        :raises FileNotFoundError: If the CSV file does not exist.
        :raises DataValidationError: If schema, datatypes, or value ranges are invalid.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.filepath}")

        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            raise DataValidationError(f"Failed to read CSV file: {e}")

        # Check for empty dataframe
        if df.empty:
            raise DataValidationError("Dataset is empty. At least 2 observations are required.")

        # Strip whitespace from column headers
        df.columns = df.columns.str.strip()

        # Validate required columns
        required_cols = [EXPERIENCE_COLUMN, SALARY_COLUMN]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}. Expected columns: {required_cols}"
            )

        # Check for missing values
        null_counts = df[required_cols].isnull().sum().to_dict()
        if any(cnt > 0 for cnt in null_counts.values()):
            raise DataValidationError(f"Dataset contains null values: {null_counts}")

        # Validate numeric types
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    raise DataValidationError(f"Column '{col}' must contain valid numeric values.")

        # Logical boundary checks (non-negative values)
        if (df[EXPERIENCE_COLUMN] < 0).any():
            raise DataValidationError("Experience values cannot be negative.")
        if (df[SALARY_COLUMN] <= 0).any():
            raise DataValidationError("Salary values must be positive and greater than zero.")

        # Sort dataset by experience for clean sequential modeling
        df = df.sort_values(by=EXPERIENCE_COLUMN).reset_index(drop=True)

        return df

    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Computes descriptive statistical metrics for dataset features.

        :param df: Input DataFrame with verified schema.
        :return: Dictionary containing statistical metrics for each column.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for col in [EXPERIENCE_COLUMN, SALARY_COLUMN]:
            series = df[col]
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            stats[col] = {
                "count": float(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "25%": q25,
                "median": float(series.median()),
                "75%": q75,
                "max": float(series.max()),
                "iqr": q75 - q25,
            }
        return stats

    @staticmethod
    def validate_experience_input(value: Union[int, float, str]) -> float:
        """
        Sanitizes and validates a single experience input from the user or API.

        :param value: Input experience in years.
        :return: Validated float value.
        :raises ValueError: If input cannot be converted or is outside reasonable bounds.
        """
        try:
            exp_float = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Input '{value}' is not a valid number.")

        if np.isnan(exp_float) or np.isinf(exp_float):
            raise ValueError("Experience input cannot be NaN or Infinite.")

        if exp_float < MIN_EXPERIENCE_YEARS:
            raise ValueError(
                f"Experience cannot be negative (min: {MIN_EXPERIENCE_YEARS} years)."
            )

        if exp_float > MAX_EXPERIENCE_YEARS:
            raise ValueError(
                f"Experience of {exp_float} years exceeds realistic career span "
                f"(max: {MAX_EXPERIENCE_YEARS} years)."
            )

        return exp_float
