"""
Preprocessor Module
===================
Handles feature extraction, matrix formatting, and dataset partitioning.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    EXPERIENCE_COLUMN,
    SALARY_COLUMN,
    TEST_SPLIT_RATIO,
    RANDOM_STATE,
)


class DataPreprocessor:
    """
    Transforms raw tabular DataFrames into formatted feature matrices
    and targets for supervised machine learning algorithms.
    """

    def __init__(
        self,
        test_size: float = TEST_SPLIT_RATIO,
        random_state: int = RANDOM_STATE,
        shuffle: bool = True,
    ) -> None:
        """
        Initialize preprocessor configuration.

        :param test_size: Fraction of samples reserved for model evaluation (0.0 to 0.5).
        :param random_state: Seed for reproducible random splits.
        :param shuffle: Whether to shuffle data prior to splitting.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def prepare_data(
        self, df: pd.DataFrame, split: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts feature matrix X and target vector y, optionally splitting into train/test sets.

        :param df: Cleaned input DataFrame.
        :param split: If True, partitions data into train and test sets.
        :return: (X_train, y_train, X_test, y_test) or (X, y, None, None)
        """
        # Feature matrix X must be 2D: shape (n_samples, 1)
        X = df[[EXPERIENCE_COLUMN]].values.astype(np.float64)

        # Target vector y: shape (n_samples,)
        y = df[SALARY_COLUMN].values.astype(np.float64)

        if not split or self.test_size <= 0.0 or len(df) < 5:
            # If dataset is too compact or splitting disabled, train on full set
            return X, y, None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        return X_train, y_train, X_test, y_test
