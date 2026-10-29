"""
Visualizer Module
=================
Generates modern, publication-grade analytical and diagnostic plots
using Matplotlib for salary and experience regression analysis.
"""

from pathlib import Path
from typing import Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_PATH, EXPERIENCE_COLUMN, SALARY_COLUMN
from src.model import SalaryModelTrainer


class SalaryVisualizer:
    """
    Renders diagnostic regression dashboards and visual charts.
    """

    def __init__(self, trainer: SalaryModelTrainer, df: pd.DataFrame) -> None:
        """
        Initialize visualizer with a trained model and reference dataset.

        :param trainer: Fitted SalaryModelTrainer.
        :param df: Cleaned DataFrame with experience and salary columns.
        """
        self.trainer = trainer
        self.df = df

    def generate_dashboard(
        self,
        save_path: Union[str, Path] = FIGURE_PATH,
        highlight_points: Optional[List[float]] = None,
        show_plot: bool = False,
    ) -> Path:
        """
        Builds and saves a 4-panel regression diagnostic dashboard.

        :param save_path: Destination image path (PNG).
        :param highlight_points: Optional list of experience values to highlight on plot.
        :param show_plot: Whether to call plt.show() interactively.
        :return: Path to the generated figure file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Style settings
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
        fig.patch.set_facecolor("#f8f9fa")

        # Color palette
        c_primary = "#2563eb"    # Modern Blue
        c_accent = "#dc2626"     # Crimson Red
        c_secondary = "#059669"  # Emerald Green
        c_purple = "#7c3aed"     # Violet

        # Extract data
        X = self.df[[EXPERIENCE_COLUMN]].values
        y = self.df[SALARY_COLUMN].values
        y_pred = self.trainer.predict(X)
        residuals = y - y_pred

        # -------------------------------------------------------------
        # 1. Main Regression Fit Plot (Top-Left)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#ffffff")
        ax1.scatter(X, y, color=c_primary, s=80, alpha=0.85, edgecolors="black", label="Actual Data Points", zorder=4)

        # Generate smooth curve for regression line
        x_min, x_max = float(np.min(X)), float(np.max(X))
        padding = max(1.0, (x_max - x_min) * 0.1)
        x_smooth = np.linspace(max(0, x_min - padding), x_max + padding, 200).reshape(-1, 1)
        y_smooth = self.trainer.predict(x_smooth)
        ax1.plot(x_smooth, y_smooth, color=c_accent, linewidth=2.5, label="Fitted Regression Line", zorder=3)

        # Highlight custom prediction points if provided
        if highlight_points:
            for pt in highlight_points:
                pred_val = self.trainer.predict(np.array([[pt]]))[0]
                ax1.scatter([pt], [pred_val], color="#f59e0b", s=180, marker="*", edgecolor="black", zorder=5, label=f"Prediction ({pt} yrs: ${pred_val:,.0f})")
                ax1.vlines(x=pt, ymin=0, ymax=pred_val, color="#f59e0b", linestyle=":", alpha=0.7)
                ax1.hlines(y=pred_val, xmin=0, xmax=pt, color="#f59e0b", linestyle=":", alpha=0.7)

        ax1.set_title("Salary vs. Experience Regression Line", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel("Years of Experience", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Salary (USD)", fontsize=11, fontweight="bold")
        ax1.yaxis.set_major_formatter("${x:,.0f}")
        ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

        # Equation annotation box
        eq_text = self.trainer.get_equation_string()
        ax1.text(
            0.05, 0.05,
            f"Equation:\n{eq_text}\nR² Score: {self.trainer.metrics.get('r2_score', 0):.4f}",
            transform=ax1.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#93c5fd", alpha=0.9),
            verticalalignment="bottom",
        )

        # -------------------------------------------------------------
        # 2. Residuals vs Predicted Plot (Top-Right)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#ffffff")
        ax2.scatter(y_pred, residuals, color=c_purple, s=70, alpha=0.8, edgecolors="black", zorder=4)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.5, zorder=3)

        # Draw vertical residual stems
        for p, r in zip(y_pred, residuals):
            ax2.vlines(p, 0, r, color="#c4b5fd", linestyle="-", linewidth=1.2, zorder=2)

        ax2.set_title("Residual Plot (Error Homoscedasticity)", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel("Predicted Salary (USD)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Residuals [Actual - Predicted] (USD)", fontsize=11, fontweight="bold")
        ax2.xaxis.set_major_formatter("${x:,.0f}")
        ax2.yaxis.set_major_formatter("${x:,.0f}")

        # Metrics text box
        mae_val = self.trainer.metrics.get("mae", 0.0)
        rmse_val = self.trainer.metrics.get("rmse", 0.0)
        ax2.text(
            0.05, 0.90,
            f"MAE:  ${mae_val:,.2f}\nRMSE: ${rmse_val:,.2f}",
            transform=ax2.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3e8ff", edgecolor="#d8b4fe", alpha=0.9),
            verticalalignment="top",
        )

        # -------------------------------------------------------------
        # 3. Actual vs. Predicted Parity Plot (Bottom-Left)
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        ax3.set_facecolor("#ffffff")
        ax3.scatter(y, y_pred, color=c_secondary, s=80, alpha=0.85, edgecolors="black", zorder=4)

        # 45-degree ideal fit line
        min_val = min(float(np.min(y)), float(np.min(y_pred))) * 0.95
        max_val = max(float(np.max(y)), float(np.max(y_pred))) * 1.05
        ax3.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1.5, label="Ideal 1:1 Parity", zorder=3)

        ax3.set_xlim(min_val, max_val)
        ax3.set_ylim(min_val, max_val)
        ax3.set_title("Actual vs. Predicted Parity Plot", fontsize=13, fontweight="bold", pad=10)
        ax3.set_xlabel("Actual Salary (USD)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Predicted Salary (USD)", fontsize=11, fontweight="bold")
        ax3.xaxis.set_major_formatter("${x:,.0f}")
        ax3.yaxis.set_major_formatter("${x:,.0f}")
        ax3.legend(loc="upper left", frameon=True, facecolor="white")

        # -------------------------------------------------------------
        # 4. Career Experience Tier Salary Averages (Bottom-Right)
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        ax4.set_facecolor("#ffffff")

        # Create career tiers
        tier_bins = [0, 2, 5, 8, 100]
        tier_labels = ["Junior (0-2 yrs)", "Mid-Level (3-5 yrs)", "Senior (6-8 yrs)", "Lead/Principal (9+ yrs)"]
        temp_df = self.df.copy()
        temp_df["Tier"] = pd.cut(temp_df[EXPERIENCE_COLUMN], bins=tier_bins, labels=tier_labels, right=True)
        tier_stats = temp_df.groupby("Tier", observed=False)[SALARY_COLUMN].mean().reset_index()

        bars = ax4.bar(
            tier_stats["Tier"].astype(str),
            tier_stats[SALARY_COLUMN],
            color=["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"],
            edgecolor="black",
            alpha=0.85,
            width=0.55,
            zorder=3
        )

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax4.annotate(
                    f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold"
                )

        ax4.set_title("Average Salary by Career Seniority Tier", fontsize=13, fontweight="bold", pad=10)
        ax4.set_ylabel("Average Salary (USD)", fontsize=11, fontweight="bold")
        ax4.set_xticks(range(len(tier_labels)))
        ax4.set_xticklabels(tier_labels, rotation=15, ha="right", fontsize=9.5)
        ax4.yaxis.set_major_formatter("${x:,.0f}")

        # Main overall title
        fig.suptitle(
            "Salary Prediction Based on Experience System — Diagnostic Dashboard",
            fontsize=16,
            fontweight="bold",
            color="#1e293b",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close(fig)
        return save_path
