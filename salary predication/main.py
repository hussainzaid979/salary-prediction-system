"""
Salary Prediction Based on Experience System
============================================
Main entry-point offering both a stylish interactive terminal console
and flexible command-line arguments for automated inference.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

from src.config import (
    DEFAULT_DATA_PATH,
    MARKET_DATA_PATH,
    DEFAULT_MODEL_PATH,
    FIGURE_PATH,
    EXPERIENCE_COLUMN,
    SALARY_COLUMN,
    UIColors as C,
)
from src.data_loader import DataLoader, DataValidationError
from src.preprocessor import DataPreprocessor
from src.model import SalaryModelTrainer
from src.predictor import SalaryPredictor
from src.visualizer import SalaryVisualizer


def print_banner() -> None:
    """Displays an expressive, stylish ASCII art banner for the system."""
    banner = f"""
{C.CYAN}{C.BOLD}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗ █████╗ ██╗      █████╗ ██████╗ ██╗   ██╗                         ║
║   ██╔════╝██╔══██╗██║     ██╔══██╗██╔══██╗╚██╗ ██╔╝                         ║
║   ███████╗███████║██║     ███████║██████╔╝ ╚████╔╝                          ║
║   ╚════██║██╔══██║██║     ██╔══██║██╔══██╗  ╚██╔╝                           ║
║   ███████║██║  ██║███████╗██║  ██║██║  ██║   ██║                            ║
║   ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                            ║
║                PREDICTION BASED ON EXPERIENCE SYSTEM                          ║
║                                                                               ║
║   {C.GREEN}● Predictive Engine: Scikit-Learn | Data: Pandas & NumPy | Plots: Matplotlib{C.CYAN} ║
╚═══════════════════════════════════════════════════════════════════════════════╝{C.RESET}
"""
    print(banner)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats list of headers and string rows into a clean ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = "| " + " | ".join([f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]) + " |"

    formatted_rows = []
    for row in rows:
        formatted_row = "| " + " | ".join([f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))]) + " |"
        formatted_rows.append(formatted_row)

    return f"{sep}\n{header_str}\n{sep}\n" + "\n".join(formatted_rows) + f"\n{sep}"


class SalaryApplication:
    """
    Coordinates dataset loading, model training, analytical visualization,
    and user interactions within the terminal environment.
    """

    def __init__(self, dataset_path: Path = DEFAULT_DATA_PATH) -> None:
        """Initialize the application state."""
        self.dataset_path = Path(dataset_path)
        self.data_loader = DataLoader(self.dataset_path)
        self.df: Optional[pd.DataFrame] = None
        self.trainer: Optional[SalaryModelTrainer] = None
        self.predictor: Optional[SalaryPredictor] = None

        # Auto-load dataset on initialization
        self.load_dataset()

    def load_dataset(self) -> bool:
        """Loads and prepares the active dataset."""
        try:
            self.df = self.data_loader.load_data()
            return True
        except Exception as e:
            print(f"{C.RED}[!] Error loading dataset: {e}{C.RESET}")
            return False

    def train_model(self, model_type: str = "linear", degree: int = 2) -> bool:
        """
        Executes model training pipeline with feature extraction and evaluation.
        """
        if self.df is None:
            if not self.load_dataset():
                return False

        print(f"\n{C.CYAN}--- Training Regression Model ({model_type.upper()}) ---{C.RESET}")
        preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(self.df, split=len(self.df) >= 8)

        self.trainer = SalaryModelTrainer(model_type=model_type, degree=degree)
        metrics = self.trainer.fit(X_train, y_train, X_test, y_test)
        self.trainer.save_model(DEFAULT_MODEL_PATH)

        self.predictor = SalaryPredictor(self.trainer)

        # Display Metrics
        print(f"{C.GREEN}✔ Model successfully trained and saved to: {DEFAULT_MODEL_PATH}{C.RESET}\n")
        print(f"{C.BOLD}Mathematical Equation:{C.RESET}")
        print(f"  {C.YELLOW}{self.trainer.get_equation_string()}{C.RESET}\n")

        headers = ["Metric", "Value", "Interpretation"]
        r2_val = metrics['r2_score']
        mae_val = metrics['mae']
        rmse_val = metrics['rmse']
        cv_mean = metrics.get('cv_r2_mean', r2_val)

        rows = [
            ["R² Score", f"{r2_val:.4f}", f"{r2_val*100:.1f}% variance explained by experience"],
            ["Adjusted R²", f"{metrics['adjusted_r2']:.4f}", "Penalized for model complexity"],
            ["MAE", f"${mae_val:,.2f}", "Mean Absolute Error across observations"],
            ["RMSE", f"${rmse_val:,.2f}", "Root Mean Squared Error (penalizes outliers)"],
            ["MAPE", f"{metrics['mape_percent']:.2f}%", "Mean Absolute Percentage Error"],
            ["5-Fold CV R²", f"{cv_mean:.4f}", "Cross-Validation stability metric"],
        ]

        print(format_table(headers, rows))
        return True

    def predict_salary_interactive(self) -> None:
        """Prompts user for experience input and displays prediction results."""
        if self.predictor is None:
            # Attempt to auto-train if not already trained
            if not self.train_model():
                return

        print(f"\n{C.CYAN}--- Interactive Salary Predictor ---{C.RESET}")
        print(f"{C.DIM}Enter years of experience (e.g. 1, 3.5, 7.2) or 'q' to return to menu.{C.RESET}\n")

        while True:
            raw_input = input(f"{C.BOLD}Enter Experience (Years): {C.RESET}").strip()
            if raw_input.lower() in ["q", "quit", "exit"]:
                break

            try:
                result = self.predictor.predict_single(raw_input)
                print(f"\n{C.GREEN}{C.BOLD}════════════════ PREDICTION SUMMARY ════════════════{C.RESET}")
                print(f"  Experience Level:        {C.BOLD}{result.experience_years:g} Years{C.RESET}")
                print(f"  Predicted Annual Salary: {C.GREEN}{C.BOLD}{result.formatted_annual}{C.RESET}")
                print(f"  Estimated Monthly Comp:  {C.CYAN}{result.formatted_monthly}{C.RESET}")
                print(f"  95% Prediction Interval: {C.YELLOW}{result.formatted_interval}{C.RESET}")
                print(f"  Underlying Formula:      {C.DIM}{result.equation}{C.RESET}")
                print(f"{C.GREEN}════════════════════════════════════════════════════{C.RESET}\n")
            except ValueError as ve:
                print(f"{C.RED}[!] Input Error: {ve}{C.RESET}\n")
            except Exception as e:
                print(f"{C.RED}[!] Prediction failed: {e}{C.RESET}\n")

    def run_batch_prediction(self, inputs: Optional[List[float]] = None) -> None:
        """Runs inference across multiple experience inputs."""
        if self.predictor is None:
            if not self.train_model():
                return

        if inputs is None:
            raw = input(f"\n{C.BOLD}Enter comma-separated experience years (e.g. 1.5, 3.0, 6.5, 9.0): {C.RESET}").strip()
            if not raw:
                return
            try:
                inputs = [float(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                print(f"{C.RED}[!] Invalid numeric list.{C.RESET}")
                return

        batch_df = self.predictor.predict_batch(inputs)
        print(f"\n{C.CYAN}--- Batch Prediction Results ---{C.RESET}")
        headers = ["Exp (Yrs)", "Predicted (USD)", "Monthly (USD)", "95% Prediction Range"]
        rows = []
        for _, row in batch_df.iterrows():
            rows.append([
                f"{row[EXPERIENCE_COLUMN]:g}",
                row["Formatted Annual"],
                row["Formatted Monthly"],
                row["95% Prediction Range"],
            ])
        print(format_table(headers, rows))

    def generate_visualizations(self) -> None:
        """Renders and saves comprehensive Matplotlib charts."""
        if self.trainer is None:
            if not self.train_model():
                return

        print(f"\n{C.CYAN}--- Generating Diagnostic Visualizations ---{C.RESET}")
        visualizer = SalaryVisualizer(self.trainer, self.df)
        saved_file = visualizer.generate_dashboard(FIGURE_PATH, highlight_points=[3.5, 7.5])
        print(f"{C.GREEN}✔ Diagnostic charts successfully generated and saved to:{C.RESET}")
        print(f"  {C.BOLD}{saved_file.resolve()}{C.RESET}\n")

    def show_dataset_summary(self) -> None:
        """Displays formatted database overview and summary statistics."""
        if self.df is None:
            if not self.load_dataset():
                return

        print(f"\n{C.CYAN}--- Active Database: {self.dataset_path.name} ({len(self.df)} records) ---{C.RESET}")
        headers = ["Index", EXPERIENCE_COLUMN, SALARY_COLUMN]
        rows = []
        for idx, row in self.df.iterrows():
            rows.append([str(idx + 1), f"{row[EXPERIENCE_COLUMN]:g} yrs", f"${row[SALARY_COLUMN]:,.0f}"])
        print(format_table(headers, rows))

        stats = DataLoader.get_summary_statistics(self.df)
        print(f"\n{C.BOLD}Statistical Summary:{C.RESET}")
        stat_headers = ["Metric", EXPERIENCE_COLUMN, SALARY_COLUMN]
        stat_metrics = ["count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr"]
        stat_rows = []
        for m in stat_metrics:
            exp_val = stats[EXPERIENCE_COLUMN][m]
            sal_val = stats[SALARY_COLUMN][m]
            if m in ["mean", "std", "min", "25%", "median", "75%", "max", "iqr"]:
                stat_rows.append([m.upper(), f"{exp_val:.2f}", f"${sal_val:,.2f}"])
            else:
                stat_rows.append([m.upper(), f"{exp_val:.0f}", f"{sal_val:.0f}"])
        print(format_table(stat_headers, stat_rows))

    def switch_dataset(self) -> None:
        """Toggles between standard sample dataset and market dataset."""
        print(f"\n{C.CYAN}--- Switch Database ---{C.RESET}")
        print(f"  1. Default Sample Database ({DEFAULT_DATA_PATH.name} - 10 records, perfect linear fit)")
        print(f"  2. Realistic Market Database ({MARKET_DATA_PATH.name} - 30 records with natural variance)")
        choice = input(f"{C.BOLD}Select dataset (1 or 2): {C.RESET}").strip()

        if choice == "1":
            self.dataset_path = DEFAULT_DATA_PATH
        elif choice == "2":
            self.dataset_path = MARKET_DATA_PATH
        else:
            print(f"{C.RED}[!] Invalid choice.{C.RESET}")
            return

        self.data_loader = DataLoader(self.dataset_path)
        if self.load_dataset():
            print(f"{C.GREEN}✔ Switched to: {self.dataset_path.name}{C.RESET}")
            # Retrain model on new dataset
            self.train_model()

    def run_menu(self) -> None:
        """Runs the main interactive terminal application loop."""
        print_banner()
        # Train initial model silently
        self.train_model()

        while True:
            print(f"\n{C.BOLD}{C.BLUE}═════════════════════ MAIN MENU ═════════════════════{C.RESET}")
            print(f"  {C.BOLD}[1]{C.RESET} 🔮 Predict Salary (Single Input)")
            print(f"  {C.BOLD}[2]{C.RESET} 📑 Batch Prediction (Multiple Inputs)")
            print(f"  {C.BOLD}[3]{C.RESET} 📊 View Database & Statistical Summary")
            print(f"  {C.BOLD}[4]{C.RESET} 🧠 Retrain Model (Linear or Polynomial)")
            print(f"  {C.BOLD}[5]{C.RESET} 📈 Generate & Save Visualizations")
            print(f"  {C.BOLD}[6]{C.RESET} 🔄 Switch Dataset (Sample vs. Market)")
            print(f"  {C.BOLD}[7]{C.RESET} 🧪 Run Pipeline Unit Tests")
            print(f"  {C.BOLD}[0]{C.RESET} 🚪 Exit")
            print(f"{C.BLUE}═════════════════════════════════════════════════════{C.RESET}")

            choice = input(f"{C.BOLD}Choose an option (0-7): {C.RESET}").strip()

            if choice == "1":
                self.predict_salary_interactive()
            elif choice == "2":
                self.run_batch_prediction()
            elif choice == "3":
                self.show_dataset_summary()
            elif choice == "4":
                m_type = input("Choose model type ['linear' (default) or 'polynomial']: ").strip().lower() or "linear"
                deg = 2
                if m_type == "polynomial":
                    deg_input = input("Enter polynomial degree [default 2]: ").strip()
                    deg = int(deg_input) if deg_input.isdigit() else 2
                self.train_model(model_type=m_type, degree=deg)
            elif choice == "5":
                self.generate_visualizations()
            elif choice == "6":
                self.switch_dataset()
            elif choice == "7":
                print(f"\n{C.CYAN}--- Executing Pipeline Unit Tests ---{C.RESET}")
                import unittest
                suite = unittest.defaultTestLoader.discover("tests")
                runner = unittest.TextTestRunner(verbosity=2)
                runner.run(suite)
            elif choice == "0":
                print(f"\n{C.GREEN}Thank you for using the Salary Prediction System. Goodbye!{C.RESET}\n")
                sys.exit(0)
            else:
                print(f"{C.RED}[!] Invalid choice. Please enter a number between 0 and 7.{C.RESET}")


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line arguments for scripted/CLI execution."""
    parser = argparse.ArgumentParser(
        description="Salary Prediction Based on Experience System — Automated ML Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--predict",
        type=float,
        metavar="YEARS",
        help="Instantly predict salary for a specific number of years of experience.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        metavar="YEARS_LIST",
        help="Comma-separated list of experience values for batch inference (e.g. '1.5,3,5.5,8').",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train regression model, compute evaluation metrics, and save model artifact.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save publication-grade diagnostic regression charts.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Custom CSV dataset filepath (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch the legacy interactive terminal console instead of the website.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for localhost web server (default: 5000).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for web server (default: 127.0.0.1).",
    )
    return parser.parse_args()


def main() -> None:
    """Main program entry point."""
    args = parse_arguments()
    dataset_path = Path(args.dataset)
    app = SalaryApplication(dataset_path=dataset_path)

    # If CLI flags were provided, run in non-interactive batch/scripted mode
    if args.predict is not None:
        print_banner()
        app.train_model()
        res = app.predictor.predict_single(args.predict)
        print(f"\n{C.GREEN}{C.BOLD}════════════════ PREDICTION RESULT ════════════════{C.RESET}")
        print(f"  Experience:       {res.experience_years:g} Years")
        print(f"  Annual Salary:    {C.GREEN}{res.formatted_annual}{C.RESET}")
        print(f"  Monthly Salary:   {C.CYAN}{res.formatted_monthly}{C.RESET}")
        print(f"  95% Interval:     {C.YELLOW}{res.formatted_interval}{C.RESET}")
        print(f"  Formula:          {res.equation}")
        print(f"{C.GREEN}═══════════════════════════════════════════════════{C.RESET}\n")
        return

    if args.batch is not None:
        print_banner()
        app.train_model()
        values = [float(v.strip()) for v in args.batch.split(",") if v.strip()]
        app.run_batch_prediction(values)
        return

    if args.train:
        print_banner()
        app.train_model()
        return

    if args.visualize:
        print_banner()
        app.train_model()
        app.generate_visualizations()
        return

    if args.cli:
        # User explicitly requested CLI interactive menu
        app.run_menu()
        return

    # Default: Launch stylish localhost website
    print_banner()
    print(f"\n{C.CYAN}🌐 Launching Salary Prediction System Web Application on Localhost...{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}👉 Open in your web browser: http://{args.host}:{args.port}{C.RESET}\n")
    from app import run_server
    run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
