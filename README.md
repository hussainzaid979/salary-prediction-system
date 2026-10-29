# Salary Prediction Based on Experience System

An end-to-end machine-learning web application that estimates compensation from years of professional experience. It combines a Flask dashboard, regression models, interactive visualizations, statistical prediction intervals, and a REST API.

## Features

- **Salary predictions:** Estimate annual, monthly, weekly, and hourly compensation.
- **Regression models:** Train Simple Linear Regression or Degree-2 Polynomial Regression.
- **Two datasets:** Switch between a 10-record canonical benchmark and a 30-record market dataset.
- **Interactive dashboard:** Explore predictions, career milestones, dataset statistics, and regression plots.
- **Prediction intervals:** Calculate 95% statistical prediction intervals using residual standard error and Student's *t* critical values.
- **Model evaluation:** Review \(R^2\), adjusted \(R^2\), MAE, RMSE, MAPE, and cross-validation results.
- **Diagnostics:** Generate a four-panel Matplotlib regression analysis figure.
- **REST API and CLI:** Use HTTP endpoints or run predictions and training tasks from the terminal.
- **Automated tests:** Includes 13 unit and integration tests.

## Technology Stack

- **Backend:** Python 3.10+, Flask
- **Machine learning:** scikit-learn
- **Data processing:** Pandas, NumPy
- **Model persistence:** Joblib
- **Visualization:** Matplotlib, Chart.js
- **Frontend:** HTML, Jinja2, CSS, JavaScript
- **Testing:** Python `unittest`

## Project Structure

```text
salary predication/
├── data/
│   ├── salary_data.csv
│   └── salary_data_market.csv
├── models/
│   └── salary_model.joblib
├── reports/
│   └── figures/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   ├── predictor.py
│   └── visualizer.py
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── images/
├── templates/
│   └── index.html
├── tests/
│   ├── test_pipeline.py
│   └── test_web_app.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

**Requirements:** Python 3.10 or newer.

1. Clone or download the project and open a terminal in the project directory.
2. Create and activate a virtual environment (recommended).

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

Start the Flask web application:

```bash
python app.py
```

Alternatively, launch it through the project launcher:

```bash
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Command-Line Usage

Make a single prediction:

```bash
python main.py --predict 4.5
```

Predict multiple experience milestones:

```bash
python main.py --batch "1, 2.5, 5, 8.5, 10"
```

Start the interactive terminal menu:

```bash
python main.py --cli
```

Generate the diagnostic figure:

```bash
python main.py --visualize
```

## REST API

The application exposes the following JSON endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Predict compensation for one experience value |
| `POST` | `/api/batch` | Predict compensation for multiple experience values |
| `GET` | `/api/dataset` | Retrieve dataset points and trend-line coordinates |
| `POST` | `/api/retrain` | Retrain using a selected dataset and model |
| `GET` | `/api/plot` | Retrieve the diagnostic PNG figure |

Example request:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"experience": 4.5}'
```

The documented canonical dataset predicts salary using:

\[
\text{Salary} = 40{,}000 + (10{,}000 \times \text{Experience})
\]

For example, 4.5 years of experience corresponds to a predicted annual salary of **$85,000** in this benchmark dataset.

## Run Tests

Run the automated unit and integration tests:

```bash
python -m unittest discover -s tests -v
```

The project documentation reports 13 tests covering data validation, preprocessing, model training and persistence, prediction, and web API behavior.

## Datasets and Interpretation

- `data/salary_data.csv`: 10-record canonical benchmark dataset with an exact linear relationship.
- `data/salary_data_market.csv`: 30-record dataset with market variation and non-linear seniority changes.

The canonical dataset has zero residual error by construction, so its 95% prediction interval collapses to the predicted value. Use the market dataset to examine intervals that reflect residual variation.

**Note:** Predictions depend on the supplied datasets and fitted model. They are estimates, not guarantees of an individual's market compensation.

## Troubleshooting

- **Port 5000 is occupied:** The original documentation suggests trying `python main.py --port 5050`, then opening `http://127.0.0.1:5050`.
- **Adding salary records:** Append rows to the relevant CSV using the expected experience and salary columns, then retrain the model in the dashboard.

## License

The source README describes the project as an advanced data-science and machine-learning engineering system but does not specify a license. Add a license file and update this section if you intend to distribute the project under a particular license.
