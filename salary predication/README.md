# 💼 Salary Prediction Based on Experience System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Web Framework](https://img.shields.io/badge/Framework-Flask%203.1-emerald.svg)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Plots-Matplotlib%20%26%20Chart.js-red.svg)](https://matplotlib.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-13%2F13%20Passing-brightgreen.svg)]()
[![Localhost](https://img.shields.io/badge/Localhost-http%3A%2F%2F127.0.0.1%3A5000-purple.svg)](http://127.0.0.1:5000)

An end-to-end Machine Learning web application and predictive analytics system designed to model, analyze, and forecast professional compensation based on career experience. Built with a decoupled modular architecture, this project features an interactive **Localhost Web Dashboard** with real-time dynamic plotting, rigorous econometric confidence intervals, multi-dataset management, and publication-ready diagnostic charts.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Languages & Technologies](#-languages--technologies)
3. [Project Hierarchy & Architecture](#-project-hierarchy--architecture)
4. [Key Features (A to Z)](#-key-features-a-to-z)
5. [Mathematical & Econometric Foundations](#-mathematical--econometric-foundations)
6. [Database Schema & Datasets](#-database-schema--datasets)
7. [Installation & Setup Guide](#-installation--setup-guide)
8. [Localhost Web Application Walkthrough](#-localhost-web-application-walkthrough)
9. [REST API Documentation](#-rest-api-documentation)
10. [Command-Line Interface (CLI) Fallback](#-command-line-interface-cli-fallback)
11. [Testing & Quality Assurance](#-testing--quality-assurance)
12. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🌟 Project Overview

The **Salary Prediction Based on Experience System** bridges statistical modeling and full-stack software development. It enables HR analysts, engineers, recruiters, and professionals to:
- Estimate annual, monthly, weekly, and hourly market compensation.
- Calculate **95% statistical prediction intervals** based on residual standard error ($s_e$).
- Visualize observations alongside fitted regression curves with a real-time reactive target marker.
- Inspect model assumptions via a 4-panel diagnostic dashboard (Linear fit, Error homoscedasticity, $1:1$ parity, and seniority tiers).
- Toggle between a **Canonical Benchmark Database** (10 records, perfect linear progression) and a **Realistic Market Database** (30 records with natural variance).
- Retrain between Simple Linear and Polynomial Regression algorithms directly from the web interface.

---

## 💻 Languages & Technologies

| Layer | Language / Library | Purpose & Details |
| :--- | :--- | :--- |
| **Backend Core** | **Python 3.10+** | High-performance numerical computations, OOP architecture, and data pipelines. |
| **Web Server** | **Flask 3.1** | Lightweight WSGI server, routing, REST API controllers, template rendering. |
| **Machine Learning** | **Scikit-learn 1.9** | Ordinary Least Squares (`LinearRegression`), `PolynomialFeatures`, K-Fold cross-validation, and metrics (`r2_score`, `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`). |
| **Data Processing** | **Pandas 2.2+ & NumPy 2.2+** | DataFrame operations, CSV ingestion, quantile computations, matrix reshaping. |
| **Model Persistence** | **Joblib 1.4+** | Serializes trained model pipelines, weights, intercepts, and training metadata to disk. |
| **Diagnostics** | **Matplotlib 3.11** | High-resolution 300 DPI 4-quadrant diagnostic regression figure generation. |
| **Frontend UI** | **HTML5 & Jinja2** | Semantic layout, template variables, accessible forms, tabs, and responsive grids. |
| **Styling** | **Modern CSS3** | Custom properties (CSS variables), Glassmorphism, Flexbox, responsive CSS Grid, custom range sliders, badge pills. |
| **Client Scripts** | **JavaScript (ES6+)** | Asynchronous Fetch API, debounced input synchronizers, dynamic DOM rendering, tab switching. |
| **Client Plotting** | **Chart.js 4.4** | Real-time reactive scatter & curve canvas chart with live query pin marker. |
| **Unit Testing** | **Python Unittest** | 13 automated unit and integration tests covering pipeline and web endpoints. |

---

## 📂 Project Hierarchy & Architecture

```text
salary predication/
├── data/
│   ├── salary_data.csv                    # Canonical 10-sample database (exact prompt specification)
│   └── salary_data_market.csv             # Extended 30-sample benchmark dataset with realistic market noise
├── models/
│   └── salary_model.joblib                # Serialized model artifact containing fitted weights & metadata
├── reports/
│   └── figures/
│       └── salary_prediction_analysis.png # High-resolution 300 DPI 4-panel analytical diagnostic dashboard
├── src/
│   ├── __init__.py                        # Package initialization and public exports
│   ├── config.py                          # Global configuration: directory paths, schema constants, UI colors
│   ├── data_loader.py                     # Data ingestion, schema validation, boundary sanitization, statistics
│   ├── preprocessor.py                    # Feature extraction, 2D matrix formatting, train-test splitting
│   ├── model.py                           # Linear & Polynomial regression training, metrics, and persistence
│   ├── predictor.py                       # Inference engine, 95% confidence intervals, compensation breakdowns
│   └── visualizer.py                      # Matplotlib 4-quadrant diagnostic figure generator
├── static/
│   ├── css/
│   │   └── style.css                      # Modern CSS design system, responsive grid, card components
│   ├── js/
│   │   └── app.js                         # Dynamic frontend controller, Chart.js sync, debounced fetch calls
│   └── images/
│       └── salary_prediction_analysis.png # Web-accessible copy of diagnostic dashboard
├── templates/
│   └── index.html                         # Responsive Single-Page Application (SPA) web interface
├── tests/
│   ├── __init__.py                        # Test package initialization
│   ├── test_pipeline.py                   # 7 Unit tests verifying data integrity, model fit, and inference
│   └── test_web_app.py                    # 6 Unit tests verifying Flask routes, JSON contracts, and error handling
├── app.py                                 # Primary Flask localhost application server
├── main.py                                # Dual-mode launcher: web application server + CLI query interface
├── requirements.txt                       # Explicit list of production Python dependencies
└── README.md                              # Complete A-to-Z project manual
```

---

## ⚡ Key Features (A to Z)

- **A — Automated Unit Testing**: 13 automated tests (`test_pipeline.py` & `test_web_app.py`) verifying schema, boundary rules, linear convergence ($R^2 = 1.0$), inference precision, and HTTP endpoints.
- **B — Batch Milestone Prediction**: Evaluate multi-year milestones simultaneously (e.g. `1.0, 2.5, 5.0, 7.5, 10.0, 15.0`) in tabular view.
- **C — Chart.js Real-Time Reactive Plot**: Interactive client-side canvas plot showing training points, fitted curve, and a live target marker that moves dynamically with the experience slider.
- **D — Diagnostic 4-Panel Dashboard**: Generates publication-grade 300 DPI Matplotlib figures displaying regression fit, error homoscedasticity residuals, $1:1$ parity, and seniority tier averages.
- **E — Econometric 95% Prediction Intervals**: Employs Student's $t$ critical distribution and Residual Standard Error ($s_e$) to calculate statistically sound upper and lower bounds.
- **F — Flexible Model Selection**: Toggle between Simple Linear Regression and Polynomial Regression (Degree 2) directly from the browser.
- **G — Granular Pay Breakdown**: Outputs Annual, Monthly, Weekly (52 weeks), and Hourly (2,080 working hours) compensation rates.
- **H — Hourly & Weekly Equivalent Wage Engine**: Automates wage calculation for payroll, contractor, and HR budgeting scenarios.
- **I — Interactive Input Synchronizer**: Bi-directional real-time synchronization between the range slider ($0$ to $25$ yrs) and direct numeric input ($0$ to $50$ yrs).
- **J — Joblib Serialization**: Persists fitted weights, intercept, polynomial transformers, and summary metrics into portable `.joblib` artifacts.
- **K — K-Fold Cross-Validation**: Validates generalization capability across multiple partitioned folds to detect overfitting.
- **L — Localhost Web Interface**: Modern, responsive single-page browser application served at `http://127.0.0.1:5000`.
- **M — Multi-Dataset Management**: Switch on-the-fly between the canonical benchmark dataset ($10$ rows) and a real-world market dataset ($30$ rows).
- **N — Numerical Boundary Sanitization**: Validates user inputs against non-numeric entries, negative numbers, infinite values, and career longevity bounds ($>50$ yrs).
- **O — Ordinary Least Squares (OLS)**: Closed-form analytical optimization minimizing the sum of squared residuals.
- **P — Preset Career Milestones**: One-click chips for rapid evaluation of Junior ($1.0$ yr), Associate ($3.0$ yrs), Mid-Level ($5.0$ yrs), Senior ($7.5$ yrs), Lead ($10.0$ yrs), and Principal ($15.0$ yrs).
- **Q — Quick CLI Flag Interface**: Scriptable terminal flags (`--predict`, `--batch`, `--train`, `--visualize`) for automation and headless environments.
- **R — Responsive Design**: Clean Glassmorphic styling with CSS variables, responsive CSS Grid, and Flexbox for desktop and mobile viewports.
- **S — Seniority Tier Classification**: Automatically labels compensation into Associate/Junior, Mid-Level, Senior, Lead/Staff, or Principal/Executive bands.
- **T — Tabular Database Explorer**: In-browser inspection of raw database observations, complete with descriptive statistical metrics (Mean, Std, Median, Min, Max, IQR).
- **U — Unicode & Cross-Platform Console**: Configured for error-free execution across Windows (cp1252/UTF-8), macOS, and Linux.
- **V — Visual Formula Callout**: Displays human-readable mathematical formulas (e.g., `Salary = $40,000.00 + ($10,000.00 × Experience)`).
- **W — WSGI Production Ready**: Standard Flask WSGI application compatible with Gunicorn, Waitress, or Uvicorn.
- **X — X-Y Correlation Analysis**: Calculates variance explanation ($R^2$), adjusted $R^2$, MAE, RMSE, and MAPE.
- **Y — Yearly Increment Extraction**: Isolates slope ($\beta_1$) to quantify exact annual monetary return per year of career tenure.
- **Z — Zero-Reload State Updates**: Powered by asynchronous JavaScript (`fetch`) and debounced REST API communication.

---

## 🧮 Mathematical & Econometric Foundations

### 1. Simple Linear Regression Model
The relationship between experience ($X$) and salary ($Y$) is parameterized as:
$$\hat{Y} = \beta_0 + \beta_1 X + \epsilon$$

Using Ordinary Least Squares (OLS), the objective is to minimize the Sum of Squared Residuals ($\text{SSR}$):
$$\text{SSR}(\beta_0, \beta_1) = \sum_{i=1}^{n} \left(y_i - (\beta_0 + \beta_1 x_i)\right)^2$$

Setting partial derivatives with respect to $\beta_0$ and $\beta_1$ to zero yields the analytical solutions:
$$\beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}$$

#### Canonical Database Solution:
- Mean Experience: $\bar{x} = 5.50\text{ years}$
- Mean Salary: $\bar{y} = \$95,000.00$
- Derived Slope ($\beta_1$): $\$10,000.00 / \text{year}$
- Derived Intercept ($\beta_0$): $\$40,000.00$
- **Predictive Formula**:
$$\text{Salary} = \$40,000.00 + (\$10,000.00 \times \text{Experience})$$

---

### 2. Polynomial Regression (Degree 2)
To model diminishing or accelerating returns in senior careers:
$$\hat{Y} = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon$$
Transformed through feature matrix $\Phi(X) = [1, X, X^2]$ and solved via normal equations:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

### 3. Evaluation & Goodness-of-Fit Metrics

- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
- **Adjusted $R^2$**:
  $$R^2_{\text{adj}} = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$
  where $n$ is sample size and $p$ is the number of predictors.
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \left(\frac{1}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|\right) \times 100\%$$

---

### 4. Statistical 95% Prediction Interval
For arbitrary experience queries $x_0$, the prediction interval accounts for both parameter estimation uncertainty and individual variance:
$$\text{PI}_{95\%} = \hat{y}_0 \pm t_{\alpha/2, \, n-2} \cdot s_e \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}$$
where $s_e = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - 2}}$ is the Residual Standard Error (RSE), and $t_{\alpha/2, n-2}$ is the Student's $t$ critical threshold.

---

## 💾 Database Schema & Datasets

### 1. Canonical Sample Database (`data/salary_data.csv`)
The primary benchmark database specified for the project:

| Experience (Years) | Salary (USD) | Seniority Tier | Monthly Comp |
|:---:|:---:|:---:|:---:|
| **1.0** | $50,000.00 | Junior Level | $4,166.67 |
| **2.0** | $60,000.00 | Junior Level | $5,000.00 |
| **3.0** | $70,000.00 | Mid-Level | $5,833.33 |
| **4.0** | $80,000.00 | Mid-Level | $6,666.67 |
| **5.0** | $90,000.00 | Mid-Level | $7,500.00 |
| **6.0** | $100,000.00 | Senior Level | $8,333.33 |
| **7.0** | $110,000.00 | Senior Level | $9,166.67 |
| **8.0** | $120,000.00 | Senior Level | $10,000.00 |
| **9.0** | $130,000.00 | Lead / Principal | $10,833.33 |
| **10.0** | $140,000.00 | Lead / Principal | $11,666.67 |

### 2. Realistic Market Database (`data/salary_data_market.csv`)
An extended 30-observation dataset incorporating market variation, bonus variance, and non-linear senior jumps ($1.1$ to $12.0$ years of experience). Used to test real-world residual spread, cross-validation, and confidence bands.

---

## ⚙️ Installation & Setup Guide

### Step 1: Verify Python Environment
Ensure Python 3.10 or higher is installed:
```bash
python --version
```

### Step 2: Navigate to Project Directory
```bash
cd "salary predication"
```

### Step 3: Install Required Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Complete Test Suite
Execute all 13 automated tests to verify model convergence and web endpoints:
```bash
python -m unittest discover -s tests -v
```

---

## 🚀 Localhost Web Application Walkthrough

### Starting the Web Server
Launch the application using either of the following commands:
```bash
python app.py
```
*or*
```bash
python main.py
```

You will see:
```text
======================================================================
🚀 Salary Prediction System Web Server running at:
👉 Local URL: http://127.0.0.1:5000
======================================================================
```

### Accessing the Web Application
Open your preferred browser (Chrome, Firefox, Edge, Safari) and navigate to:
```
http://127.0.0.1:5000
```

### User Interface Guide:
1. **Interactive Experience Input**:
   - Drag the slider between $0$ and $25$ years, or type an exact decimal into the input box (e.g. `4.5` or `12.3`).
   - Click any **Quick Milestone** chip (`1.0 yr`, `3.0 yrs`, `5.0 yrs`, `7.5 yrs`, `10.0 yrs`, `15.0 yrs`).
2. **Instant Live Predictions**:
   - The compensation card updates in real time with Annual, Monthly, Weekly, and Hourly rates.
   - Shows the active 95% Confidence Interval and Seniority Tier badge.
3. **Interactive Regression Curve**:
   - Watch the gold square marker track your position along the fitted regression line on the Chart.js canvas.
4. **Model Retraining & Dataset Switcher**:
   - Select either the **Canonical Sample (10 Rows)** or **Real Market Extended (30 Rows)**.
   - Choose between **Simple Linear Regression** or **Polynomial Regression (Deg 2)**.
   - Click **🔄 Retrain & Refresh Pipeline** to update all metrics, charts, and tables without restarting the server.
5. **Detailed Exploratory Tabs**:
   - **💾 Database & Statistics**: Displays all rows with calculated seniority and monthly compensation.
   - **🖼️ 4-Panel Diagnostic Dashboard**: View the high-res 300 DPI Matplotlib diagnostic dashboard.
   - **📑 Batch Milestone Predictor**: Enter comma-separated values to generate batch compensation reports.

---

## 🔌 REST API Documentation

The server exposes clean JSON endpoints for integration into third-party services:

### 1. `POST /api/predict`
Calculates compensation and confidence intervals for a single experience figure.

- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "experience": 4.5
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "experience": 4.5,
    "predicted_salary": 85000.0,
    "formatted_annual": "$85,000.00",
    "formatted_monthly": "$7,083.33",
    "formatted_weekly": "$1,634.62",
    "formatted_hourly": "$40.87",
    "formatted_interval": "$85,000.00 – $85,000.00",
    "seniority_tier": "Mid-Level Professional",
    "lower_bound": 85000.0,
    "upper_bound": 85000.0,
    "equation": "Salary = $40,000.00 + ($10,000.00 × Experience)"
  }
  ```

---

### 2. `POST /api/batch`
Performs batch prediction on an array of experience milestones.

- **Request Body**:
  ```json
  {
    "experiences": [1.0, 3.5, 6.0, 9.0]
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "results": [
      {
        "experience": 1.0,
        "salary": 50000.0,
        "formatted_annual": "$50,000.00",
        "formatted_monthly": "$4,166.67",
        "interval": "$50,000.00 – $50,000.00"
      },
      {
        "experience": 3.5,
        "salary": 75000.0,
        "formatted_annual": "$75,000.00",
        "formatted_monthly": "$6,250.00",
        "interval": "$75,000.00 – $75,000.00"
      }
    ]
  }
  ```

---

### 3. `GET /api/dataset`
Returns raw data points and calculated trend line coordinates for plotting.

- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "dataset_name": "salary_data.csv",
    "dataset_id": "default",
    "record_count": 10,
    "data_points": [
      {"experience": 1.0, "salary": 50000.0},
      {"experience": 2.0, "salary": 60000.0}
    ],
    "trend_line": [
      {"experience": 0.5, "salary": 45000.0},
      {"experience": 1.0, "salary": 50000.0}
    ]
  }
  ```

---

### 4. `POST /api/retrain`
Dynamically retrains the model with a selected algorithm and dataset.

- **Request Body**:
  ```json
  {
    "dataset": "market",
    "model_type": "linear",
    "degree": 2
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "message": "Successfully retrained LINEAR model on salary_data_market.csv",
    "equation": "Salary = $38,474.78 + ($10,201.73 × Experience)",
    "metrics": {
      "r2_score": 0.9979,
      "adjusted_r2": 0.9974,
      "mae": 1099.43,
      "rmse": 1267.62,
      "mape_percent": 1.19,
      "cv_r2_mean": 0.9985
    }
  }
  ```

---

### 5. `GET /api/plot`
Streams the generated 300 DPI Matplotlib PNG diagnostic figure.

---

## 📟 Command-Line Interface (CLI) Fallback

For terminal power users, automated cron jobs, or headless servers, `main.py` provides full CLI functionality:

### Instant Prediction
```bash
python main.py --predict 4.5
```
**Output:**
```text
════════════════ PREDICTION RESULT ════════════════
  Experience:       4.5 Years
  Annual Salary:    $85,000.00
  Monthly Salary:   $7,083.33
  95% Interval:     $85,000.00 – $85,000.00
  Formula:          Salary = $40,000.00 + ($10,000.00 × Experience)
═══════════════════════════════════════════════════
```

### Batch Milestone Inference
```bash
python main.py --batch "1, 2.5, 5, 8.5, 10"
```

### Legacy Interactive Terminal Menu
```bash
python main.py --cli
```

### Headless Diagnostic Figure Generation
```bash
python main.py --visualize
```

---

## 🧪 Testing & Quality Assurance

The project includes 13 automated unit and integration tests split into two test modules:

```bash
python -m unittest discover -s tests -v
```

### Test Suite Execution Output:
```text
test_01_dataset_schema_and_values (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_02_data_validation_boundaries (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_03_preprocessor_formatting (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_04_linear_model_training (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_05_prediction_inference (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_06_batch_prediction (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_07_model_persistence (test_pipeline.TestSalaryPredictionPipeline) ... ok
test_01_index_page (test_web_app.TestWebApplication) ... ok
test_02_predict_endpoint_valid (test_web_app.TestWebApplication) ... ok
test_03_predict_endpoint_invalid_input (test_web_app.TestWebApplication) ... ok
test_04_dataset_endpoint (test_web_app.TestWebApplication) ... ok
test_05_batch_endpoint (test_web_app.TestWebApplication) ... ok
test_06_model_info_and_retrain (test_web_app.TestWebApplication) ... ok

----------------------------------------------------------------------
Ran 13 tests in 7.523s

OK
```

### What the Tests Verify:
1. **Data Ingestion Integrity**: Confirms exactly 10 rows in canonical database, correct column types, and lack of nulls.
2. **Boundary Validation**: Rejects negative years, non-numeric strings, and unrealistic career spans ($>50$ yrs).
3. **2D Feature Matrix Reshaping**: Verifies matrix dimensions $(n, 1)$ required by Scikit-learn estimators.
4. **OLS Mathematical Convergence**: Confirms exact convergence on benchmark data ($\beta_0 = 40,000$, $\beta_1 = 10,000$, $R^2 = 1.000$, $\text{MAE} = 0.00$).
5. **Inference Accuracy**: Validates precise interpolation ($4.5\text{ yrs} \to \$85,000.00$) and extrapolation ($12.0\text{ yrs} \to \$160,000.00$).
6. **Batch Inference Consistency**: Checks multi-milestone data structure and DataFrame column outputs.
7. **Joblib Serialization**: Verifies saving and loading model weights without accuracy degradation.
8. **Web HTTP Contracts**: Tests all Flask routes, HTTP status codes, JSON response schemas, and 422 input validation guards.

---

## ❓ Troubleshooting & FAQ

### Q1: Port 5000 is already in use by another service.
**Solution:** Start the server on an alternative port (e.g. 5050):
```bash
python main.py --port 5050
```
Then open `http://127.0.0.1:5050` in your browser.

### Q2: How can I add new salary records to the dataset?
**Solution:** Open `data/salary_data.csv` in any text editor or Excel, append your rows adhering to the format:
```csv
Experience (Years),Salary (USD)
11,150000
12,160000
```
Then click **🔄 Retrain & Refresh Pipeline** on the web page to immediately retrain the model.

### Q3: Why is the 95% Confidence Interval identical to the estimate on the sample dataset?
**Solution:** In the canonical 10-sample database, the data is mathematically linear ($\text{Salary} = 40,000 + 10,000 \times \text{Exp}$). Hence, residuals $e_i = 0$ and Residual Standard Error $s_e = 0.00$. To view variance-based statistical intervals (e.g., $\pm \$3,000$), switch the dataset selector to the **Real Market Extended (30 Rows)** database.

---

## 📄 License & Attribution
Developed as an advanced Data Science & Machine Learning engineering system demonstrating best practices in predictive modeling, statistical inference, and full-stack Python application design.
