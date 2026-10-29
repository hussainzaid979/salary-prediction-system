/**
 * Salary Prediction System — Frontend Controller
 * Synchronizes inputs, connects to localhost REST API, and renders Chart.js visualizations.
 */

// Global Chart instance
let regressionChart = null;
let currentExperience = 4.5;
let debounceTimer = null;

// DOM Elements
const expSlider = document.getElementById("expSlider");
const expInput = document.getElementById("expInput");
const expDisplay = document.getElementById("expDisplay");

const annualSalaryEl = document.getElementById("annualSalary");
const monthlySalaryEl = document.getElementById("monthlySalary");
const weeklySalaryEl = document.getElementById("weeklySalary");
const hourlySalaryEl = document.getElementById("hourlySalary");
const intervalRangeEl = document.getElementById("intervalRange");
const tierBadgeEl = document.getElementById("tierBadge");
const formulaTextEl = document.getElementById("formulaText");

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  initRegressionChart();
  loadDatasetTable();
  predictSalary(currentExperience);
});

/**
 * Attaches synchronized events to slider and input field
 */
function setupEventListeners() {
  expSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    expInput.value = val;
    expDisplay.textContent = `${val.toFixed(1)} Years`;
    debouncePrediction(val);
  });

  expInput.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val) && val >= 0) {
      expSlider.value = Math.min(val, 25);
      expDisplay.textContent = `${val.toFixed(1)} Years`;
      debouncePrediction(val);
    }
  });
}

/**
 * Debounce helper to prevent excessive API calls while sliding
 */
function debouncePrediction(val) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    predictSalary(val);
  }, 80);
}

/**
 * Sets experience from quick preset buttons
 */
function setExperience(years) {
  currentExperience = years;
  expSlider.value = Math.min(years, 25);
  expInput.value = years;
  expDisplay.textContent = `${years.toFixed(1)} Years`;
  predictSalary(years);
}

/**
 * Calls localhost API to predict compensation
 */
async function predictSalary(experience) {
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experience: experience })
    });

    const data = await response.json();
    if (data.success) {
      annualSalaryEl.textContent = data.formatted_annual;
      monthlySalaryEl.textContent = data.formatted_monthly;
      weeklySalaryEl.textContent = data.formatted_weekly;
      hourlySalaryEl.textContent = data.formatted_hourly;
      intervalRangeEl.textContent = data.formatted_interval;
      tierBadgeEl.textContent = data.seniority_tier;
      formulaTextEl.textContent = data.equation;

      // Update query position on Chart.js
      updateChartTargetPoint(data.experience, data.predicted_salary);
    } else {
      console.error("Prediction error:", data.error);
    }
  } catch (err) {
    console.error("Failed to connect to /api/predict:", err);
  }
}

/**
 * Initializes interactive Chart.js regression chart
 */
async function initRegressionChart() {
  try {
    const res = await fetch("/api/dataset");
    const json = await res.json();
    if (!json.success) return;

    const dataPoints = json.data_points.map(pt => ({ x: pt.experience, y: pt.salary }));
    const trendPoints = json.trend_line.map(pt => ({ x: pt.experience, y: pt.salary }));

    const ctx = document.getElementById("regressionChart").getContext("2d");

    if (regressionChart) {
      regressionChart.destroy();
    }

    regressionChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Observations (Data)",
            data: dataPoints,
            backgroundColor: "#2563eb",
            borderColor: "#1d4ed8",
            pointRadius: 6,
            pointHoverRadius: 8,
            order: 3
          },
          {
            label: "Fitted Regression Curve",
            data: trendPoints,
            type: "line",
            borderColor: "#dc2626",
            borderWidth: 2.5,
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            order: 2
          },
          {
            label: "Current Query Marker",
            data: [{ x: currentExperience, y: 85000 }],
            backgroundColor: "#f59e0b",
            borderColor: "#b45309",
            borderWidth: 2,
            pointRadius: 10,
            pointHoverRadius: 12,
            pointStyle: "rectRot",
            order: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const x = context.parsed.x;
                const y = context.parsed.y;
                return ` ${context.dataset.label}: ${x.toFixed(1)} yrs → $${y.toLocaleString()}`;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Years of Experience",
              font: { weight: "bold", size: 12 }
            },
            grid: { color: "#e2e8f0" }
          },
          y: {
            title: {
              display: true,
              text: "Salary (USD)",
              font: { weight: "bold", size: 12 }
            },
            ticks: {
              callback: (value) => "$" + value.toLocaleString()
            },
            grid: { color: "#e2e8f0" }
          }
        }
      }
    });

  } catch (err) {
    console.error("Error loading regression chart:", err);
  }
}

/**
 * Updates the gold marker coordinates on the chart in real time
 */
function updateChartTargetPoint(exp, salary) {
  if (!regressionChart) return;
  regressionChart.data.datasets[2].data = [{ x: exp, y: salary }];
  regressionChart.update("none"); // Update smoothly without reanimating entire canvas
}

/**
 * Populates active dataset table
 */
async function loadDatasetTable() {
  try {
    const res = await fetch("/api/dataset");
    const json = await res.json();
    if (!json.success) return;

    const tbody = document.getElementById("datasetTableBody");
    tbody.innerHTML = "";

    document.getElementById("dbRecordCount").textContent = json.record_count;

    json.data_points.forEach((pt, index) => {
      const exp = pt.experience;
      const salary = pt.salary;
      const monthly = salary / 12.0;

      let tier = "Junior (0-2 yrs)";
      if (exp >= 9.0) tier = "Lead / Principal (9+ yrs)";
      else if (exp >= 6.0) tier = "Senior (6-8 yrs)";
      else if (exp >= 3.0) tier = "Mid-Level (3-5 yrs)";

      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${index + 1}</strong></td>
        <td>${exp.toFixed(1)} yrs</td>
        <td><strong>$${salary.toLocaleString("en-US", { minimumFractionDigits: 2 })}</strong></td>
        <td>$${monthly.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td><span class="chip">${tier}</span></td>
      `;
      tbody.appendChild(row);
    });

  } catch (err) {
    console.error("Error loading dataset table:", err);
  }
}

/**
 * Retrains the machine learning model with selected algorithm & dataset
 */
async function handleRetrain() {
  const dataset = document.getElementById("datasetSelect").value;
  const modelType = document.getElementById("modelSelect").value;
  const retrainBtn = document.getElementById("retrainBtn");

  retrainBtn.disabled = true;
  retrainBtn.textContent = "⏳ Retraining Model...";

  try {
    const res = await fetch("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: dataset, model_type: modelType, degree: 2 })
    });

    const data = await res.json();
    if (data.success) {
      // Update Metrics on UI
      const m = data.metrics;
      document.getElementById("metricR2").textContent = m.r2_score.toFixed(4);
      document.getElementById("metricAdjR2").textContent = m.adjusted_r2.toFixed(4);
      document.getElementById("metricMAE").textContent = "$" + m.mae.toLocaleString("en-US", { minimumFractionDigits: 2 });
      document.getElementById("metricRMSE").textContent = "$" + m.rmse.toLocaleString("en-US", { minimumFractionDigits: 2 });
      document.getElementById("metricMAPE").textContent = m.mape_percent.toFixed(2) + "%";
      document.getElementById("metricCV").textContent = m.cv_r2_mean.toFixed(4);

      // Refresh chart and tables
      await initRegressionChart();
      await loadDatasetTable();
      await predictSalary(parseFloat(expInput.value));

      // Force refresh diagnostic image
      const img = document.getElementById("dashboardImg");
      img.src = `/static/images/salary_prediction_analysis.png?t=${Date.now()}`;

      retrainBtn.textContent = "✔ Retrained Successfully!";
      setTimeout(() => {
        retrainBtn.disabled = false;
        retrainBtn.textContent = "🔄 Retrain & Refresh Pipeline";
      }, 1500);
    } else {
      alert("Retraining failed: " + data.error);
      retrainBtn.disabled = false;
      retrainBtn.textContent = "🔄 Retrain & Refresh Pipeline";
    }
  } catch (err) {
    alert("Connection error: " + err);
    retrainBtn.disabled = false;
    retrainBtn.textContent = "🔄 Retrain & Refresh Pipeline";
  }
}

/**
 * Executes batch inference for milestone inputs
 */
async function runBatchPrediction() {
  const rawInput = document.getElementById("batchInputs").value;
  const expArray = rawInput.split(",")
    .map(x => parseFloat(x.trim()))
    .filter(x => !isNaN(x) && x >= 0);

  if (expArray.length === 0) {
    alert("Please enter valid comma-separated numbers.");
    return;
  }

  try {
    const res = await fetch("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiences: expArray })
    });

    const data = await res.json();
    if (data.success) {
      const tbody = document.getElementById("batchTableBody");
      tbody.innerHTML = "";

      data.results.forEach(item => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${item.experience} yrs</strong></td>
          <td style="color: #059669; font-weight: 700;">${item.formatted_annual}</td>
          <td>${item.formatted_monthly}</td>
          <td><span class="chip">${item.interval}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    alert("Batch prediction error: " + err);
  }
}

/**
 * Handles Tab navigation
 */
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(pane => pane.classList.remove("active"));

  event.target.classList.add("active");
  document.getElementById(tabId).classList.add("active");
}
