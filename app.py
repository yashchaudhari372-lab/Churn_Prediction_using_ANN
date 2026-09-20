import os
import pickle
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

# Set headless environment flags for TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Base directory for relative model file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_ann_model.keras")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Cached module-level references
model = None
encoders = None
scaler = None
init_error = None

def load_artifacts():
    global model, encoders, scaler, init_error
    try:
        # Validate existence of all required model artifacts
        for name, path in [("ANN Model", MODEL_PATH), ("Encoders", ENCODERS_PATH), ("Scaler", SCALER_PATH)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required artifact '{name}' not found at {path}")

        # Lazy load Keras to optimize Vercel serverless startup
        import keras
        model = keras.models.load_model(MODEL_PATH)

        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        # Validate encoder contents
        required_keys = ['le_gender', 'ohe_geo', 'feature_columns', 'geo_categories']
        for key in required_keys:
            if key not in encoders:
                raise KeyError(f"Missing essential key '{key}' in encoders.pkl")

    except Exception as e:
        init_error = f"Artifact Initialization Failed: {str(e)}"

# Pre-load artifacts at import time for serverless container reuse
load_artifacts()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Churn Prediction using ANN | AI-Powered Customer Churn Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

  <style>
    :root {
      --bg-gradient: radial-gradient(circle at 10% 20%, rgb(10, 15, 30) 0%, rgb(16, 24, 48) 90.2%);
      --card-bg: rgba(22, 33, 62, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --input-bg: rgba(15, 23, 42, 0.6);
      --input-border: rgba(255, 255, 255, 0.15);
      --badge-low: #10b981;
      --badge-med: #f59e0b;
      --badge-high: #ef4444;
      --glow: rgba(59, 130, 246, 0.25);
    }

    [data-theme="galaxy"] {
      --bg-gradient: radial-gradient(circle at 20% 30%, #0f0c1b 0%, #1e1136 100%);
      --card-bg: rgba(36, 20, 68, 0.65);
      --card-border: rgba(216, 180, 254, 0.12);
      --primary: #a855f7;
      --primary-hover: #9333ea;
      --accent: #ec4899;
      --input-bg: rgba(20, 10, 38, 0.7);
      --glow: rgba(168, 85, 247, 0.3);
    }

    [data-theme="emerald"] {
      --bg-gradient: radial-gradient(circle at 20% 20%, #031a14 0%, #082f25 100%);
      --card-bg: rgba(6, 46, 36, 0.65);
      --card-border: rgba(52, 211, 153, 0.12);
      --primary: #10b981;
      --primary-hover: #059669;
      --accent: #34d399;
      --input-bg: rgba(2, 28, 21, 0.7);
      --glow: rgba(16, 185, 129, 0.3);
    }

    [data-theme="sunset"] {
      --bg-gradient: radial-gradient(circle at 50% 20%, #1a0b1c 0%, #2b121e 100%);
      --card-bg: rgba(49, 18, 38, 0.65);
      --card-border: rgba(251, 113, 133, 0.15);
      --primary: #f43f5e;
      --primary-hover: #e11d48;
      --accent: #fb923c;
      --input-bg: rgba(26, 10, 22, 0.7);
      --glow: rgba(244, 63, 94, 0.3);
    }

    [data-theme="cyber"] {
      --bg-gradient: radial-gradient(circle at 10% 10%, #02151b 0%, #042533 100%);
      --card-bg: rgba(4, 41, 56, 0.65);
      --card-border: rgba(34, 211, 238, 0.15);
      --primary: #06b6d4;
      --primary-hover: #0891b2;
      --accent: #38bdf8;
      --input-bg: rgba(2, 22, 31, 0.7);
      --glow: rgba(6, 182, 212, 0.3);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      transition: background-color 0.3s ease, border-color 0.3s ease;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
      background: var(--bg-gradient);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2.5rem 1.25rem;
      display: flex;
      justify-content: center;
    }

    .dashboard-container {
      width: 100%;
      max-width: 1200px;
      animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }

    .brand-title {
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
      font-weight: 400;
    }

    .theme-selector {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      padding: 0.4rem 0.8rem;
      border-radius: 9999px;
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .theme-selector label {
      font-size: 0.8rem;
      color: var(--text-muted);
      font-weight: 600;
    }

    .theme-selector select {
      background: transparent;
      border: none;
      color: var(--text-main);
      font-weight: 600;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
    }

    .theme-selector select option {
      background: #0f172a;
      color: #fff;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(14px);
      padding: 1.25rem 1.5rem;
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .kpi-card h4 {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }

    .kpi-card .val {
      font-size: 1.4rem;
      font-weight: 700;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2rem;
    }

    @media (min-width: 992px) {
      .main-grid {
        grid-template-columns: 1.15fr 0.85fr;
      }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(16px);
      border-radius: 20px;
      padding: 2rem;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    }

    .card-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .form-group label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #cbd5e1;
    }

    .form-group input, .form-group select {
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      color: var(--text-main);
      padding: 0.75rem 1rem;
      border-radius: 10px;
      font-size: 0.95rem;
      outline: none;
    }

    .form-group input:focus, .form-group select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--glow);
    }

    .helper-text {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    .btn-row {
      display: flex;
      gap: 1rem;
      margin-top: 1.75rem;
    }

    .btn {
      flex: 1;
      padding: 0.85rem 1.5rem;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      border: none;
      display: inline-flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }

    .btn-primary {
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 4px 14px var(--glow);
    }

    .btn-primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      color: #cbd5e1;
      border: 1px solid var(--card-border);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    .result-container {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      animation: fadeIn 0.6s ease;
    }

    .prediction-banner {
      padding: 1.5rem;
      border-radius: 16px;
      text-align: center;
      border: 1px solid;
    }

    .banner-churn {
      background: rgba(239, 68, 68, 0.12);
      border-color: rgba(239, 68, 68, 0.3);
    }

    .banner-stay {
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.3);
    }

    .prediction-title {
      font-size: 1.45rem;
      font-weight: 800;
    }

    .prediction-title.churn { color: #f87171; }
    .prediction-title.stay { color: #34d399; }

    .risk-badge {
      display: inline-block;
      margin-top: 0.5rem;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .badge-High { background: var(--badge-high); color: #fff; }
    .badge-Medium { background: var(--badge-med); color: #000; }
    .badge-Low { background: var(--badge-low); color: #fff; }

    .metrics-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    .metric-card {
      background: rgba(0, 0, 0, 0.2);
      padding: 1rem;
      border-radius: 12px;
      text-align: center;
      border: 1px solid var(--card-border);
    }

    .metric-card span {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .metric-card h3 {
      font-size: 1.3rem;
      margin-top: 0.2rem;
      font-weight: 700;
    }

    .chart-box {
      position: relative;
      width: 100%;
      height: 220px;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .profile-summary {
      background: rgba(0, 0, 0, 0.2);
      border-radius: 12px;
      padding: 1.25rem;
      font-size: 0.85rem;
      line-height: 1.6;
      color: #cbd5e1;
    }

    .profile-summary ul {
      margin-left: 1.2rem;
      margin-top: 0.4rem;
    }

    .alert {
      padding: 1rem;
      border-radius: 10px;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
      border: 1px solid;
    }

    .alert-error {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }

    .disclaimer-card {
      margin-top: 2rem;
      padding: 1.25rem 1.5rem;
      background: rgba(0, 0, 0, 0.15);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="dashboard-container">
    <div class="header-bar">
      <div>
        <h1 class="brand-title">Churn Prediction using ANN</h1>
        <p class="brand-subtitle">AI-Powered Customer Churn Intelligence</p>
      </div>
      <div class="theme-selector">
        <label for="themeSelect">Theme:</label>
        <select id="themeSelect" onchange="changeTheme(this.value)">
          <option value="ocean">Ocean Blue</option>
          <option value="galaxy">Purple Galaxy</option>
          <option value="emerald">Emerald</option>
          <option value="sunset">Sunset</option>
          <option value="cyber">Cyber Cyan</option>
        </select>
      </div>
    </div>

    <!-- Status & Info KPI Grid -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <h4>Model Architecture</h4>
        <div class="val">ANN (Sequential)</div>
      </div>
      <div class="kpi-card">
        <h4>Input Features</h4>
        <div class="val">11 Dimensions</div>
      </div>
      <div class="kpi-card">
        <h4>Task</h4>
        <div class="val">Binary Classification</div>
      </div>
      <div class="kpi-card">
        <h4>System Status</h4>
        <div class="val" style="color: {% if init_error %}#f87171{% else %}#34d399{% endif %};">
          {% if init_error %}Degraded{% else %}Ready{% endif %}
        </div>
      </div>
    </div>

    {% if init_error %}
      <div class="alert alert-error">
        <strong>Runtime Initialization Alert:</strong> {{ init_error }}
      </div>
    {% endif %}

    {% if error_msg %}
      <div class="alert alert-error">
        <strong>Validation Error:</strong> {{ error_msg }}
      </div>
    {% endif %}

    <div class="main-grid">
      <!-- Input Form Card -->
      <div class="card">
        <h2 class="card-title">Customer Information</h2>
        <form method="POST" action="/" id="predictionForm">
          <div class="form-grid">
            <div class="form-group">
              <label for="CreditScore">Credit Score</label>
              <input type="number" id="CreditScore" name="CreditScore" min="300" max="900" step="1" required
                value="{{ form_data.get('CreditScore', '650') }}" />
              <span class="helper-text">Standard range: 300 - 850</span>
            </div>

            <div class="form-group">
              <label for="Geography">Geography</label>
              <select id="Geography" name="Geography" required>
                {% for geo in geo_categories %}
                  <option value="{{ geo }}" {% if form_data.get('Geography') == geo %}selected{% endif %}>{{ geo }}</option>
                {% endfor %}
              </select>
              <span class="helper-text">Primary bank branch jurisdiction</span>
            </div>

            <div class="form-group">
              <label for="Gender">Gender</label>
              <select id="Gender" name="Gender" required>
                {% for gen in gender_categories %}
                  <option value="{{ gen }}" {% if form_data.get('Gender') == gen %}selected{% endif %}>{{ gen }}</option>
                {% endfor %}
              </select>
              <span class="helper-text">Client gender demographic</span>
            </div>

            <div class="form-group">
              <label for="Age">Age</label>
              <input type="number" id="Age" name="Age" min="18" max="100" step="1" required
                value="{{ form_data.get('Age', '40') }}" />
              <span class="helper-text">Age in years</span>
            </div>

            <div class="form-group">
              <label for="Tenure">Tenure</label>
              <input type="number" id="Tenure" name="Tenure" min="0" max="10" step="1" required
                value="{{ form_data.get('Tenure', '5') }}" />
              <span class="helper-text">Years account has remained active</span>
            </div>

            <div class="form-group">
              <label for="Balance">Balance (€)</label>
              <input type="number" id="Balance" name="Balance" min="0" step="0.01" required
                value="{{ form_data.get('Balance', '60000.00') }}" />
              <span class="helper-text">Current holding balance</span>
            </div>

            <div class="form-group">
              <label for="NumOfProducts">Number of Products</label>
              <select id="NumOfProducts" name="NumOfProducts" required>
                {% for n in [1, 2, 3, 4] %}
                  <option value="{{ n }}" {% if form_data.get('NumOfProducts', '1')|int == n %}selected{% endif %}>{{ n }}</option>
                {% endfor %}
              </select>
              <span class="helper-text">Total banking accounts/services</span>
            </div>

            <div class="form-group">
              <label for="HasCrCard">Has Credit Card</label>
              <select id="HasCrCard" name="HasCrCard" required>
                <option value="1" {% if form_data.get('HasCrCard', '1') == '1' %}selected{% endif %}>Yes</option>
                <option value="0" {% if form_data.get('HasCrCard') == '0' %}selected{% endif %}>No</option>
              </select>
              <span class="helper-text">Indicates valid credit line</span>
            </div>

            <div class="form-group">
              <label for="IsActiveMember">Active Member</label>
              <select id="IsActiveMember" name="IsActiveMember" required>
                <option value="1" {% if form_data.get('IsActiveMember', '1') == '1' %}selected{% endif %}>Yes</option>
                <option value="0" {% if form_data.get('IsActiveMember') == '0' %}selected{% endif %}>No</option>
              </select>
              <span class="helper-text">Recent transactional engagement</span>
            </div>

            <div class="form-group">
              <label for="EstimatedSalary">Estimated Salary (€)</label>
              <input type="number" id="EstimatedSalary" name="EstimatedSalary" min="0" step="0.01" required
                value="{{ form_data.get('EstimatedSalary', '50000.00') }}" />
              <span class="helper-text">Calculated yearly earning</span>
            </div>
          </div>

          <div class="btn-row">
            <button type="submit" class="btn btn-primary" id="submitBtn">
              <span>Predict Churn</span>
            </button>
            <button type="button" class="btn btn-secondary" onclick="resetForm()">
              <span>Reset</span>
            </button>
          </div>
        </form>
      </div>

      <!-- Result Card -->
      <div class="card">
        <h2 class="card-title">Prediction Intelligence</h2>

        {% if prediction %}
          <div class="result-container">
            <div class="prediction-banner {% if prediction.churn_label == 1 %}banner-churn{% else %}banner-stay{% endif %}">
              <div class="prediction-title {% if prediction.churn_label == 1 %}churn{% else %}stay{% endif %}">
                {% if prediction.churn_label == 1 %}Customer Likely to Churn{% else %}Customer Likely to Stay{% endif %}
              </div>
              <span class="risk-badge badge-{{ prediction.risk_level }}">{{ prediction.risk_level }} Risk</span>
            </div>

            <div class="metrics-row">
              <div class="metric-card">
                <span>Churn Probability</span>
                <h3 style="color: #f87171;">{{ prediction.churn_prob }}%</h3>
              </div>
              <div class="metric-card">
                <span>Retention Probability</span>
                <h3 style="color: #34d399;">{{ prediction.retention_prob }}%</h3>
              </div>
            </div>

            <div class="chart-box">
              <canvas id="churnChart"></canvas>
            </div>

            <div class="profile-summary">
              <strong>Customer Profile Context:</strong>
              <ul>
                <li>{{ form_data.Geography }} resident, age {{ form_data.Age }}, {{ form_data.Tenure }} yr(s) relationship</li>
                <li>Balance: €{{ "{:,.2f}".format(form_data.Balance|float) }} across {{ form_data.NumOfProducts }} product(s)</li>
                <li>Activity Status: {% if form_data.IsActiveMember == '1' %}Active Account{% else %}Inactive Account{% endif %}</li>
              </ul>
            </div>
          </div>
        {% else %}
          <div style="text-align: center; padding: 4rem 1rem; color: var(--text-muted);">
            <svg style="width: 48px; height: 48px; margin-bottom: 1rem; opacity: 0.4;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            <p>Submit client profile metrics on the left to evaluate retention probability via the ANN inference engine.</p>
          </div>
        {% endif %}
      </div>
    </div>

    <!-- Technical Model Information & Disclaimer -->
    <div class="disclaimer-card">
      <h4 style="color: var(--text-main); margin-bottom: 0.5rem; font-size: 0.95rem;">About the Model</h4>
      <p style="margin-bottom: 0.75rem;">
        This application uses a pre-trained Artificial Neural Network (ANN) model to estimate customer churn probability based on customer demographic, financial and account-related features.
      </p>
      <p style="font-size: 0.75rem;">
        <em>Disclaimer: Prediction generated by the trained ANN model. Results are probabilistic and should be interpreted as model estimates.</em>
      </p>
    </div>
  </div>

  <script>
    function changeTheme(theme) {
      if (theme === 'ocean') {
        document.documentElement.removeAttribute('data-theme');
      } else {
        document.documentElement.setAttribute('data-theme', theme);
      }
      localStorage.setItem('dashboard_theme', theme);
    }

    // Restore user theme preference
    const savedTheme = localStorage.getItem('dashboard_theme');
    if (savedTheme) {
      document.getElementById('themeSelect').value = savedTheme;
      changeTheme(savedTheme);
    }

    function resetForm() {
      window.location.href = "/";
    }

    // Client form state handling
    document.getElementById('predictionForm').addEventListener('submit', function() {
      const btn = document.getElementById('submitBtn');
      btn.innerHTML = '<span>Processing Inference...</span>';
      btn.style.opacity = '0.7';
      btn.style.pointerEvents = 'none';
    });

    {% if prediction %}
    // Render Doughnut Chart for Risk Analytics
    const ctx = document.getElementById('churnChart').getContext('2d');
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Churn Risk', 'Retention Chance'],
        datasets: [{
          data: [{{ prediction.churn_prob }}, {{ prediction.retention_prob }}],
          backgroundColor: ['#ef4444', '#10b981'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#cbd5e1',
              font: { family: 'Plus Jakarta Sans', size: 12 }
            }
          }
        }
      }
    });
    {% endif %}
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global model, encoders, scaler, init_error

    if init_error:
        return render_template_string(
            HTML_TEMPLATE,
            init_error=init_error,
            error_msg=None,
            prediction=None,
            form_data={},
            geo_categories=['France', 'Germany', 'Spain'],
            gender_categories=['Female', 'Male']
        )

    # Extract dynamic categories from pickled encoders
    geo_categories = list(encoders.get('geo_categories', ['France', 'Germany', 'Spain']))
    gender_categories = list(encoders['le_gender'].classes_)

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            # 1. Server-side Input Validation
            credit_score = float(form_data.get("CreditScore", ""))
            gender = form_data.get("Gender", "").strip()
            age = float(form_data.get("Age", ""))
            tenure = float(form_data.get("Tenure", ""))
            balance = float(form_data.get("Balance", ""))
            num_of_products = int(form_data.get("NumOfProducts", ""))
            has_cr_card = int(form_data.get("HasCrCard", ""))
            is_active_member = int(form_data.get("IsActiveMember", ""))
            estimated_salary = float(form_data.get("EstimatedSalary", ""))
            geography = form_data.get("Geography", "").strip()

            if not (300 <= credit_score <= 900):
                raise ValueError("Credit score must fall between 300 and 900.")
            if not (18 <= age <= 100):
                raise ValueError("Age must be between 18 and 100.")
            if not (0 <= tenure <= 10):
                raise ValueError("Tenure must be between 0 and 10 years.")
            if balance < 0 or estimated_salary < 0:
                raise ValueError("Balance and Salary cannot be negative values.")
            if gender not in gender_categories:
                raise ValueError(f"Invalid Gender selection. Choose from: {gender_categories}")
            if geography not in geo_categories:
                raise ValueError(f"Invalid Geography selection. Choose from: {geo_categories}")
            if num_of_products not in [1, 2, 3, 4]:
                raise ValueError("Number of Products must be between 1 and 4.")
            if has_cr_card not in [0, 1] or is_active_member not in [0, 1]:
                raise ValueError("Invalid binary status input.")

            # 2. Gender transformation using existing LabelEncoder
            encoded_gender = int(encoders['le_gender'].transform([gender])[0])

            # 3. Geography transformation using existing OneHotEncoder (drop='first')
            # Expected categories: France (dropped), Germany, Spain
            geo_df = pd.DataFrame([[geography]], columns=['Geography'])
            geo_encoded = encoders['ohe_geo'].transform(geo_df)

            # Extract Germany and Spain encoded values
            geo_germany = float(geo_encoded[0][0])
            geo_spain = float(geo_encoded[0][1])

            # 4. Construct exact 11-feature DataFrame matching feature_columns
            feature_data = {
                'CreditScore': [credit_score],
                'Gender': [encoded_gender],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary],
                'Geography_Germany': [geo_germany],
                'Geography_Spain': [geo_spain]
            }

            feature_order = encoders.get('feature_columns', [
                'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
            ])
            input_df = pd.DataFrame(feature_data)[feature_order]

            # 5. Scale features using existing StandardScaler
            scaled_features = scaler.transform(input_df)

            # 6. Generate ANN Model Prediction
            prob_arr = model.predict(scaled_features, verbose=0)
            churn_probability = float(prob_arr[0][0])
            retention_probability = 1.0 - churn_probability

            churn_label = 1 if churn_probability >= 0.5 else 0

            # Determine qualitative risk level
            if churn_probability >= 0.65:
                risk_level = "High"
            elif churn_probability >= 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            prediction_result = {
                "churn_label": churn_label,
                "churn_prob": round(churn_probability * 100, 2),
                "retention_prob": round(retention_probability * 100, 2),
                "risk_level": risk_level
            }

            return render_template_string(
                HTML_TEMPLATE,
                init_error=None,
                error_msg=None,
                prediction=prediction_result,
                form_data=form_data,
                geo_categories=geo_categories,
                gender_categories=gender_categories
            )

        except Exception as e:
            return render_template_string(
                HTML_TEMPLATE,
                init_error=None,
                error_msg=f"Prediction failed: {str(e)}",
                prediction=None,
                form_data=form_data,
                geo_categories=geo_categories,
                gender_categories=gender_categories
            )

    return render_template_string(
        HTML_TEMPLATE,
        init_error=None,
        error_msg=None,
        prediction=None,
        form_data={},
        geo_categories=geo_categories,
        gender_categories=gender_categories
    )

if __name__ == "__main__":
    app.run(debug=True)
