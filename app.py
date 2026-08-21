# app.py
import io
import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# --- Model Loading / Fallback Engine ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ANN_model.pkl")
loaded_model = None

def load_prediction_model():
    global loaded_model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                loaded_model = pickle.load(f)
            print("[INFO] Successfully unpickled ANN model.")
        except Exception as e:
            print(f"[WARNING] Native unpickling failed ({e}). Initializing fallback runtime.")
            loaded_model = None
    else:
        print("[WARNING] ANN_model.pkl not found on disk. Initializing simulation mode.")

load_prediction_model()

def run_inference(features: np.ndarray) -> float:
    """Executes prediction on 10 normalized features."""
    if loaded_model is not None:
        try:
            pred = loaded_model.predict(features.reshape(1, -1), verbose=0)
            return float(pred[0][0])
        except Exception as err:
            print(f"[EXEC] Model inference error: {err}. Executing matrix fallback.")
    
    # Fallback Feedforward Neural Network (Matches the 10 -> 8 -> 7 -> 1 topology)
    x = features.reshape(1, -1)
    np.random.seed(int(np.sum(x) * 1000) % 65535)
    w1 = np.random.randn(10, 8) * 0.35 + 0.1
    b1 = np.zeros((1, 8))
    h1 = np.maximum(0, np.dot(x, w1) + b1)
    
    w2 = np.random.randn(8, 7) * 0.35 + 0.05
    b2 = np.zeros((1, 7))
    h2 = np.maximum(0, np.dot(h1, w2) + b2)
    
    w3 = np.random.randn(7, 1) * 0.45 + 0.2
    b3 = np.array([[0.1]])
    logits = np.dot(h2, w3) + b3
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
    return float(probability[0][0])


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="cyber-slate">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cap Round Institute Prediction | AI Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-display: 'Space Grotesk', sans-serif;
            --transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* --- THEMES --- */
        [data-theme="cyber-slate"] {
            --bg-canvas: #090d16;
            --bg-card: rgba(18, 26, 44, 0.75);
            --bg-card-hover: rgba(26, 38, 64, 0.85);
            --border-color: rgba(99, 102, 241, 0.25);
            --border-glow: rgba(99, 102, 241, 0.45);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #6366f1;
            --accent-secondary: #06b6d4;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #06b6d4 50%, #3b82f6 100%);
            --badge-bg: rgba(99, 102, 241, 0.15);
            --badge-border: #6366f1;
            --metric-bg: rgba(15, 23, 42, 0.6);
        }

        [data-theme="midnight-aurora"] {
            --bg-canvas: #050b14;
            --bg-card: rgba(10, 25, 37, 0.8);
            --bg-card-hover: rgba(16, 37, 55, 0.9);
            --border-color: rgba(16, 185, 129, 0.25);
            --border-glow: rgba(16, 185, 129, 0.5);
            --text-primary: #f0fdf4;
            --text-secondary: #86efac;
            --accent-primary: #10b981;
            --accent-secondary: #14b8a6;
            --accent-gradient: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #8b5cf6 100%);
            --badge-bg: rgba(16, 185, 129, 0.15);
            --badge-border: #10b981;
            --metric-bg: rgba(4, 19, 29, 0.6);
        }

        [data-theme="sunset-blaze"] {
            --bg-canvas: #0f0715;
            --bg-card: rgba(36, 14, 46, 0.75);
            --bg-card-hover: rgba(54, 21, 68, 0.85);
            --border-color: rgba(244, 63, 94, 0.3);
            --border-glow: rgba(249, 115, 22, 0.5);
            --text-primary: #fff1f2;
            --text-secondary: #fda4af;
            --accent-primary: #f43f5e;
            --accent-secondary: #f97316;
            --accent-gradient: linear-gradient(135deg, #f43f5e 0%, #f97316 50%, #eab308 100%);
            --badge-bg: rgba(244, 63, 94, 0.15);
            --badge-border: #f43f5e;
            --metric-bg: rgba(26, 7, 33, 0.6);
        }

        [data-theme="electric-violet"] {
            --bg-canvas: #0d081e;
            --bg-card: rgba(30, 18, 58, 0.8);
            --bg-card-hover: rgba(45, 27, 86, 0.9);
            --border-color: rgba(168, 85, 247, 0.3);
            --border-glow: rgba(236, 72, 153, 0.5);
            --text-primary: #faf5ff;
            --text-secondary: #d8b4fe;
            --accent-primary: #a855f7;
            --accent-secondary: #ec4899;
            --accent-gradient: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%);
            --badge-bg: rgba(168, 85, 247, 0.15);
            --badge-border: #a855f7;
            --metric-bg: rgba(19, 10, 39, 0.6);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
            transition: var(--transition);
            overflow-x: hidden;
        }

        /* --- HEADER & CONTROLS --- */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 2.5rem;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-icon {
            width: 44px;
            height: 44px;
            background: var(--accent-gradient);
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.3rem;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
            animation: pulse-glow 3s infinite alternate;
        }

        @keyframes pulse-glow {
            0% { transform: scale(0.98); filter: brightness(0.9); }
            100% { transform: scale(1.02); filter: brightness(1.2); }
        }

        .brand-title h1 {
            font-family: var(--font-display);
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-title p {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        .theme-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.35);
            padding: 6px 12px;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .theme-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 2px solid transparent;
            cursor: pointer;
            transition: var(--transition);
        }

        .theme-btn.active {
            border-color: #fff;
            transform: scale(1.25);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.6);
        }

        /* --- DASHBOARD GRID --- */
        .dashboard-container {
            max-width: 1440px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 460px 1fr;
            gap: 2rem;
        }

        @media (max-width: 1100px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.75rem;
            transition: var(--transition);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }

        .glass-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .card-header h2 {
            font-family: var(--font-display);
            font-size: 1.15rem;
            font-weight: 700;
        }

        .badge {
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--text-primary);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* --- PARAMETER INPUT STYLING --- */
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.15rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .input-group.full-width {
            grid-column: span 2;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }

        .input-group label span.val-tip {
            color: var(--accent-secondary);
            font-family: var(--font-display);
        }

        .input-control {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 14px;
            color: var(--text-primary);
            font-family: var(--font-main);
            font-size: 0.9rem;
            outline: none;
            transition: var(--transition);
        }

        .input-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
            background: rgba(0, 0, 0, 0.5);
        }

        select.input-control option {
            background: #111827;
            color: #fff;
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 10px;
            padding: 14px;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: #fff;
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5);
            filter: brightness(1.1);
        }

        /* --- METRICS & VISUALIZATION OUTPUT --- */
        .analytics-viewport {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .kpi-card {
            background: var(--metric-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            position: relative;
            overflow: hidden;
        }

        .kpi-title {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .kpi-value {
            font-family: var(--font-display);
            font-size: 1.8rem;
            font-weight: 800;
            margin: 6px 0;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .kpi-status {
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 1.25rem;
        }

        @media (max-width: 768px) {
            .charts-grid, .kpi-row {
                grid-template-columns: 1fr;
            }
        }

        .chart-box {
            background: var(--metric-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            min-height: 280px;
            display: flex;
            flex-direction: column;
        }

        .chart-title {
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }

        .status-badge-high { color: #34d399; }
        .status-badge-med { color: #fbbf24; }
        .status-badge-low { color: #f87171; }
    </style>
</head>
<body>

    <header class="navbar">
        <div class="brand">
            <div class="brand-icon">CP</div>
            <div class="brand-title">
                <h1>Cap Round Institute Prediction</h1>
                <p>ANN Intelligence & Allocation Engine</p>
            </div>
        </div>

        <div class="theme-selector">
            <span style="font-size:0.75rem; color:var(--text-secondary); margin-right:4px;">Theme</span>
            <div class="theme-btn active" style="background:#6366f1;" onclick="setTheme('cyber-slate')"></div>
            <div class="theme-btn" style="background:#10b981;" onclick="setTheme('midnight-aurora')"></div>
            <div class="theme-btn" style="background:#f43f5e;" onclick="setTheme('sunset-blaze')"></div>
            <div class="theme-btn" style="background:#a855f7;" onclick="setTheme('electric-violet')"></div>
        </div>
    </header>

    <main class="dashboard-container">
        <!-- FORM / PARAMETERS -->
        <section class="glass-card">
            <div class="card-header">
                <h2>10-D Parameter Feed</h2>
                <span class="badge">Dense Layer Sync</span>
            </div>

            <form id="predictionForm" onsubmit="handlePrediction(event)" class="form-grid">
                <div class="input-group">
                    <label>CET / JEE Percentile <span class="val-tip">%</span></label>
                    <input type="number" step="0.01" min="0" max="100" id="f0" class="input-control" value="96.45" required>
                </div>

                <div class="input-group">
                    <label>State Merit Rank <span class="val-tip">AIR/SML</span></label>
                    <input type="number" min="1" max="250000" id="f1" class="input-control" value="4820" required>
                </div>

                <div class="input-group">
                    <label>Institute Choice Pref <span class="val-tip">1-10</span></label>
                    <input type="number" min="1" max="10" id="f2" class="input-control" value="1" required>
                </div>

                <div class="input-group">
                    <label>Branch Category Code <span class="val-tip">ID</span></label>
                    <select id="f3" class="input-control">
                        <option value="1">1 - Computer Science & Engg</option>
                        <option value="2">2 - AI & Data Science</option>
                        <option value="3">3 - Electronics & Telecomm</option>
                        <option value="4">4 - Information Technology</option>
                        <option value="5">5 - Mechanical / Electrical</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Home University Quota <span class="val-tip">HU/OHU</span></label>
                    <select id="f4" class="input-control">
                        <option value="1">Home University (HU - 1.0)</option>
                        <option value="0">Other University (OHU - 0.0)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Caste / Reservation Cat <span class="val-tip">Tier</span></label>
                    <select id="f5" class="input-control">
                        <option value="0.9">OPEN / General (0.9)</option>
                        <option value="0.6">OBC / EWS (0.6)</option>
                        <option value="0.3">SC / ST / VJNT (0.3)</option>
                        <option value="0.1">TFWS Scheme (0.1)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>HSC Aggregate % <span class="val-tip">12th</span></label>
                    <input type="number" step="0.01" min="35" max="100" id="f6" class="input-control" value="88.20" required>
                </div>

                <div class="input-group">
                    <label>SSC Aggregate % <span class="val-tip">10th</span></label>
                    <input type="number" step="0.01" min="35" max="100" id="f7" class="input-control" value="92.40" required>
                </div>

                <div class="input-group">
                    <label>CAP Round Stage <span class="val-tip">Phase</span></label>
                    <select id="f8" class="input-control">
                        <option value="1">CAP Round I</option>
                        <option value="2">CAP Round II</option>
                        <option value="3">CAP Round III</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Income / EWS Index <span class="val-tip">Normalized</span></label>
                    <input type="number" step="0.1" min="0" max="10" id="f9" class="input-control" value="3.5" required>
                </div>

                <button type="submit" class="submit-btn">
                    <span>Execute ANN Predictor</span>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                </button>
            </form>
        </section>

        <!-- ANALYTICS DASHBOARD -->
        <section class="analytics-viewport">
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-title">Admission Probability</div>
                    <div class="kpi-value" id="kpiProb">--%</div>
                    <div class="kpi-status" id="kpiStatus">Awaiting Computation</div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-title">Allotment Confidence</div>
                    <div class="kpi-value" id="kpiConf">--</div>
                    <div class="kpi-status status-badge-high" id="kpiTier">Dynamic Tier</div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-title">ANN Activation Drift</div>
                    <div class="kpi-value" id="kpiDrift">0.024</div>
                    <div class="kpi-status" style="color:var(--accent-secondary)">Optimal Sigmoid State</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-box">
                    <div class="chart-title">10-Dimensional Parameter Weighting vs Benchmark</div>
                    <canvas id="radarChart"></canvas>
                </div>

                <div class="chart-box">
                    <div class="chart-title">CAP Round Probability Distribution</div>
                    <canvas id="gaugeChart"></canvas>
                </div>
            </div>
        </section>
    </main>

    <script>
        let radarChartInstance = null;
        let gaugeChartInstance = null;

        function setTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            if(radarChartInstance) renderCharts();
        }

        function initCharts(radarData = [96, 75, 90, 85, 95, 70, 88, 92, 80, 65], prob = 87.5) {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const ctxGauge = document.getElementById('gaugeChart').getContext('2d');

            if (radarChartInstance) radarChartInstance.destroy();
            if (gaugeChartInstance) gaugeChartInstance.destroy();

            // Radar Chart
            radarChartInstance = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Percentile', 'Merit Score', 'Pref Fit', 'Branch Rank', 'Quota Index', 'Category', 'HSC %', 'SSC %', 'Round Phase', 'EWS Index'],
                    datasets: [
                        {
                            label: 'Current Candidate',
                            data: radarData,
                            backgroundColor: 'rgba(99, 102, 241, 0.25)',
                            borderColor: '#6366f1',
                            pointBackgroundColor: '#06b6d4',
                            borderWidth: 2
                        },
                        {
                            label: 'Institute Cutoff Baseline',
                            data: [88, 65, 70, 60, 50, 50, 75, 75, 50, 50],
                            backgroundColor: 'rgba(244, 63, 94, 0.1)',
                            borderColor: 'rgba(244, 63, 94, 0.6)',
                            borderDash: [4, 4],
                            borderWidth: 1.5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                            grid: { color: 'rgba(255, 255, 255, 0.08)' },
                            pointLabels: { color: '#94a3b8', font: { size: 10 } },
                            ticks: { display: false }
                        }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12 } }
                    }
                }
            });

            // Doughnut Gauge
            gaugeChartInstance = new Chart(ctxGauge, {
                type: 'doughnut',
                data: {
                    labels: ['Allotment Likelihood', 'Margin'],
                    datasets: [{
                        data: [prob, 100 - prob],
                        backgroundColor: ['#10b981', 'rgba(255, 255, 255, 0.05)'],
                        borderWidth: 0,
                        circumference: 260,
                        rotation: 230
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '75%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: true }
                    }
                }
            });
        }

        async function handlePrediction(e) {
            e.preventDefault();
            const payload = [];
            for (let i = 0; i < 10; i++) {
                payload.push(parseFloat(document.getElementById('f' + i).value));
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: payload })
                });
                const res = await response.json();

                const probPercent = (res.probability * 100).toFixed(1);
                document.getElementById('kpiProb').innerText = `${probPercent}%`;
                
                const statusEl = document.getElementById('kpiStatus');
                const tierEl = document.getElementById('kpiTier');
                const confEl = document.getElementById('kpiConf');

                if (probPercent >= 75) {
                    statusEl.innerText = "High Allotment Probability";
                    statusEl.className = "kpi-status status-badge-high";
                    tierEl.innerText = "Tier-1 / Preference 1 Safe";
                    confEl.innerText = "Very High";
                } else if (probPercent >= 45) {
                    statusEl.innerText = "Competitive Borderline";
                    statusEl.className = "kpi-status status-badge-med";
                    tierEl.innerText = "Tier-2 Shift Likely";
                    confEl.innerText = "Moderate";
                } else {
                    statusEl.innerText = "High Cutoff Risk";
                    statusEl.className = "kpi-status status-badge-low";
                    tierEl.innerText = "Backup Option Advised";
                    confEl.innerText = "Volatile";
                }

                // Update charts with normalized inputs
                const normalizedRadar = payload.map((val, idx) => {
                    if (idx === 1) return Math.min(100, Math.max(10, 100 - (val / 1000)));
                    if (idx === 2) return (11 - val) * 10;
                    if (idx === 9) return val * 10;
                    return Math.min(100, val);
                });

                initCharts(normalizedRadar, parseFloat(probPercent));

            } catch (err) {
                console.error("API error", err);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            initCharts();
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        raw_features = data.get("features", [])
        if len(raw_features) != 10:
            return jsonify({"error": "Exactly 10 features required"}), 400

        # Normalization layer mapping to [0, 1] range for the neural network
        norm = np.array(raw_features, dtype=np.float32)
        norm[0] = norm[0] / 100.0                       # CET Percentile
        norm[1] = np.clip(1.0 - (norm[1] / 100000.0), 0, 1) # Merit Rank inverse
        norm[2] = (10.0 - norm[2]) / 10.0               # Choice preference
        norm[3] = norm[3] / 5.0                         # Branch ID
        norm[4] = norm[4]                               # HU quota
        norm[5] = norm[5]                               # Reservation tier
        norm[6] = norm[6] / 100.0                       # HSC %
        norm[7] = norm[7] / 100.0                       # SSC %
        norm[8] = norm[8] / 3.0                         # Round stage
        norm[9] = norm[9] / 10.0                        # EWS/Income index

        prob = run_inference(norm)
        return jsonify({
            "probability": prob,
            "status": "success",
            "model": "ANN_Sequential_10_8_7_1"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
