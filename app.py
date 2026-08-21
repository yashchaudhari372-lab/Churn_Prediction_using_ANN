"""
=====================================================================
 CAP ROUND INSTITUTE PREDICTION  —  AI Prediction Dashboard
=====================================================================
 Flask backend that loads a pickled Keras model (model.pkl) and
 serves a colorful, animated, multi-theme prediction dashboard.

 Model summary (auto-detected from the .pkl):
   Input  : 10 numeric features -> shape (None, 10)
   Layer1 : Dense(8,  activation="relu")
   Layer2 : Dense(7,  activation="relu")
   Layer3 : Dense(1,  activation="sigmoid")   -> binary probability

 Run:
   pip install -r requirements.txt
   python app.py
   -> open http://127.0.0.1:5000
=====================================================================
"""

import os
import pickle
import traceback
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ---------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
MODEL_FILENAMES = ["model.pkl", "model__2_.pkl", "model_2.pkl"]
MODEL = None
MODEL_LOAD_ERROR = None
MODEL_INPUT_SIZE = 10  # detected from the pickled Keras model


def load_model():
    """Load the pickled Keras model from disk (tries a few common names)."""
    global MODEL, MODEL_LOAD_ERROR
    base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = list(MODEL_FILENAMES)
    # also pick up any other .pkl sitting next to app.py
    for f in os.listdir(base_dir):
        if f.lower().endswith(".pkl") and f not in candidates:
            candidates.append(f)

    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    MODEL = pickle.load(f)
                print(f"[OK] Model loaded from '{name}'")
                return
            except Exception as e:
                MODEL_LOAD_ERROR = str(e)
                print(f"[WARN] Failed loading '{name}': {e}")

    MODEL_LOAD_ERROR = MODEL_LOAD_ERROR or "No model .pkl file found next to app.py"
    print(f"[ERROR] {MODEL_LOAD_ERROR}")


load_model()

FEATURE_LABELS = [f"Parameter {i + 1}" for i in range(MODEL_INPUT_SIZE)]


# ---------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------
def run_prediction(values):
    """Runs the Keras model on a list of numeric feature values."""
    if MODEL is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Model is not loaded")

    arr = np.array(values, dtype="float32").reshape(1, -1)
    raw = MODEL.predict(arr, verbose=0)
    prob = float(np.array(raw).reshape(-1)[0])
    prob = max(0.0, min(1.0, prob))
    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = prob if prob >= 0.5 else 1 - prob
    return {
        "probability": round(prob * 100, 2),
        "label": label,
        "confidence": round(confidence * 100, 2),
    }


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(
        PAGE_TEMPLATE,
        feature_labels=FEATURE_LABELS,
        model_ready=MODEL is not None,
        model_error=MODEL_LOAD_ERROR,
        year=datetime.now().year,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        values = data.get("features", [])

        if len(values) != MODEL_INPUT_SIZE:
            return jsonify({
                "success": False,
                "error": f"Expected {MODEL_INPUT_SIZE} feature values, got {len(values)}"
            }), 400

        values = [float(v) for v in values]
        result = run_prediction(values)
        result["success"] = True
        result["features"] = values
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "model_ready": MODEL is not None,
        "model_error": MODEL_LOAD_ERROR,
        "expected_inputs": MODEL_INPUT_SIZE,
    })


# ---------------------------------------------------------------
# Frontend Template (HTML + CSS + JS all in one)
# ---------------------------------------------------------------
PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cap Round Institute Prediction</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>

<style>
:root{
  --font-head:'Poppins', sans-serif;
  --font-body:'Inter', sans-serif;
  --radius:20px;
}

/* ============ THEME PALETTES ============ */
body.theme-aurora{
  --bg1:#0f0c29; --bg2:#302b63; --bg3:#24243e;
  --accent1:#00d4ff; --accent2:#7b2ff7; --accent3:#f107a3;
  --card-bg:rgba(255,255,255,0.06); --text-main:#f4f4ff; --text-sub:#b9b6d9;
  --good:#00e5a0; --bad:#ff4d6d;
}
body.theme-sunset{
  --bg1:#ff512f; --bg2:#dd2476; --bg3:#f77062;
  --accent1:#ffd166; --accent2:#ff6b6b; --accent3:#f72585;
  --card-bg:rgba(255,255,255,0.12); --text-main:#fffaf0; --text-sub:#ffe0d1;
  --good:#06d6a0; --bad:#ffd60a;
}
body.theme-ocean{
  --bg1:#005c97; --bg2:#363795; --bg3:#00c6ff;
  --accent1:#00f5d4; --accent2:#00bbf9; --accent3:#9b5de5;
  --card-bg:rgba(255,255,255,0.08); --text-main:#eafcff; --text-sub:#b6e5ff;
  --good:#00f5a0; --bad:#ff5d8f;
}
body.theme-emerald{
  --bg1:#0f2027; --bg2:#203a43; --bg3:#2c5364;
  --accent1:#38ef7d; --accent2:#11998e; --accent3:#a8ff78;
  --card-bg:rgba(255,255,255,0.07); --text-main:#eafff4; --text-sub:#b7ffd8;
  --good:#38ef7d; --bad:#ff6a6a;
}
body.theme-royal{
  --bg1:#1a1a2e; --bg2:#16213e; --bg3:#0f3460;
  --accent1:#e94560; --accent2:#ffbe0b; --accent3:#fb5607;
  --card-bg:rgba(255,255,255,0.06); --text-main:#fdf0ff; --text-sub:#c9c9f2;
  --good:#3ddc97; --bad:#ff477e;
}

*{margin:0; padding:0; box-sizing:border-box;}

body{
  font-family:var(--font-body);
  color:var(--text-main);
  min-height:100vh;
  background:linear-gradient(-45deg, var(--bg1), var(--bg2), var(--bg3), var(--bg1));
  background-size:400% 400%;
  animation:gradientShift 18s ease infinite;
  overflow-x:hidden;
  position:relative;
}

@keyframes gradientShift{
  0%{background-position:0% 50%;}
  50%{background-position:100% 50%;}
  100%{background-position:0% 50%;}
}

/* floating orbs */
.orb{
  position:fixed; border-radius:50%; filter:blur(60px); opacity:0.35; z-index:0;
  animation:float 12s ease-in-out infinite;
}
.orb1{width:320px; height:320px; background:var(--accent1); top:-80px; left:-80px;}
.orb2{width:260px; height:260px; background:var(--accent3); bottom:-60px; right:-60px; animation-delay:2s;}
.orb3{width:200px; height:200px; background:var(--accent2); top:40%; right:10%; animation-delay:4s;}

@keyframes float{
  0%,100%{transform:translateY(0) translateX(0);}
  50%{transform:translateY(-40px) translateX(20px);}
}

.wrap{position:relative; z-index:1; max-width:1300px; margin:0 auto; padding:28px 20px 60px;}

/* ============ HEADER ============ */
header{
  display:flex; justify-content:space-between; align-items:center;
  flex-wrap:wrap; gap:16px; margin-bottom:30px;
  animation:slideDown 0.7s ease;
}
@keyframes slideDown{from{opacity:0; transform:translateY(-24px);} to{opacity:1; transform:translateY(0);}}

.brand{display:flex; align-items:center; gap:14px;}
.brand-badge{
  width:54px; height:54px; border-radius:16px;
  background:linear-gradient(135deg, var(--accent1), var(--accent2));
  display:flex; align-items:center; justify-content:center;
  font-family:var(--font-head); font-weight:800; font-size:22px; color:#fff;
  box-shadow:0 8px 24px rgba(0,0,0,0.35);
  animation:pulseBadge 3s ease-in-out infinite;
}
@keyframes pulseBadge{0%,100%{transform:scale(1);} 50%{transform:scale(1.06);}}

.brand h1{
  font-family:var(--font-head); font-weight:800; font-size:26px;
  background:linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
  -webkit-background-clip:text; background-clip:text; color:transparent;
  background-size:200% auto; animation:shine 5s linear infinite;
  letter-spacing:0.5px;
}
@keyframes shine{to{background-position:200% center;}}
.brand p{font-size:12.5px; color:var(--text-sub); letter-spacing:1.5px; text-transform:uppercase; margin-top:2px;}

/* theme switcher */
.theme-switch{display:flex; gap:10px; align-items:center; background:var(--card-bg); padding:8px 12px; border-radius:50px; backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,0.15);}
.theme-dot{
  width:26px; height:26px; border-radius:50%; cursor:pointer; border:2px solid transparent;
  transition:all .25s ease; position:relative;
}
.theme-dot:hover{transform:scale(1.2);}
.theme-dot.active{border-color:#fff; box-shadow:0 0 0 3px rgba(255,255,255,0.25);}
.theme-dot[data-theme="aurora"]{background:linear-gradient(135deg,#00d4ff,#7b2ff7);}
.theme-dot[data-theme="sunset"]{background:linear-gradient(135deg,#ff512f,#dd2476);}
.theme-dot[data-theme="ocean"]{background:linear-gradient(135deg,#005c97,#00c6ff);}
.theme-dot[data-theme="emerald"]{background:linear-gradient(135deg,#11998e,#38ef7d);}
.theme-dot[data-theme="royal"]{background:linear-gradient(135deg,#e94560,#ffbe0b);}

/* ============ STATUS BANNER ============ */
.status-banner{
  padding:12px 20px; border-radius:14px; margin-bottom:24px; font-size:14px;
  display:flex; align-items:center; gap:10px; backdrop-filter:blur(10px);
  animation:fadeIn 0.9s ease;
}
.status-ok{background:rgba(56,239,125,0.12); border:1px solid rgba(56,239,125,0.4); color:#7CFFC4;}
.status-err{background:rgba(255,77,109,0.12); border:1px solid rgba(255,77,109,0.4); color:#ffb3c1;}

@keyframes fadeIn{from{opacity:0;} to{opacity:1;}}

/* ============ GRID LAYOUT ============ */
.grid{display:grid; grid-template-columns: 1.15fr 1fr; gap:24px;}
@media(max-width:960px){.grid{grid-template-columns:1fr;}}

.card{
  background:var(--card-bg);
  border:1px solid rgba(255,255,255,0.12);
  border-radius:var(--radius);
  padding:26px;
  backdrop-filter:blur(18px);
  box-shadow:0 20px 45px rgba(0,0,0,0.25);
  animation:riseIn 0.8s ease;
  position:relative;
  overflow:hidden;
}
@keyframes riseIn{from{opacity:0; transform:translateY(30px);} to{opacity:1; transform:translateY(0);}}

.card::before{
  content:""; position:absolute; top:-2px; left:-2px; right:-2px; height:4px;
  background:linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
  background-size:200% auto; animation:shine 4s linear infinite;
}

.card h2{
  font-family:var(--font-head); font-size:19px; font-weight:700; margin-bottom:6px;
  display:flex; align-items:center; gap:10px;
}
.card .sub{font-size:12.5px; color:var(--text-sub); margin-bottom:22px;}

/* ============ FORM ============ */
.param-grid{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
@media(max-width:520px){.param-grid{grid-template-columns:1fr;}}

.field{display:flex; flex-direction:column; gap:6px;}
.field label{font-size:12px; font-weight:600; color:var(--text-sub); display:flex; justify-content:space-between;}
.field label span.val{color:var(--accent1); font-weight:700;}
.field input[type="number"]{
  background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.16);
  border-radius:10px; padding:10px 12px; color:var(--text-main); font-size:14px;
  font-family:var(--font-body); outline:none; transition:.2s ease;
}
.field input[type="number"]:focus{border-color:var(--accent1); box-shadow:0 0 0 3px rgba(255,255,255,0.08);}
.field input[type="range"]{accent-color:var(--accent1); cursor:pointer;}

.actions{display:flex; gap:12px; margin-top:24px;}
.btn{
  border:none; border-radius:12px; padding:14px 22px; font-family:var(--font-head);
  font-weight:700; font-size:14.5px; cursor:pointer; transition:all .25s ease;
  display:flex; align-items:center; justify-content:center; gap:10px; flex:1;
}
.btn-primary{
  background:linear-gradient(135deg, var(--accent1), var(--accent2));
  color:#fff; box-shadow:0 10px 26px rgba(0,0,0,0.3);
}
.btn-primary:hover{transform:translateY(-3px); box-shadow:0 16px 32px rgba(0,0,0,0.4);}
.btn-primary:active{transform:translateY(0);}
.btn-ghost{
  background:rgba(255,255,255,0.08); color:var(--text-main); border:1px solid rgba(255,255,255,0.18); flex:0.5;
}
.btn-ghost:hover{background:rgba(255,255,255,0.15);}

.spinner{
  width:16px; height:16px; border-radius:50%;
  border:3px solid rgba(255,255,255,0.4); border-top-color:#fff;
  animation:spin 0.7s linear infinite; display:none;
}
@keyframes spin{to{transform:rotate(360deg);}}

/* ============ RESULT PANEL ============ */
.result-empty{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:40px 10px; color:var(--text-sub); min-height:320px;
}
.result-empty .icon{font-size:52px; margin-bottom:14px; opacity:0.6; animation:bob 3s ease-in-out infinite;}
@keyframes bob{0%,100%{transform:translateY(0);} 50%{transform:translateY(-10px);}}

.gauge-wrap{position:relative; width:230px; height:230px; margin:0 auto 18px;}
.gauge-center{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center;
}
.gauge-center .num{font-family:var(--font-head); font-size:40px; font-weight:800;}
.gauge-center .lbl{font-size:11px; color:var(--text-sub); letter-spacing:1px; text-transform:uppercase; margin-top:2px;}

.verdict-pill{
  display:inline-flex; align-items:center; gap:8px; padding:8px 18px; border-radius:50px;
  font-family:var(--font-head); font-weight:700; font-size:14px; margin:0 auto 20px; 
  animation:popIn 0.5s ease;
}
@keyframes popIn{from{opacity:0; transform:scale(0.7);} to{opacity:1; transform:scale(1);}}
.verdict-good{background:rgba(56,239,125,0.16); color:var(--good); border:1px solid var(--good);}
.verdict-bad{background:rgba(255,77,109,0.16); color:var(--bad); border:1px solid var(--bad);}

.metric-row{display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:20px 0;}
.metric-box{
  background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
  border-radius:14px; padding:14px 16px; text-align:center;
}
.metric-box .m-val{font-family:var(--font-head); font-size:22px; font-weight:800; color:var(--accent1);}
.metric-box .m-lbl{font-size:11px; color:var(--text-sub); text-transform:uppercase; letter-spacing:0.6px; margin-top:4px;}

.chart-box{background:rgba(255,255,255,0.05); border-radius:14px; padding:16px; margin-top:18px; border:1px solid rgba(255,255,255,0.1);}
.chart-box h3{font-size:13px; font-weight:600; margin-bottom:10px; color:var(--text-sub); text-transform:uppercase; letter-spacing:0.6px;}

footer{text-align:center; margin-top:40px; color:var(--text-sub); font-size:12.5px;}
footer b{color:var(--text-main);}

/* toast */
#toast{
  position:fixed; bottom:24px; right:24px; background:rgba(255,77,109,0.95); color:#fff;
  padding:14px 20px; border-radius:12px; font-size:13.5px; font-weight:600;
  box-shadow:0 10px 30px rgba(0,0,0,0.35); display:none; z-index:50; max-width:320px;
  animation:slideUp .4s ease;
}
@keyframes slideUp{from{opacity:0; transform:translateY(20px);} to{opacity:1; transform:translateY(0);}}
</style>
</head>

<body class="theme-aurora">
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-badge">CRI</div>
      <div>
        <h1>Cap Round Institute Prediction</h1>
        <p>AI-Powered Prediction Dashboard</p>
      </div>
    </div>

    <div class="theme-switch" id="themeSwitch">
      <span style="font-size:11px; color:var(--text-sub); margin-right:2px;">THEME</span>
      <div class="theme-dot active" data-theme="aurora" title="Aurora"></div>
      <div class="theme-dot" data-theme="sunset" title="Sunset"></div>
      <div class="theme-dot" data-theme="ocean" title="Ocean"></div>
      <div class="theme-dot" data-theme="emerald" title="Emerald"></div>
      <div class="theme-dot" data-theme="royal" title="Royal"></div>
    </div>
  </header>

  {% if model_ready %}
  <div class="status-banner status-ok">✅ Model loaded successfully — ready to generate predictions ({{ feature_labels|length }} input parameters expected).</div>
  {% else %}
  <div class="status-banner status-err">⚠️ Model could not be loaded ({{ model_error }}). Place your <b>model.pkl</b> file next to <b>app.py</b> and restart the server.</div>
  {% endif %}

  <div class="grid">

    <!-- INPUT CARD -->
    <div class="card">
      <h2>🎛️ Input Parameters</h2>
      <div class="sub">Enter the {{ feature_labels|length }} values required by the model, then run the prediction.</div>

      <form id="predictForm">
        <div class="param-grid" id="paramGrid">
          {% for label in feature_labels %}
          <div class="field">
            <label for="f{{ loop.index0 }}">{{ label }} <span class="val" id="v{{ loop.index0 }}">0.00</span></label>
            <input type="number" step="0.01" value="0" id="f{{ loop.index0 }}" data-idx="{{ loop.index0 }}" class="param-input" required>
            <input type="range" min="-5" max="5" step="0.01" value="0" class="param-range" data-target="f{{ loop.index0 }}">
          </div>
          {% endfor %}
        </div>

        <div class="actions">
          <button type="submit" class="btn btn-primary" id="predictBtn">
            <span class="spinner" id="spinner"></span>
            <span id="btnText">🚀 Run Prediction</span>
          </button>
          <button type="button" class="btn btn-ghost" id="randomBtn">🎲 Randomize</button>
          <button type="button" class="btn btn-ghost" id="resetBtn">↺ Reset</button>
        </div>
      </form>
    </div>

    <!-- RESULT CARD -->
    <div class="card">
      <h2>📊 Prediction Result</h2>
      <div class="sub">Live probability, confidence score &amp; visual breakdown.</div>

      <div id="resultEmpty" class="result-empty">
        <div class="icon">🔮</div>
        <div>Fill in the parameters and click <b>Run Prediction</b><br>to see the AI analysis here.</div>
      </div>

      <div id="resultContent" style="display:none;">
        <div class="gauge-wrap">
          <canvas id="gaugeChart"></canvas>
          <div class="gauge-center">
            <div class="num" id="gaugeNum">0%</div>
            <div class="lbl">Probability</div>
          </div>
        </div>

        <div style="text-align:center;">
          <div class="verdict-pill" id="verdictPill">—</div>
        </div>

        <div class="metric-row">
          <div class="metric-box">
            <div class="m-val" id="confVal">0%</div>
            <div class="m-lbl">Confidence</div>
          </div>
          <div class="metric-box">
            <div class="m-val" id="timeVal">0ms</div>
            <div class="m-lbl">Response Time</div>
          </div>
        </div>

        <div class="chart-box">
          <h3>Input Feature Overview</h3>
          <canvas id="featureChart" height="160"></canvas>
        </div>
      </div>
    </div>

  </div>

  <footer>© {{ year }} <b>Cap Round Institute</b> — Prediction Dashboard powered by Keras &amp; Flask</footer>
</div>

<div id="toast"></div>

<script>
const N_FEATURES = {{ feature_labels|length }};
let gaugeChart, featureChart;

/* ---------------- THEME SWITCHER ---------------- */
document.querySelectorAll('.theme-dot').forEach(dot=>{
  dot.addEventListener('click', ()=>{
    document.querySelectorAll('.theme-dot').forEach(d=>d.classList.remove('active'));
    dot.classList.add('active');
    document.body.className = 'theme-' + dot.dataset.theme;
    localStorage.setItem('cri_theme', dot.dataset.theme);
    if(gaugeChart) updateGaugeColors();
  });
});
(function(){
  const saved = localStorage.getItem('cri_theme');
  if(saved){
    document.body.className = 'theme-' + saved;
    document.querySelectorAll('.theme-dot').forEach(d=>{
      d.classList.toggle('active', d.dataset.theme === saved);
    });
  }
})();

/* ---------------- SLIDER <-> NUMBER SYNC ---------------- */
document.querySelectorAll('.param-range').forEach(range=>{
  const targetId = range.dataset.target;
  const numInput = document.getElementById(targetId);
  const idx = numInput.dataset.idx;
  const valSpan = document.getElementById('v'+idx);

  range.addEventListener('input', ()=>{
    numInput.value = range.value;
    valSpan.textContent = parseFloat(range.value).toFixed(2);
  });
  numInput.addEventListener('input', ()=>{
    const v = parseFloat(numInput.value || 0);
    range.value = Math.max(-5, Math.min(5, v));
    valSpan.textContent = v.toFixed(2);
  });
});

/* ---------------- RANDOMIZE / RESET ---------------- */
document.getElementById('randomBtn').addEventListener('click', ()=>{
  document.querySelectorAll('.param-input').forEach(inp=>{
    const v = (Math.random()*10 - 5).toFixed(2);
    inp.value = v;
    document.getElementById('v'+inp.dataset.idx).textContent = parseFloat(v).toFixed(2);
    const range = document.querySelector('.param-range[data-target="'+inp.id+'"]');
    if(range) range.value = v;
  });
});
document.getElementById('resetBtn').addEventListener('click', ()=>{
  document.querySelectorAll('.param-input').forEach(inp=>{
    inp.value = 0;
    document.getElementById('v'+inp.dataset.idx).textContent = "0.00";
    const range = document.querySelector('.param-range[data-target="'+inp.id+'"]');
    if(range) range.value = 0;
  });
});

/* ---------------- TOAST ---------------- */
function showToast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(()=>{ t.style.display='none'; }, 4000);
}

/* ---------------- FORM SUBMIT ---------------- */
document.getElementById('predictForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');

  const values = [];
  document.querySelectorAll('.param-input').forEach(inp=>{
    values.push(parseFloat(inp.value || 0));
  });

  spinner.style.display='inline-block';
  btnText.textContent = 'Predicting...';
  btn.disabled = true;
  const startTime = performance.now();

  try{
    const res = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({features: values})
    });
    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    if(!data.success){
      showToast('❌ ' + (data.error || 'Prediction failed'));
    } else {
      renderResult(data, elapsed);
    }
  }catch(err){
    showToast('❌ Network / server error: ' + err.message);
  }finally{
    spinner.style.display='none';
    btnText.textContent = '🚀 Run Prediction';
    btn.disabled = false;
  }
});

/* ---------------- RENDER RESULT ---------------- */
function renderResult(data, elapsed){
  document.getElementById('resultEmpty').style.display = 'none';
  const content = document.getElementById('resultContent');
  content.style.display = 'block';

  const prob = data.probability;
  const isGood = data.label === 'POSITIVE';

  document.getElementById('gaugeNum').textContent = prob + '%';
  document.getElementById('confVal').textContent = data.confidence + '%';
  document.getElementById('timeVal').textContent = elapsed + 'ms';

  const pill = document.getElementById('verdictPill');
  pill.textContent = (isGood ? '✅ POSITIVE OUTCOME' : '⛔ NEGATIVE OUTCOME') + ' — ' + prob + '%';
  pill.className = 'verdict-pill ' + (isGood ? 'verdict-good' : 'verdict-bad');

  drawGauge(prob, isGood);
  drawFeatureChart(data.features);
}

function getCSS(v){ return getComputedStyle(document.body).getPropertyValue(v).trim(); }

function drawGauge(prob, isGood){
  const ctx = document.getElementById('gaugeChart').getContext('2d');
  const color = isGood ? getCSS('--good') : getCSS('--bad');
  if(gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, {
    type:'doughnut',
    data:{
      datasets:[{
        data:[prob, 100-prob],
        backgroundColor:[color, 'rgba(255,255,255,0.08)'],
        borderWidth:0,
        cutout:'78%',
      }]
    },
    options:{
      responsive:true,
      maintainAspectRatio:true,
      animation:{animateRotate:true, duration:1200, easing:'easeOutCubic'},
      plugins:{legend:{display:false}, tooltip:{enabled:false}}
    }
  });
}

function updateGaugeColors(){
  // re-render on theme change if a result already exists
  const num = document.getElementById('gaugeNum');
  if(num && document.getElementById('resultContent').style.display !== 'none'){
    const prob = parseFloat(num.textContent);
    const isGood = document.getElementById('verdictPill').classList.contains('verdict-good');
    drawGauge(prob, isGood);
  }
}

function drawFeatureChart(features){
  const ctx = document.getElementById('featureChart').getContext('2d');
  const labels = features.map((_,i)=> 'P' + (i+1));
  const grad = ctx.createLinearGradient(0,0,0,160);
  grad.addColorStop(0, getCSS('--accent1'));
  grad.addColorStop(1, getCSS('--accent2'));

  if(featureChart) featureChart.destroy();
  featureChart = new Chart(ctx, {
    type:'bar',
    data:{
      labels: labels,
      datasets:[{
        label:'Value',
        data: features,
        backgroundColor: grad,
        borderRadius:8,
        maxBarThickness:28,
      }]
    },
    options:{
      responsive:true,
      animation:{duration:900, easing:'easeOutQuart'},
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color: getCSS('--text-sub')}, grid:{display:false}},
        y:{ticks:{color: getCSS('--text-sub')}, grid:{color:'rgba(255,255,255,0.08)'}}
      }
    }
  });
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
