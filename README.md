# CRDT Mesh Network — Research Evaluation Suite (v13)

**Paper:** *Deterministic and Learned Resilience in CRDT-Based Wireless Mesh Networks Under Physical Degradation: Design, Evaluation, and Phase-Transition Analysis*

A complete, reproducible research toolkit implementing and evaluating four forwarding strategies for CRDT-based infrastructure-less wireless mesh networks.

---

## What This Code Does

The simulator models a wireless mesh network where nodes exchange messages using a Conflict-Free Replicated Data Type (OR-Set). Four forwarding strategies are compared:

| Strategy | Description |
|---|---|
| **Flood** | Epidemic broadcast to all neighbors — maximum reachability baseline |
| **Heuristic** | Score-based relay selection (link quality × energy × buffer × freshness) |
| **Adaptive** | Two-state controller: floods in sparse regime, uses heuristic in dense regime |
| **AI** | 12-feature logistic regression classifier trained on heuristic traces |

The evaluation sweeps path-loss exponents (n=2.0–4.5), partition durations (15–120s), network sizes (20–100 nodes), and two mobility models (Random Waypoint, Gauss-Markov).

---

## System Requirements

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (comes with Python)
- **~500 MB disk** for dependencies
- **Docker Desktop ≥ 24** — only needed for the optional ns-3 path (most users skip this)
- **RAM**: 2 GB minimum, 4 GB recommended for 100-node runs

---

## Installation

### Step 1 — Clone or unzip the project

```bash
unzip crdt-mesh-v13.zip
cd crdt-v13
```

### Step 2 — Create a virtual environment (strongly recommended)

```bash
# Create
python3 -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate — Windows CMD
venv\Scripts\activate.bat
```

> **Why a venv?** Keeps dependencies isolated. Without it, pip installs into your system Python and can cause version conflicts.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `tqdm`.

Expected install time: 1–3 minutes on a typical connection.

---

## Running the Simulator

### Quickest start — Python-only, no Docker needed

```bash
python main.py --skip-docker
```

This will:
1. Train the ML classifier on heuristic simulation traces (~60s)
2. Run all 4 scenarios across default seeds
3. Save `simulation_results.csv` in the current directory
4. Print a summary table

Expected runtime: **3–8 minutes** on a modern laptop at default settings (20 nodes, 3 seeds).

---

### Controlling the run

```bash
# More seeds = more statistically robust results (slower)
python main.py --skip-docker --seeds 10

# Different node counts
python main.py --skip-docker --nodes 50

# Skip ML training (uses pre-set fallback weights — fast testing)
python main.py --skip-docker --skip-train

# Save outputs to a specific folder
python main.py --skip-docker --output-dir results/run1/
```

---

### Running individual experiment scenarios

```bash
# Run the experiment runner directly with more control
python -m experiments.runner --seeds 5 --nodes 20 --scenarios all

# Specific scenario only
python -m experiments.runner --scenarios partition --seeds 3
```

---

### Running only the ML pipeline

```bash
python3 << 'EOF'
from simulator.network import NetworkConfig, MeshNetwork
from ml.pipeline import MLPipeline

cfg = NetworkConfig(n_nodes=20, sim_duration=300.0)
cfg.forwarding_mode = "heuristic"
net = MeshNetwork(cfg)
net.run()

traces = net.metrics.forwarding_traces
print(f"Collected {len(traces)} traces")

pipeline = MLPipeline(traces)
result = pipeline.train_logistic_regression()
print(f"AUC (hold-out): {result['auc_holdout']:.4f}")
print(f"AUC (5-fold CV): {result['auc_cv_mean']:.4f}")
EOF
```

---

### Running the channel sweep experiment

```bash
python3 << 'EOF'
import json
from experiments.runner import ExperimentRunner
from simulator.network import NetworkConfig

cfg = NetworkConfig(n_nodes=20, sim_duration=120.0)
runner = ExperimentRunner(cfg, n_seeds=3)
ai_weights = runner.generate_ai_training_data()

# Results will be in runner.results
for r in runner.results[:3]:
    print(r)
EOF
```

---

### Running unit tests

```bash
# Install pytest first
pip install pytest

# Run all 26 tests
python -m pytest tests/ -v

# Run only CRDT correctness tests
python -m pytest tests/test_crdt.py -v
```

Expected output: **26 passed** in ~5 seconds.

---

## Viewing Results

### Dashboard (browser, no server needed)

```bash
# Open the dashboard in your browser
open dashboard.html          # macOS
start dashboard.html         # Windows
xdg-open dashboard.html      # Linux
```

Then click "Load CSV" and select `simulation_results.csv`.

The dashboard shows PDR, latency, energy, and overhead across all protocols and scenarios interactively.

### Raw CSV

`simulation_results.csv` has one row per (scenario, strategy, seed) with columns:

| Column | Description |
|---|---|
| `scenario` | Scenario name (Static, Partition, Mobility, Scalability) |
| `forwarding` | Strategy name (flood / heuristic / ai / adaptive) |
| `seed` | Random seed used |
| `pdr` | Packet Delivery Ratio (0–1) |
| `latency_median` | Median end-to-end latency (ms) |
| `energy_mean` | Mean per-node energy consumption (J) |
| `total_overhead` | (control + duplicate bytes) / data bytes |
| `convergence_time` | Time to global CRDT agreement (s) |

---

## Optional: ns-3 Docker Path

> **Only needed if you want the full ns-3 C++ simulation.** Most users use the Python path above.

### Prerequisites

- Docker Desktop ≥ 24 installed and running
- ~4 GB free disk for the ns-3 image build (~20 min first time)

```bash
# Build ns-3 image and run all scenarios
python main.py

# With custom parameters
python main.py --nodes 30 --sim-time 60 --seeds 5 --output-dir results/ns3/
```

The ns-3 simulation source is in `scratch/crdt_mesh.cc`. The Docker image is built from `Dockerfile` using ns-3.37.

---

## Project Structure

```
crdt-v13/
│
├── main.py                    # Top-level CLI — run this first
├── requirements.txt           # Python dependencies
├── dashboard.html             # Interactive results viewer (open in browser)
├── simulation_results.csv     # Output from the last run
│
├── simulator/                 # Core discrete-event simulator
│   ├── engine.py              # Priority-queue event scheduler (O(log n))
│   ├── network.py             # MeshNetwork orchestrator + NetworkConfig
│   ├── node.py                # ConstrainedNode + EnergyModel (ESP32 profile)
│   ├── crdt.py                # HLC, OR-Set, LWW-Register, G-Counter, PN-Counter
│   ├── forwarding.py          # FloodStrategy / HeuristicStrategy / AIStrategy / AdaptiveStrategy
│   ├── wireless.py            # Log-distance path-loss channel + SINR delivery model
│   └── metrics.py             # MetricsCollector (PDR, latency, energy, forwarding traces)
│
├── ml/
│   └── pipeline.py            # MLPipeline: feature prep, logistic regression, CV, export
│
├── experiments/
│   └── runner.py              # ExperimentRunner: 4 scenarios, seed loops, CSV save
│
├── analysis/
│   ├── parse_ns3.py           # ns-3 CSV trace parser
│   └── plotting.py            # Academic publication-style plots
│
├── tests/
│   └── test_crdt.py           # 26 unit tests: CRDT axioms, HLC, OR-Set semantics
│
├── paper/
│   ├── main.tex               # LaTeX paper source
│   ├── references.bib         # BibTeX bibliography
│   └── figs/                  # PDF figures (fig1–fig5)
│
├── esp32/
│   ├── crdt_mesh_esp32.ino    # Arduino sketch for ESP32 deployment
│   └── HARDWARE_CRITIQUE.md   # Hardware deployment notes
│
├── scratch/
│   └── crdt_mesh.cc           # ns-3 C++ simulation source
│
└── Dockerfile                 # ns-3.37 build image
```

---

## Compiling the Paper (LaTeX)

```bash
cd paper/

# Compile (run twice for cross-references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or with latexmk (handles all passes automatically)
latexmk -pdf main.tex
```

Requires a LaTeX distribution: [TeX Live](https://tug.org/texlive/) (Linux/macOS) or [MiKTeX](https://miktex.org/) (Windows).

The compiled PDF will be `paper/main.pdf`. All figures are pre-compiled as PDFs in `paper/figs/`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'sklearn'`
```bash
pip install scikit-learn
```

### `ModuleNotFoundError: No module named 'numpy'`
```bash
pip install -r requirements.txt
```

### Tests fail with import errors
Make sure you're running from the **project root** (the folder containing `main.py`):
```bash
cd crdt-v13
python -m pytest tests/ -v
```

### Docker build fails
Check that Docker Desktop is running. The build requires internet access to pull the ns-3 base image. This is optional — use `--skip-docker` to avoid Docker entirely.

### Simulation is very slow
Reduce nodes or seeds:
```bash
python main.py --skip-docker --nodes 10 --seeds 2
```

### `FileNotFoundError` when saving results
The `--output-dir` folder is created automatically. If you hit permission errors, try:
```bash
python main.py --skip-docker --output-dir ./results/
```

---

## Key Design Decisions

**Why logistic regression, not a neural network?**
Logistic regression is interpretable, deployable on embedded hardware (ESP32), and produces a calibrated probability score. The AUC=0.69 result is an honest measurement of what 1-hop local features can achieve — a neural network on the same features would not significantly improve AUC because the bottleneck is feature scope, not model capacity.

**Why is the AI strategy worse at n=3.5?**
The classifier is trained exclusively on well-connected (n≤3.0) topologies. At n=3.5, the graph transitions to sparse connectivity and the model's learned utility thresholds no longer transfer correctly. This is the topology-sensitivity effect quantified in the paper. It is not a bug — it is the key finding.

**Why does the adaptive strategy beat AI at n=3.5?**
The adaptive controller uses a simple rule: if fewer than 3 neighbors or mean success rate < 0.5, flood everything. This rule fires exactly at the transition region and is topology-aware by design. The ML model has no equivalent regime-switching mechanism.

**Why is flooding the PDR baseline, not the target?**
Flooding generates O(n²) duplicates. The goal is to match flooding's PDR with far less overhead. The heuristic achieves ~89% PDR with ~10pp fewer duplicates. The adaptive controller achieves ~93% PDR at similar overhead. The AI trades 2pp PDR for 3pp fewer duplicates in the nominal regime.

---

## Citation

If you use this code or paper, please cite:

```
Aaditya Binod Yadav. Deterministic and Learned Resilience in CRDT-Based
Wireless Mesh Networks Under Physical Degradation: Design, Evaluation,
and Phase-Transition Analysis. 2025.
```
#   C R D T -  
 