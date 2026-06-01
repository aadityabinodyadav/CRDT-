# CRDT Mesh Network — IEEE Evaluation Suite

A complete, reproducible research toolkit for evaluating CRDT-based synchronisation
over infrastructure-less wireless mesh networks.

Implements and compares four forwarding protocols:

| Protocol     | Description |
|--------------|-------------|
| **Flood**    | Epidemic broadcast — maximum reachability baseline |
| **Heuristic**| Score-based relay selection (link quality + energy + buffer) |
| **AI**       | Logistic-regression classifier trained on heuristic traces |
| **AODV**     | Reactive unicast-to-all ns-3 baseline |

---

## Quick Start (Python simulator — no ns-3 required)

```bash
pip install -r requirements.txt

# Train + run all 4 scenarios, 3 seeds each
python main.py --skip-docker

# Or use the runner directly with more control
python -m experiments.runner --seeds 5 --nodes 30 --scenarios all
```

This produces `simulation_results.csv` and plots in `plots/`.

---

## Full ns-3 Pipeline (Docker)

```bash
# Build the ns-3 Docker image and run all scenarios
python main.py

# Fine-grained control
python main.py --nodes 30 --sim-time 60 --seeds 5 --output-dir results/
```

Requirements: Docker Desktop ≥ 24.

---

## Dashboard

Open `dashboard.html` in any browser and load a `simulation_results.csv` to
visualise PDR, latency, energy, and overhead across all protocols and scenarios.

---

## Project Layout

```
crdt-mesh-aplus/
├── main.py                  # Top-level CLI pipeline (ns-3 + Python)
├── dashboard.html           # Interactive results viewer
├── requirements.txt
├── Dockerfile               # ns-3.37 build image
│
├── simulator/               # Pure-Python discrete-event simulator
│   ├── engine.py            # Priority-queue event kernel
│   ├── network.py           # Simulation orchestrator + NetworkConfig
│   ├── node.py              # ConstrainedNode + EnergyModel (ESP32)
│   ├── crdt.py              # HLC, OR-Set, LWW-Register, G-Counter
│   ├── forwarding.py        # FloodStrategy / HeuristicStrategy / AIStrategy
│   ├── wireless.py          # Path-loss channel + SINR model
│   └── metrics.py           # MetricsCollector (PDR, latency, energy, …)
│
├── ml/
│   └── pipeline.py          # MLPipeline: train, cross-validate, export
│
├── experiments/
│   └── runner.py            # ExperimentRunner: 4 IEEE scenarios + CLI
│
├── analysis/
│   ├── parse_ns3.py         # ns-3 CSV trace parser
│   └── plotting.py          # Academic bar/significance plots
│
└── scratch/
    └── crdt_mesh.cc         # ns-3 C++ simulation source
```

---

## Bugs Fixed in This Version

| File | Bug | Fix |
|------|-----|-----|
| `experiments/runner.py` | `\\n` literal instead of newline in f-strings | Replaced with actual `\n` |
| `experiments/runner.py` | Missing `run_scenario_4_scalability` | Added with node-count sweep |
| `ml/pipeline.py` | `os.makedirs(os.path.dirname("weights.json"))` crashes on bare filename | Guard with `Path.parent` check |
| `main.py` | No CLI — impossible to skip Docker or vary parameters | Added `argparse` with `--skip-docker`, `--nodes`, `--sim-time`, etc. |
| `main.py` | `runner.save_results()` never called in Python path | Called with configurable output path |
| `simulator/forwarding.py` | `HeuristicStrategy` logged only 8 features; `AIStrategy` trains on 12 — training traces were misaligned | Extended to full 12-feature vector matching `AIStrategy._extract_features` |
| `simulator/forwarding.py` / `main.py` / `ml/pipeline.py` | Fallback weights had 8 entries; `AIStrategy` expects 12 — `zip` silently truncated 4 features | Extended all fallback weight lists to 12 entries |
| `main.py` | `--skip-train` logic inverted: training always ran, result was sometimes discarded | Fixed branch to `None if args.skip_train else runner.generate_ai_training_data()` |
| `simulator/crdt.py` | `_gc_oldest()` only cleared half of tombstoned entries per call — `_remove_set` grew unboundedly under memory pressure | Compact all fully-tombstoned entries unconditionally |
| `ml/pipeline.py` | Trained tree model not serializable for ns-3/Docker path — AI mode only worked in-process | Added `export_tree_model` / `load_tree_model` (pickle) |

---

## Key Metrics

- **PDR** — Packet Delivery Ratio (unique receptions / possible receptions)
- **Latency** — Mean, median, P95, std of end-to-end delay
- **Energy** — Mean per-node consumption (J), modelled on ESP32 datasheet
- **Overhead** — (control + duplicate bytes) / data bytes
- **Convergence** — Time to global CRDT state agreement

## References

Shapiro, M. et al. (2011). Conflict-Free Replicated Data Types. SSS 2011.  
Kulkarni, S. et al. (2014). Logical Physical Clocks. OPODIS 2014.  
Vahdat, A. & Becker, D. (2000). Epidemic Routing. Duke CS Tech Report.
