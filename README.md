# CRDT Mesh Network: Deterministic and Learned Resilience in Wireless Mesh Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Research Artifact](https://img.shields.io/badge/Research-Artifact-orange)]()

A reproducible research framework for studying forwarding strategies in infrastructure-less wireless mesh networks using Conflict-Free Replicated Data Types (CRDTs).

This repository accompanies the paper:

> **Deterministic and Learned Resilience in CRDT-Based Wireless Mesh Networks Under Physical Degradation: Design, Evaluation, and Phase-Transition Analysis**
> Aaditya Binod Yadav (2025)

---

## Abstract

Infrastructure-less communication systems are critical in disaster response, remote environments, and disrupted network conditions where centralized connectivity is unavailable.

This project investigates how different forwarding strategies affect CRDT state propagation in wireless mesh networks under varying physical-layer degradation.

Four forwarding approaches are evaluated:

* Flooding
* Heuristic Relay Selection
* Adaptive Regime Switching
* Machine Learning-Based Relay Selection

The evaluation explores:

* Channel degradation
* Network partitioning
* Mobility
* Scalability

using a reproducible discrete-event simulation framework with optional ns-3 validation.

---

## Key Contributions

### CRDT-Based Mesh Synchronization

Supports:

* OR-Set
* LWW Register
* G-Counter
* PN-Counter
* Hybrid Logical Clocks (HLC)

### Forwarding Algorithms

| Strategy  | Purpose                         |
| --------- | ------------------------------- |
| Flood     | Reachability baseline           |
| Heuristic | Local utility-based forwarding  |
| Adaptive  | Topology-aware regime switching |
| AI        | Learned forwarding decisions    |

### Physical Layer Modeling

* Log-distance path loss
* SINR-based packet delivery
* Dynamic partitions
* Node mobility

### Machine Learning Pipeline

* Logistic Regression
* Feature Engineering
* Cross Validation
* AUC Evaluation
* Model Export

### Research Metrics

* Packet Delivery Ratio (PDR)
* Latency
* Energy Consumption
* Forwarding Overhead
* CRDT Convergence Time

---

# Architecture

```text
+-------------------+
|   Application     |
|      CRDTs        |
+-------------------+
          |
+-------------------+
| Forwarding Layer  |
| Flood / Heuristic |
| Adaptive / AI     |
+-------------------+
          |
+-------------------+
| Wireless Channel  |
| Path Loss + SINR  |
+-------------------+
          |
+-------------------+
| Event Simulator   |
+-------------------+
```

---

# Repository Structure

```text
crdt-v13/
│
├── main.py
├── requirements.txt
├── dashboard.html
│
├── simulator/
│   ├── engine.py
│   ├── network.py
│   ├── node.py
│   ├── crdt.py
│   ├── forwarding.py
│   ├── wireless.py
│   └── metrics.py
│
├── ml/
│   └── pipeline.py
│
├── experiments/
│   └── runner.py
│
├── analysis/
│   ├── plotting.py
│   └── parse_ns3.py
│
├── tests/
│   └── test_crdt.py
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figs/
│
├── esp32/
│   └── crdt_mesh_esp32.ino
│
├── scratch/
│   └── crdt_mesh.cc
│
└── Dockerfile
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/USERNAME/crdt-mesh-network.git

cd crdt-mesh-network
```

## Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Start

Run the complete Python simulation:

```bash
python main.py --skip-docker
```

This will:

1. Train the AI forwarding model
2. Execute all experiment scenarios
3. Generate evaluation results
4. Save results to CSV

Expected runtime:

```text
3–8 minutes
```

---

# Running Experiments

## Increase Statistical Confidence

```bash
python main.py --skip-docker --seeds 10
```

## Change Network Size

```bash
python main.py --skip-docker --nodes 50
```

## Skip Model Training

```bash
python main.py --skip-docker --skip-train
```

## Custom Output Directory

```bash
python main.py --skip-docker \
  --output-dir results/run1/
```

---

# Running Tests

Install pytest:

```bash
pip install pytest
```

Run all tests:

```bash
python -m pytest tests/ -v
```

Expected output:

```text
26 passed
```

---

# Results

The simulator reports:

| Metric           | Description                   |
| ---------------- | ----------------------------- |
| PDR              | Packet Delivery Ratio         |
| Latency          | End-to-End Delay              |
| Energy           | Mean Node Energy Consumption  |
| Overhead         | Duplicate and Control Traffic |
| Convergence Time | Time to Global CRDT Agreement |

Results are saved as:

```text
simulation_results.csv
```

---

# Interactive Dashboard

Open:

```bash
start dashboard.html
```

or

```bash
open dashboard.html
```

Load:

```text
simulation_results.csv
```

The dashboard provides interactive visualization of:

* PDR
* Latency
* Energy
* Overhead
* Convergence Time

---

# Optional ns-3 Validation

The repository includes an ns-3 implementation for validation against the Python simulator.

Build and run:

```bash
python main.py
```

Requirements:

* Docker Desktop
* Internet connection
* ~4 GB disk space

Source:

```text
scratch/crdt_mesh.cc
```

---

# Research Findings

The study identifies a phase-transition region around:

```text
Path-Loss Exponent n ≈ 3.5
```

where:

* Network connectivity rapidly degrades.
* Learned forwarding policies become topology-sensitive.
* Adaptive regime-switching outperforms purely learned policies.
* Flooding remains the upper bound on delivery ratio but incurs substantial overhead.

These observations motivate hybrid deterministic-learned forwarding systems for resilient infrastructure-less communication.

---

# Paper

Compile:

```bash
cd paper

latexmk -pdf main.tex
```

Output:

```text
paper/main.pdf
```

---

# Citation

```bibtex
@article{yadav2025crdtmesh,
  author = {Aaditya Binod Yadav},
  title = {Deterministic and Learned Resilience in CRDT-Based Wireless Mesh Networks Under Physical Degradation: Design, Evaluation, and Phase-Transition Analysis},
  year = {2025}
}
```

---

# License

MIT License

See:

```text
LICENSE
```

for details.
