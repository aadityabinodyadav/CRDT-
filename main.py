"""
Main executable: Full IEEE ns-3 evaluation pipeline.

Trains ML weights offline via the Python simulator, then runs all ns-3
scenarios (flood / heuristic / AI / AODV) inside Docker and collects traces.

IMPROVEMENTS over original:
  - Argparse CLI for flexible scenario control
  - Saves aggregated simulation_results.csv after Python-sim runs
  - export_weights path fix (dirname of bare filename would crash os.makedirs)
  - Graceful per-scenario failure (one bad run no longer aborts the pipeline)
  - Structured summary table printed at the end
"""

import argparse
import os
import subprocess
import sys
import time
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CRDT Mesh Network — Full IEEE Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-docker", action="store_true",
                   help="Skip Docker/ns-3 runs (Python-sim only)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip ML training; use fallback weights")
    p.add_argument("--seeds", type=int, default=3,
                   help="Seeds for Python simulator runs")
    p.add_argument("--nodes", type=int, default=20,
                   help="Number of nodes in each scenario")
    p.add_argument("--sim-time", type=float, default=30.0,
                   help="ns-3 simulation time (seconds)")
    p.add_argument("--tx-power", type=float, default=16.0,
                   help="ns-3 TX power (dBm)")
    p.add_argument("--output-dir", default=".",
                   help="Directory for CSV/plot outputs")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Docker helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_docker_sim(mode: str, sim_time: float, n_nodes: int,
                   ai_weights: str = "", use_aodv: bool = False,
                   tx_power: float = 16.0) -> bool:
    """Run a single ns-3 simulation scenario inside Docker."""
    run_args = (
        f"scratch/crdt_mesh --mode={mode} --simTime={sim_time} "
        f"--nNodes={n_nodes} --txPower={tx_power}"
    )
    if ai_weights:
        run_args += f" --aiWeights={ai_weights}"
    if use_aodv:
        run_args += " --useAodv=true"

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:/workspace",
        "ns3-crdt",
        "./ns3", "run", run_args,
    ]

    print(f"  > docker run: mode={mode} nodes={n_nodes} aodv={use_aodv}")
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
        snippet = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        print(snippet)
        if result.returncode != 0:
            print(f"  [WARN] stderr: {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [WARN] Simulation timed out.")
        return False
    except FileNotFoundError:
        print("  [WARN] Docker not found. Is Docker installed?")
        return False


def build_docker_image() -> bool:
    """Build the ns-3 Docker image. Returns True on success."""
    print("[2] Building ns-3 Docker image...")
    result = subprocess.run(
        ["docker", "build", "-t", "ns3-crdt", "."],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Docker build FAILED:\n{result.stderr[-800:]}")
        return False
    print("    Docker image built successfully.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ML training
# ──────────────────────────────────────────────────────────────────────────────

FALLBACK_WEIGHTS = "0.4,0.15,0.25,0.1,-0.08,0.12,-0.05,-0.03,0.08,0.20,-0.15,0.10"


def train_ml_weights(n_nodes: int, seeds: int) -> str:
    """Train ML weights using the Python simulator."""
    print("[1] Training ML weights via Python heuristic traces...")
    try:
        from simulator.network import NetworkConfig
        from experiments.runner import ExperimentRunner

        cfg = NetworkConfig(
            n_nodes=n_nodes,
            area_width=200.0,
            area_height=200.0,
            sim_duration=100.0,
        )
        runner = ExperimentRunner(cfg, n_seeds=seeds)
        weights_dict = runner.generate_ai_training_data()
        weights = weights_dict.get('weights', [])
        weights_str = ",".join([f"{w:.6f}" for w in weights])
        print(f"    Extracted {len(weights)} weights, "
              f"AUC={weights_dict.get('auc_holdout', weights_dict.get('auc', 'N/A'))}")

        # Export weights JSON to data/
        weights_path = Path("data") / "ai_weights.json"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_text(json.dumps(weights_dict, indent=2))
        print(f"    Weights exported to {weights_path}")

        return weights_str
    except Exception as e:
        print(f"    ML training failed: {e}")
        print(f"    Using fallback weights.")
        return FALLBACK_WEIGHTS


# ──────────────────────────────────────────────────────────────────────────────
# Python-only evaluation (no Docker)
# ──────────────────────────────────────────────────────────────────────────────

def run_python_evaluation(args) -> None:
    """Run full evaluation using the pure-Python simulator."""
    print("\n[PY] Running Python-simulator evaluation...")
    from simulator.network import NetworkConfig
    from experiments.runner import ExperimentRunner

    cfg = NetworkConfig(
        n_nodes=args.nodes,
        area_width=200.0,
        area_height=200.0,
        sim_duration=100.0,
    )
    runner = ExperimentRunner(cfg, n_seeds=args.seeds)

    ai_w = None if args.skip_train else runner.generate_ai_training_data()

    runner.run_scenario_1_static(ai_w)
    runner.run_scenario_2_partition(ai_w)
    runner.run_scenario_3_mobility(ai_w)
    runner.run_scenario_4_scalability(ai_w)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = runner.save_results(str(out_dir / "simulation_results.csv"))

    # Attempt to generate plots
    try:
        from analysis.plotting import generate_all_plots
        generate_all_plots(str(out_dir / "simulation_results.csv"))
    except Exception as e:
        print(f"  [WARN] Plotting skipped: {e}")

    _print_summary_table(df)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary_table(df) -> None:
    """Print a concise summary table of key metrics per forwarding mode."""
    import pandas as pd
    key_cols = [c for c in ['pdr', 'latency_median', 'energy_mean', 'total_overhead']
                if c in df.columns]
    if not key_cols:
        return

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY  (mean across seeds & scenarios)")
    print("=" * 70)
    summary = df.groupby('forwarding')[key_cols].mean()
    print(summary.to_string(float_format="{:.4f}".format))
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  CRDT Mesh Network — Full IEEE Evaluation Pipeline")
    print("=" * 70)

    start = time.time()

    if args.skip_docker:
        run_python_evaluation(args)
    else:
        # Step 1: ML weights
        if args.skip_train:
            weights_str = FALLBACK_WEIGHTS
            print("[1] Using fallback ML weights.")
        else:
            weights_str = train_ml_weights(args.nodes, args.seeds)

        # Step 2: Docker
        if not build_docker_image():
            print("\n[WARN] Docker build failed. Falling back to Python simulator.")
            run_python_evaluation(args)
        else:
            # Step 3: ns-3 scenarios
            print("\n[3] Running ns-3 simulation scenarios...")
            scenarios = [
                ("Flood baseline",        "flood",  False),
                ("Heuristic forwarding",  "heuristic", False),
                ("AI-assisted routing",   "ai",     False),
                ("AODV reactive routing", "flood",  True),
            ]

            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            for label, mode, aodv in scenarios:
                print(f"\n--- {label} ---")
                ok = run_docker_sim(
                    mode=mode,
                    sim_time=args.sim_time,
                    n_nodes=args.nodes,
                    ai_weights=weights_str if mode == "ai" else "",
                    use_aodv=aodv,
                    tx_power=args.tx_power,
                )
                if ok:
                    suffix = f"{mode}_aodv" if aodv else mode
                    for fname in ("ns3-traces.csv", "flow-stats.csv"):
                        src = Path(fname)
                        dst = out_dir / fname.replace(".csv", f"_{suffix}.csv")
                        if src.exists():
                            src.replace(dst)
                            print(f"    Saved {dst}")

            # Step 4: Post-process
            print("\n[4] Post-processing traces & generating statistics...")
            try:
                from analysis.parse_ns3 import main as parse_traces
                from analysis.plotting import generate_all_plots
                parse_traces()
                generate_all_plots(str(out_dir / "ns3_simulation_results.csv"))
            except Exception as e:
                print(f"  [ERROR] Post-processing failed: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    out_files = sorted(Path(args.output_dir).glob("*.csv"))
    if out_files:
        print("  Output CSV files:")
        for f in out_files:
            print(f"    - {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
