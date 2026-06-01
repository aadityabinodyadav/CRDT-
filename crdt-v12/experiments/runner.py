"""
Experiment Runner for CRDT Mesh Simulator.

Defines and executes the standard IEEE evaluation scenarios:
1. Static Sparse Baseline
2. Partition & Recovery
3. Mobility Robustness
4. High-Load / Scalability

BUG FIXED: Replaced literal \\n escape sequences with actual newlines.
ADDED: run_scenario_4_scalability, __main__ entrypoint, save_results auto-call.
"""

import copy
import os
import pandas as pd
from typing import Dict, List, Any, Optional

from simulator.network import MeshNetwork, NetworkConfig
from ml.pipeline import MLPipeline


class ExperimentRunner:
    """Orchestrates running multiple simulation scenarios and seeds."""

    def __init__(self, base_config: NetworkConfig, n_seeds: int = 5):
        self.base_config = base_config
        self.n_seeds = n_seeds
        self.results: List[Dict[str, Any]] = []

    def _run_config(self, config: NetworkConfig, scenario_name: str) -> Dict:
        """Run a specific configuration across all seeds."""
        print(f"\n=== Running Scenario: {scenario_name} ({config.forwarding_mode}) ===")

        scenario_results = []

        for seed in range(42, 42 + self.n_seeds):
            print(f"  Seed {seed}...")
            run_config = copy.deepcopy(config)
            run_config.seed = seed

            network = MeshNetwork(run_config)
            stats = network.run()

            # Collapse results for dataframe
            flat_stats = {
                'scenario': scenario_name,
                'forwarding': config.forwarding_mode,
                'seed': seed,
                **stats
            }
            # Remove nested raw trace objects
            for key in ('node_stats', 'config'):
                flat_stats.pop(key, None)

            scenario_results.append(flat_stats)
            self.results.append(flat_stats)

        df = pd.DataFrame(scenario_results)
        return df.mean(numeric_only=True).to_dict()

    def generate_ai_training_data(self) -> Dict[str, Any]:
        """Run heuristic baseline to collect traces, train ML model, and return weights."""
        print("\n=== Generating AI Training Data ===")
        cfg = copy.deepcopy(self.base_config)
        cfg.forwarding_mode = "heuristic"
        cfg.sim_duration = 300.0  # long enough to get good traces

        network = MeshNetwork(cfg)
        network.run()

        traces = network.metrics.forwarding_traces
        print(f"Collected {len(traces)} forwarding traces.")

        pipeline = MLPipeline(traces)
        weights_dict = pipeline.train_logistic_regression()

        return weights_dict

    def run_scenario_1_static(self, ai_weights: Optional[Dict] = None):
        """Standard static network scenario."""
        cfg = copy.deepcopy(self.base_config)
        cfg.mobility_model = "static"

        for mode in ("flood", "heuristic"):
            c = copy.deepcopy(cfg)
            c.forwarding_mode = mode
            self._run_config(c, "Scenario 1: Static")

        if ai_weights:
            c = copy.deepcopy(cfg)
            c.forwarding_mode = "ai"
            c.ai_weights = ai_weights.get('weights')
            c.ai_bias = ai_weights.get('bias', 0.0)
            self._run_config(c, "Scenario 1: Static")

    def run_scenario_2_partition(self, ai_weights: Optional[Dict] = None):
        """Partition recovery scenario."""
        cfg = copy.deepcopy(self.base_config)
        cfg.enable_partitions = True
        cfg.partition_start_time = 100.0
        cfg.partition_duration = 60.0

        for mode in ("flood", "heuristic"):
            c = copy.deepcopy(cfg)
            c.forwarding_mode = mode
            self._run_config(c, "Scenario 2: Partition")

        if ai_weights:
            c = copy.deepcopy(cfg)
            c.forwarding_mode = "ai"
            c.ai_weights = ai_weights.get('weights')
            c.ai_bias = ai_weights.get('bias', 0.0)
            self._run_config(c, "Scenario 2: Partition")

    def run_scenario_3_mobility(self, ai_weights: Optional[Dict] = None):
        """Mobility scenario."""
        cfg = copy.deepcopy(self.base_config)
        cfg.mobility_model = "random_waypoint"
        cfg.mobility_speed_max = 5.0

        for mode in ("flood", "heuristic"):
            c = copy.deepcopy(cfg)
            c.forwarding_mode = mode
            self._run_config(c, "Scenario 3: Mobility")

        if ai_weights:
            c = copy.deepcopy(cfg)
            c.forwarding_mode = "ai"
            c.ai_weights = ai_weights.get('weights')
            c.ai_bias = ai_weights.get('bias', 0.0)
            self._run_config(c, "Scenario 3: Mobility")

    def run_scenario_4_scalability(self, ai_weights: Optional[Dict] = None):
        """
        High-Load / Scalability scenario.

        Sweeps node count [10, 20, 30, 50] to characterise how each
        forwarding mode scales with network size.  Static topology,
        message rate held constant per node.
        """
        print("\n=== Scenario 4: Scalability Sweep ===")
        node_counts = [10, 20, 30, 50]

        for n in node_counts:
            cfg = copy.deepcopy(self.base_config)
            cfg.n_nodes = n
            cfg.mobility_model = "static"
            scenario_label = f"Scenario 4: Scalability (n={n})"

            for mode in ("flood", "heuristic"):
                c = copy.deepcopy(cfg)
                c.forwarding_mode = mode
                self._run_config(c, scenario_label)

            if ai_weights:
                c = copy.deepcopy(cfg)
                c.forwarding_mode = "ai"
                c.ai_weights = ai_weights.get('weights')
                c.ai_bias = ai_weights.get('bias', 0.0)
                self._run_config(c, scenario_label)

    def save_results(self, filename: str = "simulation_results.csv") -> pd.DataFrame:
        """Export all aggregated results to CSV and return the DataFrame."""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}  ({len(df)} rows)")
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRDT Mesh Experiment Runner")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--nodes", type=int, default=20, help="Node count for base config")
    parser.add_argument("--duration", type=float, default=200.0, help="Sim duration (s)")
    parser.add_argument("--scenarios", nargs="+",
                        choices=["1", "2", "3", "4", "all"], default=["all"],
                        help="Scenarios to run")
    parser.add_argument("--output", default="simulation_results.csv")
    args = parser.parse_args()

    base_cfg = NetworkConfig(
        n_nodes=args.nodes,
        sim_duration=args.duration,
    )
    runner = ExperimentRunner(base_cfg, n_seeds=args.seeds)

    print("Generating AI training weights...")
    ai_w = runner.generate_ai_training_data()

    run_all = "all" in args.scenarios
    if run_all or "1" in args.scenarios:
        runner.run_scenario_1_static(ai_w)
    if run_all or "2" in args.scenarios:
        runner.run_scenario_2_partition(ai_w)
    if run_all or "3" in args.scenarios:
        runner.run_scenario_3_mobility(ai_w)
    if run_all or "4" in args.scenarios:
        runner.run_scenario_4_scalability(ai_w)

    runner.save_results(args.output)
    print("Done.")
