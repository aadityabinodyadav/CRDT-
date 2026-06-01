"""
Full experiment sweep for v14 paper update.
Covers: channel sweep (PL + shadowing), partition extended, latency/hops.
Uses multiprocessing to parallelize. Config confirmed to match paper:
  n_nodes=20, area=200x200, sim_duration=300s
"""
import json, statistics, time
from multiprocessing import Pool

from simulator.network import MeshNetwork, NetworkConfig

BASE = dict(n_nodes=20, sim_duration=300.0, area_width=200.0, area_height=200.0)
STRATEGIES = ["flood", "heuristic", "adaptive", "ai"]
SEEDS = [42, 43, 44, 45, 46]

# ── helpers ──────────────────────────────────────────────────────────

def ci95(vals):
    if len(vals) < 2:
        return 0.0
    import math
    s = statistics.stdev(vals)
    return round(1.96 * s / math.sqrt(len(vals)), 2)

def mean2(vals):
    return round(statistics.mean(vals), 2)

def run_one(args):
    kind, key, strat, seed, extra = args
    cfg = NetworkConfig()
    for k, v in BASE.items():
        setattr(cfg, k, v)
    for k, v in extra.items():
        setattr(cfg, k, v)
    cfg.forwarding_mode = strat
    cfg.seed = seed
    result = MeshNetwork(cfg).run()
    return kind, key, strat, seed, result

# ── task builders ─────────────────────────────────────────────────────

def pl_tasks():
    """Path-loss sweep: n_pl in [2.0,2.5,3.0,3.5,4.0,4.5]"""
    tasks = []
    for n_pl in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        for strat in STRATEGIES:
            for seed in SEEDS:
                tasks.append(("pl", n_pl, strat, seed,
                               {"path_loss_exponent": n_pl, "shadowing_std_db": 0.0}))
    return tasks

def sh_tasks():
    """Shadowing sweep: sigma in [0,4,8,10] dB, PL fixed at 3.0"""
    tasks = []
    for sigma in [0.0, 4.0, 8.0, 10.0]:
        for strat in STRATEGIES:
            for seed in SEEDS:
                tasks.append(("sh", sigma, strat, seed,
                               {"path_loss_exponent": 3.0, "shadowing_std_db": sigma}))
    return tasks

def partition_tasks():
    """Partition recovery sweep: duration in [15,30,60,120]s"""
    tasks = []
    for dur in [15, 30, 60, 120]:
        for strat in STRATEGIES:
            for seed in SEEDS:
                tasks.append(("part", dur, strat, seed,
                               {"enable_partitions": True,
                                "partition_start_time": 100.0,
                                "partition_duration": float(dur),
                                "path_loss_exponent": 3.0,
                                "shadowing_std_db": 0.0}))
    return tasks

def latency_tasks():
    """Latency/hops baseline: PL=3.0, no shadowing, single seed set"""
    tasks = []
    for strat in STRATEGIES:
        for seed in SEEDS:
            tasks.append(("lat", 0, strat, seed,
                           {"path_loss_exponent": 3.0, "shadowing_std_db": 0.0}))
    return tasks

# ── aggregation ───────────────────────────────────────────────────────

def aggregate_pdr(raw, kind):
    """Group raw results by (key, strat) → mean/ci95 PDR."""
    from collections import defaultdict
    groups = defaultdict(list)
    for k, key, strat, seed, res in raw:
        if k == kind:
            groups[(key, strat)].append(res["pdr"] * 100)
    out = {}
    keys = sorted(set(k for k, _ in groups))
    for key in keys:
        sk = str(key)
        out[sk] = {}
        for strat in STRATEGIES:
            vals = groups[(key, strat)]
            out[sk][strat] = {"mean": mean2(vals), "ci95": ci95(vals)}
    return out

def aggregate_partition(raw):
    from collections import defaultdict
    groups = defaultdict(list)
    for k, key, strat, seed, res in raw:
        if k == "part":
            groups[(key, strat)].append(res["pdr"] * 100)
    out = {}
    for dur in [15, 30, 60, 120]:
        out[str(dur)] = {}
        for strat in STRATEGIES:
            vals = groups[(dur, strat)]
            out[str(dur)][strat] = {"mean": mean2(vals), "ci95": ci95(vals)}
    return out

def aggregate_latency(raw):
    from collections import defaultdict
    # Collect all hop-latency pairs per strategy
    hop_lats = defaultdict(lambda: defaultdict(list))
    for k, key, strat, seed, res in raw:
        if k == "lat":
            hl = res.get("latency_by_hops", {})
            for hop, lats in hl.items():
                hop_lats[strat][hop].extend(lats)
    out = {}
    for strat in STRATEGIES:
        out[strat] = {}
        for hop in sorted(hop_lats[strat]):
            lats_ms = sorted([x * 1000 for x in hop_lats[strat][hop]])
            n = len(lats_ms)
            if n == 0:
                continue
            def pct(p):
                idx = min(int(p / 100 * n), n - 1)
                return round(lats_ms[idx], 1)
            out[strat][str(hop)] = {
                "count": n,
                "p50_ms": pct(50),
                "p90_ms": pct(90),
                "p99_ms": pct(99),
            }
    return out

# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_tasks = pl_tasks() + sh_tasks() + partition_tasks() + latency_tasks()
    print(f"Total tasks: {len(all_tasks)}  (~{len(all_tasks)*7/60:.0f} min wall-time with 8 workers)")

    t0 = time.time()
    with Pool(processes=8) as pool:
        raw = pool.map(run_one, all_tasks)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── PL sweep ──
    pl_results = aggregate_pdr(raw, "pl")
    with open("experiments/results_channel_sweep_pl.json", "w") as f:
        json.dump(pl_results, f, indent=2)
    print("Wrote results_channel_sweep_pl.json")

    # ── Shadowing sweep ──
    sh_results = aggregate_pdr(raw, "sh")
    with open("experiments/results_channel_sweep_sh.json", "w") as f:
        json.dump(sh_results, f, indent=2)
    print("Wrote results_channel_sweep_sh.json")

    # ── Combined channel sweep (paper table format) ──
    combined = {"path_loss": pl_results, "shadowing": sh_results}
    with open("experiments/results_channel_sweep.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("Wrote results_channel_sweep.json")

    # ── Partition extended ──
    part_results = aggregate_partition(raw)
    with open("experiments/results_partition_extended.json", "w") as f:
        json.dump(part_results, f, indent=2)
    print("Wrote results_partition_extended.json")

    # ── Latency/hops ──
    lat_results = aggregate_latency(raw)
    with open("experiments/results_latency_hops.json", "w") as f:
        json.dump(lat_results, f, indent=2)
    print("Wrote results_latency_hops.json")

    # ── Print summary ──
    print("\n=== CHANNEL SWEEP (PDR %) ===")
    print(f"{'n_pl':<6} {'flood':>8} {'heuristic':>10} {'adaptive':>10} {'ai':>8}")
    for pl in ["2.0","2.5","3.0","3.5","4.0","4.5"]:
        row = pl_results.get(pl, {})
        print(f"{pl:<6}", end="")
        for s in STRATEGIES:
            v = row.get(s, {})
            print(f"  {v.get('mean',0):>6.2f}±{v.get('ci95',0):<4.2f}", end="")
        print()

    print("\n=== SHADOWING SWEEP (PDR %) ===")
    print(f"{'sigma':<6} {'flood':>8} {'heuristic':>10} {'adaptive':>10} {'ai':>8}")
    for sig in ["0.0","4.0","8.0","10.0"]:
        row = sh_results.get(sig, {})
        print(f"{sig:<6}", end="")
        for s in STRATEGIES:
            v = row.get(s, {})
            print(f"  {v.get('mean',0):>6.2f}±{v.get('ci95',0):<4.2f}", end="")
        print()

    print("\n=== PARTITION EXTENDED (PDR %) ===")
    print(f"{'dur_s':<6} {'flood':>8} {'heuristic':>10} {'adaptive':>10} {'ai':>8}")
    for dur in ["15","30","60","120"]:
        row = part_results.get(dur, {})
        print(f"{dur:<6}", end="")
        for s in STRATEGIES:
            v = row.get(s, {})
            print(f"  {v.get('mean',0):>6.2f}±{v.get('ci95',0):<4.2f}", end="")
        print()
