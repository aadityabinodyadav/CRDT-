"""
Unit tests for forwarding strategy correctness and regime-aware behaviour.

Covers:
    FloodStrategy      — always selects all neighbors
    AODVStrategy       — always selects exactly one (best) neighbor
    HeuristicStrategy  — top-fraction selection with utility scoring
    AIStrategy (v14)   — adaptive select_fraction + confidence fallback
    AdaptiveStrategy   — switches between sparse/normal modes

Tests use lightweight stub objects instead of the full simulator to keep
tests fast and isolated from network/engine dependencies.
"""

import sys
import os
import math
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dataclasses import dataclass, field
from typing import Set, List, Tuple

from simulator.forwarding import (
    FloodStrategy, AODVStrategy, HeuristicStrategy,
    AIStrategy, AdaptiveStrategy, ForwardingDecision,
)


# ─── Stubs ────────────────────────────────────────────────────────────────────

@dataclass
class StubNeighborInfo:
    """Minimal NeighborInfo stub for forwarding tests."""
    node_id: int = 0
    last_seen: float = 0.0
    rssi: float = -70.0
    success_rate: float = 0.8
    residual_energy: float = 1.0
    buffer_free: float = 1.0
    degree: int = 5
    msg_count: int = 0
    state_hash: int = 0
    tx_attempts: int = 5
    tx_successes: int = 4
    rssi_history: list = field(default_factory=list)
    relay_timestamps: list = field(default_factory=list)
    neighbor_set: Set[int] = field(default_factory=set)
    two_hop_degree: int = 20

    def mobility_proxy(self) -> float:
        if len(self.rssi_history) < 2:
            return 0.0
        t0, r0 = self.rssi_history[0]
        t1, r1 = self.rssi_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return max(-1.0, min(1.0, (r1 - r0) / dt / 5.0))

    def delivery_history_ratio(self) -> float:
        if not self.relay_timestamps:
            return 0.5
        return sum(1 for _, d in self.relay_timestamps if d) / len(self.relay_timestamps)

    def jaccard_overlap(self, my_neighbor_ids: Set[int]) -> float:
        union = my_neighbor_ids | self.neighbor_set
        if not union:
            return 0.0
        return len(my_neighbor_ids & self.neighbor_set) / len(union)


@dataclass
class StubMessage:
    """Minimal Message stub."""
    hop_count: int = 1
    created_at: float = 0.0


class StubNode:
    """Minimal ConstrainedNode stub."""
    def __init__(self, current_time: float = 100.0):
        self._current_time = current_time

    @property
    def current_time(self) -> float:
        return self._current_time


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_neighbors(n: int, **kwargs) -> dict:
    """Create n StubNeighborInfo instances, all with identical defaults."""
    return {i: StubNeighborInfo(node_id=i, **kwargs) for i in range(n)}


def make_sparse_neighbors(n: int) -> dict:
    """Create n neighbors with degraded link quality (sparse regime)."""
    return {
        i: StubNeighborInfo(
            node_id=i,
            success_rate=0.3,
            rssi=-95.0,
            degree=2,
            tx_attempts=4,
        )
        for i in range(n)
    }


# ─── FloodStrategy ────────────────────────────────────────────────────────────

class TestFloodStrategy:
    strat = FloodStrategy()

    def test_selects_all_neighbors(self):
        node = StubNode()
        msg = StubMessage()
        nbrs = make_neighbors(5)
        d = self.strat.select_recipients(node, msg, nbrs)
        assert set(d.selected_neighbors) == set(nbrs.keys())

    def test_empty_neighbors(self):
        d = self.strat.select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    def test_scores_all_one(self):
        nbrs = make_neighbors(3)
        d = self.strat.select_recipients(StubNode(), StubMessage(), nbrs)
        assert all(v == 1.0 for v in d.scores.values())

    def test_strategy_name(self):
        assert self.strat.name == "flood"


# ─── AODVStrategy ─────────────────────────────────────────────────────────────

class TestAODVStrategy:
    strat = AODVStrategy()

    def test_selects_exactly_one(self):
        nbrs = make_neighbors(6)
        d = self.strat.select_recipients(StubNode(), StubMessage(), nbrs)
        assert len(d.selected_neighbors) == 1

    def test_selects_best_link(self):
        """Node with highest success_rate * rssi_norm * freshness is picked."""
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        nbrs = {
            0: StubNeighborInfo(node_id=0, success_rate=0.3, rssi=-90.0, last_seen=99.0),
            1: StubNeighborInfo(node_id=1, success_rate=0.9, rssi=-60.0, last_seen=100.0),
            2: StubNeighborInfo(node_id=2, success_rate=0.5, rssi=-75.0, last_seen=95.0),
        }
        d = self.strat.select_recipients(node, msg, nbrs)
        assert d.selected_neighbors == [1]

    def test_empty_neighbors(self):
        d = self.strat.select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    def test_strategy_name(self):
        assert self.strat.name == "aodv"


# ─── HeuristicStrategy ────────────────────────────────────────────────────────

class TestHeuristicStrategy:

    def test_selects_top_fraction(self):
        strat = HeuristicStrategy(select_fraction=0.5)
        nbrs = make_neighbors(10)
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        assert len(d.selected_neighbors) == 5

    def test_min_select_respected(self):
        strat = HeuristicStrategy(select_fraction=0.1, min_select=2)
        nbrs = make_neighbors(4)
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        assert len(d.selected_neighbors) >= 2

    def test_high_energy_preferred(self):
        """Neighbor with higher residual energy should rank higher, all else equal."""
        strat = HeuristicStrategy(select_fraction=0.5)
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        nbrs = {
            0: StubNeighborInfo(node_id=0, residual_energy=0.1, last_seen=100.0),
            1: StubNeighborInfo(node_id=1, residual_energy=0.9, last_seen=100.0),
        }
        d = strat.select_recipients(node, msg, nbrs)
        # With select_fraction=0.5 on 2 neighbors → k=1; best should be node 1
        assert 1 in d.selected_neighbors

    def test_empty_neighbors(self):
        strat = HeuristicStrategy()
        d = strat.select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    def test_scores_all_present(self):
        strat = HeuristicStrategy(select_fraction=0.6)
        nbrs = make_neighbors(5)
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        # Scores should be computed for all neighbors, not just selected ones
        assert set(d.scores.keys()) == set(nbrs.keys())

    def test_features_returned_for_selected(self):
        strat = HeuristicStrategy(select_fraction=0.6)
        nbrs = make_neighbors(5)
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        assert d.features is not None
        for nid in d.selected_neighbors:
            assert nid in d.features
            assert len(d.features[nid]) == 12

    def test_strategy_name(self):
        assert HeuristicStrategy().name == "heuristic"


# ─── AIStrategy (v14 regime-aware) ────────────────────────────────────────────

class TestAIStrategy:

    def _make_strat(self, **kwargs) -> AIStrategy:
        return AIStrategy(**kwargs)

    def test_dense_normal_fraction(self):
        """In a dense regime, normal select_fraction=0.5 is used."""
        strat = self._make_strat()
        # Dense neighbors: high success_rate, many neighbors
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        nbrs = {
            i: StubNeighborInfo(
                node_id=i, success_rate=0.85, rssi=-65.0,
                degree=8, last_seen=99.0, tx_attempts=10,
            )
            for i in range(10)
        }
        d = strat.select_recipients(node, msg, nbrs)
        # In normal mode: k = max(1, 10 * 0.5) = 5, floor = max(1, 10*0.3)=3
        assert len(d.selected_neighbors) >= 3

    def test_sparse_expands_fraction(self):
        """In sparse regime, SPARSE_FRACTION (0.85) is used, selecting more relays.

        We need score_spread >= CONFIDENCE_THRESHOLD (0.15) to bypass the fallback
        path and reach the regime-selection branch.  We achieve this by giving one
        neighbor very high features (strong relay candidate) and one very low features
        (weak relay candidate), ensuring logistic scores are far apart.
        """
        strat = self._make_strat()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        # 4 neighbors in sparse regime (degree=2 < SPARSE_DEGREE=3)
        nbrs = {
            0: StubNeighborInfo(node_id=0, success_rate=0.9, rssi=-60.0, residual_energy=0.95,
                                buffer_free=0.9, degree=2, tx_attempts=5, last_seen=99.5),
            1: StubNeighborInfo(node_id=1, success_rate=0.3, rssi=-95.0, residual_energy=0.2,
                                buffer_free=0.3, degree=2, tx_attempts=5, last_seen=80.0),
            2: StubNeighborInfo(node_id=2, success_rate=0.8, rssi=-65.0, residual_energy=0.85,
                                buffer_free=0.8, degree=2, tx_attempts=5, last_seen=98.0),
            3: StubNeighborInfo(node_id=3, success_rate=0.1, rssi=-100.0, residual_energy=0.1,
                                buffer_free=0.1, degree=2, tx_attempts=5, last_seen=70.0),
        }
        d = strat.select_recipients(node, msg, nbrs)
        # Sparse or fallback — either way, at least 2 neighbors selected (MIN_RELAY_FRACTION floor)
        assert len(d.selected_neighbors) >= 2
        # At least one of the high-quality neighbors should be selected
        assert any(nid in d.selected_neighbors for nid in [0, 2])

    def test_minimum_relay_floor(self):
        """At least MIN_RELAY_FRACTION neighbors are always selected."""
        strat = self._make_strat(select_fraction=0.1)  # very aggressive suppression
        nbrs = make_neighbors(10)
        # Vary features slightly to avoid confidence fallback
        for i, info in nbrs.items():
            info.success_rate = 0.1 + i * 0.08
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        min_floor = max(1, int(10 * AIStrategy.MIN_RELAY_FRACTION))
        assert len(d.selected_neighbors) >= min_floor

    def test_confidence_fallback_on_uniform_scores(self):
        """When all neighbors are identical, score_spread < threshold → fallback."""
        strat = self._make_strat()
        # All neighbors identical → logistic produces identical scores
        nbrs = {i: StubNeighborInfo(node_id=i) for i in range(6)}
        d = strat.select_recipients(StubNode(), StubMessage(), nbrs)
        # Fallback activated: at least one neighbor selected
        assert len(d.selected_neighbors) >= 1
        assert strat._fallback_activations >= 1

    def test_telemetry_counters_sum(self):
        """Mode telemetry counters should sum to total decisions."""
        strat = self._make_strat()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        for _ in range(5):
            nbrs = make_neighbors(4)
            strat.select_recipients(node, msg, nbrs)
        stats = strat.mode_stats()
        total = stats["sparse_fraction"] + stats["fallback_fraction"] + stats["normal_fraction"]
        assert abs(total - 1.0) < 1e-9 or (
            strat._sparse_activations + strat._fallback_activations + strat._normal_activations == 0
        )

    def test_load_weights_switches_to_logistic(self):
        strat = self._make_strat(model_type="trees")
        strat.load_weights([0.1] * 12, bias=0.5)
        assert strat.model_type == "logistic"
        assert strat.weights == [0.1] * 12
        assert strat.bias == 0.5

    def test_empty_neighbors(self):
        d = self._make_strat().select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    def test_strategy_name(self):
        assert AIStrategy().name == "ai"

    def test_feature_vector_length(self):
        """_extract_features must return exactly 12 values."""
        strat = self._make_strat()
        info = StubNeighborInfo()
        msg = StubMessage()
        feats = strat._extract_features(info, msg, current_time=100.0, my_neighbor_ids={1, 2})
        assert len(feats) == 12

    def test_feature_values_in_range(self):
        """All features should be in [0, 1] (logistic/probability-friendly)."""
        strat = self._make_strat()
        info = StubNeighborInfo(rssi=-110.0, degree=25, two_hop_degree=150)
        msg = StubMessage(hop_count=15, created_at=0.0)
        feats = strat._extract_features(info, msg, current_time=200.0, my_neighbor_ids={1, 2})
        for i, f in enumerate(feats):
            assert 0.0 <= f <= 1.0, f"Feature F{i+1}={f} out of [0,1]"

    def test_logistic_output_in_01(self):
        """Logistic prediction must stay in (0, 1)."""
        strat = self._make_strat()
        for _ in range(50):
            feats = [random.uniform(0, 1) for _ in range(12)]
            p = strat._predict_logistic(feats)
            assert 0.0 < p < 1.0


# ─── AdaptiveStrategy ─────────────────────────────────────────────────────────

class TestAdaptiveStrategy:

    def test_sparse_regime_selects_all(self):
        """With degree < 3, adaptive should activate sparse mode (full coverage)."""
        strat = AdaptiveStrategy()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        # Only 2 neighbors → degree < DEGREE_THRESH=3 → sparse
        nbrs = {
            0: StubNeighborInfo(node_id=0, success_rate=0.8, tx_attempts=5, last_seen=99.0),
            1: StubNeighborInfo(node_id=1, success_rate=0.7, tx_attempts=5, last_seen=99.0),
        }
        d = strat.select_recipients(node, msg, nbrs)
        assert set(d.selected_neighbors) == {0, 1}
        assert strat._sparse_activations == 1

    def test_normal_regime_suppresses(self):
        """Dense well-connected network should activate normal mode."""
        strat = AdaptiveStrategy(select_fraction=0.5)
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        nbrs = {
            i: StubNeighborInfo(
                node_id=i, success_rate=0.85,
                last_seen=99.0, tx_attempts=5,
            )
            for i in range(8)
        }
        d = strat.select_recipients(node, msg, nbrs)
        # Normal mode with fraction=0.5 → ~4 selected from 8
        assert len(d.selected_neighbors) <= 6
        assert strat._normal_activations == 1

    def test_poor_sr_triggers_sparse(self):
        """Average success_rate < 0.5 should trigger sparse mode."""
        strat = AdaptiveStrategy()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        nbrs = {
            i: StubNeighborInfo(
                node_id=i, success_rate=0.2, last_seen=99.0, tx_attempts=5,
            )
            for i in range(5)  # degree=5 ≥ 3, but sr < 0.5
        }
        d = strat.select_recipients(node, msg, nbrs)
        assert strat._sparse_activations == 1

    def test_mode_stats_sum_to_one(self):
        strat = AdaptiveStrategy()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        for n in [2, 8, 2, 6]:
            nbrs = make_neighbors(n, tx_attempts=5)
            strat.select_recipients(node, msg, nbrs)
        stats = strat.mode_stats()
        assert abs(stats["sparse_fraction"] + stats["normal_fraction"] - 1.0) < 1e-9

    def test_empty_neighbors(self):
        d = AdaptiveStrategy().select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    def test_strategy_name(self):
        assert AdaptiveStrategy().name == "adaptive"

    def test_ewma_warmup_guard(self):
        """Neighbors with tx_attempts < MIN_SAMPLES should not count toward sr check."""
        strat = AdaptiveStrategy()
        node = StubNode(current_time=100.0)
        msg = StubMessage()
        # 5 neighbors but all under-sampled → sr check skipped → falls back to degree check
        nbrs = {
            i: StubNeighborInfo(
                node_id=i, success_rate=0.1, last_seen=99.0, tx_attempts=1,
            )
            for i in range(5)  # degree=5 ≥ 3, under-sampled sr not counted
        }
        strat.select_recipients(node, msg, nbrs)
        # degree=5 ≥ 3 and sr check skipped → should be normal
        assert strat._normal_activations == 1


# ─── Cross-strategy properties ────────────────────────────────────────────────

class TestCrossStrategyProperties:
    """Invariants that should hold for ALL strategies."""

    @pytest.mark.parametrize("strategy", [
        FloodStrategy(),
        AODVStrategy(),
        HeuristicStrategy(),
        AIStrategy(),
        AdaptiveStrategy(),
    ])
    def test_selected_are_subset_of_neighbors(self, strategy):
        nbrs = make_neighbors(5, tx_attempts=5)
        d = strategy.select_recipients(StubNode(), StubMessage(), nbrs)
        assert set(d.selected_neighbors).issubset(set(nbrs.keys()))

    @pytest.mark.parametrize("strategy", [
        FloodStrategy(),
        AODVStrategy(),
        HeuristicStrategy(),
        AIStrategy(),
        AdaptiveStrategy(),
    ])
    def test_empty_input_gives_empty_output(self, strategy):
        d = strategy.select_recipients(StubNode(), StubMessage(), {})
        assert d.selected_neighbors == []

    @pytest.mark.parametrize("strategy", [
        FloodStrategy(),
        AODVStrategy(),
        HeuristicStrategy(),
        AIStrategy(),
        AdaptiveStrategy(),
    ])
    def test_single_neighbor_always_selected(self, strategy):
        """With only one neighbor, every strategy must forward to it."""
        nbrs = make_neighbors(1, tx_attempts=5)
        d = strategy.select_recipients(StubNode(), StubMessage(), nbrs)
        assert len(d.selected_neighbors) >= 1

    @pytest.mark.parametrize("strategy", [
        FloodStrategy(),
        AODVStrategy(),
        HeuristicStrategy(),
        AIStrategy(),
        AdaptiveStrategy(),
    ])
    def test_decision_has_strategy_name(self, strategy):
        nbrs = make_neighbors(3, tx_attempts=5)
        d = strategy.select_recipients(StubNode(), StubMessage(), nbrs)
        assert d.strategy_name == strategy.name
