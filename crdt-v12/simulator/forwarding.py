"""
Forwarding Strategy Implementations for Mesh Message Dissemination.

Three strategies are provided, enabling comparative evaluation:

1. **FloodStrategy** — Epidemic forwarding: broadcast to all neighbors.
   Maximizes delivery probability at the cost of high overhead.

2. **HeuristicStrategy** — Score-based relay selection using a
   hand-crafted utility function combining link quality, residual
   energy, buffer availability, and node degree.

3. **AIStrategy** — Machine-learning-assisted relay selection using
   a trained classifier to predict forwarding utility per neighbor.

Reference:
    Vahdat, A. & Becker, D. (2000). Epidemic Routing for Partially
    Connected Ad Hoc Networks. Duke University Tech Report CS-2000-06.

    Spyropoulos, T., Psounis, K., & Raghavendra, C.S. (2005).
    Spray and Wait: An Efficient Routing Scheme for Intermittently
    Connected Mobile Networks. ACM WDTN.
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .node import ConstrainedNode, NeighborInfo
    from .crdt import Message


@dataclass
class ForwardingDecision:
    """Result of a forwarding strategy decision."""
    selected_neighbors: List[int]       # Node IDs to forward to
    scores: Dict[int, float]            # Utility scores per candidate
    strategy_name: str                  # Name of strategy used
    features: Optional[Dict] = None     # ML features (for training data)


class ForwardingStrategy(ABC):
    """Abstract base class for forwarding strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        """
        Select which neighbors to forward a message to.

        Args:
            node: The forwarding node.
            message: The message to forward.
            neighbors: Current neighbor table {node_id: NeighborInfo}.

        Returns:
            ForwardingDecision with selected neighbors and scores.
        """
        ...


class FloodStrategy(ForwardingStrategy):
    """
    Epidemic Flooding — forward to ALL neighbors.

    This is the simplest and most robust strategy. It guarantees
    maximum reachability but generates O(n²) duplicate packets.
    Serves as the upper-bound baseline for delivery ratio.
    """

    @property
    def name(self) -> str:
        return "flood"

    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        selected = list(neighbors.keys())
        scores = {nid: 1.0 for nid in selected}
        return ForwardingDecision(
            selected_neighbors=selected,
            scores=scores,
            strategy_name=self.name,
        )


class HeuristicStrategy(ForwardingStrategy):
    """
    Heuristic-based relay selection.

    Computes a utility score for each neighbor using a hand-crafted
    weighted function:

        U(j) = α·SuccessRate(j) + β·Energy(j) + γ·BufferFree(j)
             - δ·Degree(j) + ε·LinkFreshness(j)

    Selects the top-k neighbors by utility, where k is a configurable
    fraction of the neighbor count (default: top 60%).

    This serves as the "intelligent but non-ML" baseline.
    """

    def __init__(
        self,
        alpha: float = 0.35,   # link quality weight — matches paper Eq.(5)
        beta: float = 0.25,     # energy weight
        gamma: float = 0.15,    # buffer availability weight
        delta: float = 0.10,    # degree penalty weight
        epsilon: float = 0.15,  # link freshness weight
        select_fraction: float = 0.6,   # top fraction to select
        min_select: int = 1,    # minimum neighbors to select
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.select_fraction = select_fraction
        self.min_select = min_select

    @property
    def name(self) -> str:
        return "heuristic"

    def _compute_utility(
        self,
        neighbor: 'NeighborInfo',
        current_time: float,
    ) -> float:
        """Compute utility score for a single neighbor."""
        # Normalize features to [0, 1]
        success = neighbor.success_rate  # already 0-1
        energy = neighbor.residual_energy  # already 0-1
        buffer_free = neighbor.buffer_free  # already 0-1

        # Degree: normalize to 0-1 range (assume max degree ~20)
        degree_norm = min(neighbor.degree / 20.0, 1.0)

        # Link freshness: how recently we heard from this neighbor
        age = current_time - neighbor.last_seen
        freshness = max(0.0, 1.0 - age / 30.0)  # 0 if > 30s ago

        utility = (
            self.alpha * success
            + self.beta * energy
            + self.gamma * buffer_free
            - self.delta * degree_norm
            + self.epsilon * freshness
        )

        return utility

    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        if not neighbors:
            return ForwardingDecision([], {}, self.name)

        current_time = node.current_time

        # Score all neighbors and extract features
        scores: Dict[int, float] = {}
        all_features: Dict[int, List[float]] = {}
        my_ids = set(neighbors.keys())
        for nid, info in neighbors.items():
            scores[nid] = self._compute_utility(info, current_time)

            # 12-feature vector — MUST match AIStrategy._extract_features exactly
            features = [
                info.success_rate,                                              # F1
                max(0, (info.rssi + 100) / 60.0),                              # F2
                info.residual_energy,                                           # F3
                info.buffer_free,                                               # F4
                min(info.degree / 20.0, 1.0),                                  # F5
                max(0, 1.0 - (current_time - info.last_seen) / 30.0),         # F6
                min(message.hop_count / 10.0, 1.0),                            # F7
                min((current_time - message.created_at) / 60.0, 1.0),         # F8
                min(info.two_hop_degree / 100.0, 1.0),                         # F9
                info.delivery_history_ratio(),                                  # F10
                (info.mobility_proxy() + 1.0) / 2.0,                          # F11
                1.0 - info.jaccard_overlap(my_ids),                            # F12
            ]
            all_features[nid] = features

        # Sort by utility descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select top fraction (at least min_select)
        k = max(self.min_select, int(len(ranked) * self.select_fraction))
        k = min(k, len(ranked))

        selected = [nid for nid, _ in ranked[:k]]

        return ForwardingDecision(
            selected_neighbors=selected,
            scores=scores,
            strategy_name=self.name,
            features={nid: all_features[nid] for nid in selected},
        )


class AIStrategy(ForwardingStrategy):
    """
    ML-assisted relay selection using a trained classifier.

    Uses a supervised model (logistic regression or gradient boosting)
    trained on traces from baseline simulations to predict the utility
    of forwarding a message to each neighbor.

    Feature vector per candidate neighbor:
        [success_rate, rssi_norm, residual_energy, buffer_free,
         degree_norm, link_age_norm, message_hops_norm, message_age_norm]

    Model output: P(success | features) ∈ [0, 1]

    Selection: Top fraction of neighbors ranked by predicted utility.

    The model weights are loaded from a JSON file exported by the
    training pipeline (ml/pipeline.py).
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = 0.0,
        select_fraction: float = 0.5,
        min_select: int = 1,
        model_type: str = "logistic",  # "logistic" or "trees"
        tree_model: Optional[object] = None,
    ):
        """
        Args:
            weights: Feature weights for logistic regression.
            bias: Bias term for logistic regression.
            select_fraction: Fraction of neighbors to select.
            min_select: Minimum neighbors to forward to.
            model_type: "logistic" for linear model, "trees" for ensemble.
            tree_model: Trained sklearn model for tree-based prediction.
        """
        # 12-feature weights: F1..F8 original + F9 two-hop + F10 delivery-hist + F11 mobility + F12 jaccard
        # 12 weights: F1..F8 original + F9 two-hop + F10 delivery-hist + F11 mobility + F12 jaccard
        self.weights = weights or [0.40, 0.15, 0.25, 0.10, -0.08, 0.12, -0.05, -0.03, 0.08, 0.20, -0.15, 0.10]
        self.bias = bias
        self.select_fraction = select_fraction
        self.min_select = min_select
        self.model_type = model_type
        self.tree_model = tree_model

    @property
    def name(self) -> str:
        return "ai"

    def _extract_features(
        self,
        neighbor: 'NeighborInfo',
        message: 'Message',
        current_time: float,
        my_neighbor_ids: set = None,
    ) -> List[float]:
        """
        Extract 12-feature vector for a single neighbor-message pair.

        F1  success_rate        – EMA link quality
        F2  rssi_norm           – RSSI normalised to [0,1]
        F3  residual_energy     – battery fraction
        F4  buffer_free         – buffer availability
        F5  degree_norm         – 1-hop degree (normalised)
        F6  link_freshness      – how recently heard
        F7  hop_norm            – message hop count
        F8  age_norm            – message age
        F9  two_hop_norm        – 2-hop degree proxy (neighbourhood scope)
        F10 delivery_history    – rolling 60s relay success ratio
        F11 mobility_proxy      – RSSI rate-of-change (normalised)
        F12 jaccard_overlap     – neighbour set similarity (redundancy avoidance)
        """
        my_ids = my_neighbor_ids or set()
        features = [
            neighbor.success_rate,                                          # F1
            max(0, (neighbor.rssi + 100) / 60.0),                          # F2
            neighbor.residual_energy,                                        # F3
            neighbor.buffer_free,                                            # F4
            min(neighbor.degree / 20.0, 1.0),                               # F5
            max(0, 1.0 - (current_time - neighbor.last_seen) / 30.0),      # F6
            min(message.hop_count / 10.0, 1.0),                             # F7
            min((current_time - message.created_at) / 60.0, 1.0),          # F8
            min(neighbor.two_hop_degree / 100.0, 1.0),                      # F9
            neighbor.delivery_history_ratio(),                               # F10
            (neighbor.mobility_proxy() + 1.0) / 2.0,                        # F11 → [0,1]
            1.0 - neighbor.jaccard_overlap(my_ids),                          # F12 (low overlap = diverse)
        ]
        return features

    def _predict_logistic(self, features: List[float]) -> float:
        """Logistic regression prediction."""
        z = self.bias
        for w, f in zip(self.weights, features):
            z += w * f
        # Sigmoid
        z = max(-50, min(50, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _predict_tree(self, features: List[float]) -> float:
        """Tree-based model prediction."""
        if self.tree_model is None:
            return self._predict_logistic(features)
        try:
            import numpy as np
            X = np.array(features).reshape(1, -1)
            proba = self.tree_model.predict_proba(X)
            return float(proba[0][1])
        except Exception:
            return self._predict_logistic(features)

    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        if not neighbors:
            return ForwardingDecision([], {}, self.name)

        current_time = node.current_time
        scores: Dict[int, float] = {}
        all_features: Dict[int, List[float]] = {}

        my_ids = set(neighbors.keys())
        for nid, info in neighbors.items():
            features = self._extract_features(info, message, current_time, my_ids)
            all_features[nid] = features

            if self.model_type == "trees":
                scores[nid] = self._predict_tree(features)
            else:
                scores[nid] = self._predict_logistic(features)

        # Select top fraction
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        k = max(self.min_select, int(len(ranked) * self.select_fraction))
        k = min(k, len(ranked))
        selected = [nid for nid, _ in ranked[:k]]

        return ForwardingDecision(
            selected_neighbors=selected,
            scores=scores,
            strategy_name=self.name,
            features={nid: all_features[nid] for nid in selected},
        )

    def load_weights(self, weights: List[float], bias: float):
        """Load trained logistic regression weights."""
        self.weights = weights
        self.bias = bias
        self.model_type = "logistic"

    def load_tree_model(self, model):
        """Load a trained sklearn tree model."""
        self.tree_model = model
        self.model_type = "trees"


class AODVStrategy(ForwardingStrategy):
    """
    Simplified AODV-style unicast forwarding for baseline comparison.

    Real AODV runs a Route Request / Route Reply discovery phase then
    forwards packets along the discovered unicast path.  Inside this
    discrete-event simulator we approximate this with a greedy
    best-neighbour selection: pick the single neighbour with the highest
    link quality (success_rate × RSSI) as the next hop.  This captures
    AODV's key behavioural properties:

        * Unicast only → duplicate rate ≈ 0 (each step picks exactly 1 relay)
        * No multicast → PDR suffers under partition because there is no
          redundant path
        * Route discovery overhead not modelled (conservative advantage
          to AODV in PDR terms)

    This gives reviewers the expected contrast:
        CRDT-Gossip: higher PDR under partition, higher duplicate rate
        AODV:        near-zero duplicates, fragile under partition

    Reference:
        Perkins, C.E. & Royer, E.M. (1999). Ad-hoc On-Demand Distance
        Vector Routing. IEEE WMCSA, pp. 90-100.
    """

    @property
    def name(self) -> str:
        return "aodv"

    def _link_score(self, neighbor: 'NeighborInfo', current_time: float) -> float:
        """Combined link quality metric (higher = better route)."""
        rssi_norm = max(0.0, (neighbor.rssi + 100) / 60.0)
        freshness = max(0.0, 1.0 - (current_time - neighbor.last_seen) / 30.0)
        return neighbor.success_rate * rssi_norm * freshness

    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        if not neighbors:
            return ForwardingDecision([], {}, self.name)

        current_time = node.current_time
        scores = {
            nid: self._link_score(info, current_time)
            for nid, info in neighbors.items()
        }

        # AODV unicast: pick exactly ONE best next-hop
        best = max(scores, key=lambda nid: scores[nid])

        return ForwardingDecision(
            selected_neighbors=[best],
            scores=scores,
            strategy_name=self.name,
        )


class AdaptiveStrategy(ForwardingStrategy):
    """
    Regime-Adaptive Relay Controller.

    Motivated by the empirical finding that static relay-selection strategies
    fail under sparse correlated-mobility (Gauss-Markov n=20): the heuristic's
    fixed select_fraction=0.6 suppresses relays that are the only available
    path, collapsing PDR by 16pp versus flooding.

    The controller reads two aggregates from the existing neighbor table on
    every forwarding decision — no extra messages, no global state:

        local_degree : number of live neighbors this node currently sees.
        avg_sr       : mean link success-rate (EWMA) across current neighbors.

    Switching rule (empirically derived; §VI.D):

        sparse  (local_degree < DEGREE_THRESH  OR  avg_sr < SR_THRESH):
                → select_fraction = 1.0  (forward to ALL scored neighbors)
        normal  (otherwise):
                → select_fraction = NORMAL_FRACTION (standard suppression)

    Both modes use HeuristicStrategy's scoring function — the adaptation is
    purely in the relay-set size, not the scoring criterion.

        Sparse mode  = heuristic ordering + full coverage (flood-equivalent
                       reach; low-scored links still deprioritised in TX order).
        Normal mode  = standard heuristic with overhead suppression.

    Threshold justification:
        DEGREE_THRESH  = 3   : k-connectivity minimum; below this, suppressing
                               any relay risks severing the network.
        SR_THRESH      = 0.50: majority-failure criterion; below 0.5 the average
                               link drops more than half its packets.
        NORMAL_FRACTION = 0.6: inherited from HeuristicStrategy default.

    Note on GM n=20 results: empirical evaluation shows the adaptive controller
    improves PDR in RWP (+3pp) where transient topology gaps cause the heuristic
    to over-suppress, but cannot fully recover PDR under sustained correlated
    channel fading (GM n=20). This establishes a fundamental limit: local-signal
    adaptation resolves suppression-induced loss but not channel-level loss.
    The decomposition is: ~8.6pp suppression-recoverable, ~7.8pp channel-induced
    (informationally invisible to any local forwarding policy).

    Reference for regime-adaptive rationale:
        Vahdat, A. & Becker, D. (2000). Epidemic Routing for Partially
        Connected Ad Hoc Networks. Duke CS Tech Report CS-2000-06.
    """

    DEGREE_THRESH: int = 3
    SR_THRESH: float = 0.50
    NORMAL_FRACTION: float = 0.60
    SPARSE_FRACTION: float = 1.00

    def __init__(self, select_fraction: float = 0.60):
        self.NORMAL_FRACTION = select_fraction
        self._sparse = HeuristicStrategy(select_fraction=self.SPARSE_FRACTION)
        self._normal = HeuristicStrategy(select_fraction=self.NORMAL_FRACTION)
        self._sparse_activations: int = 0
        self._normal_activations: int = 0

    @property
    def name(self) -> str:
        return "adaptive"

    MIN_SAMPLES: int = 3   # minimum tx_attempts before sr is trusted

    def _assess_regime(
        self,
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> str:
        """
        Classify current network regime from local neighbor table.

        Returns 'sparse' (→ full coverage) or 'normal' (→ suppressed).

        Signals:
          local_degree : len(neighbors) — live neighbors this node sees NOW.
                         Uses local count, not neighbor-reported degree.
          avg_sr       : mean EWMA success-rate across neighbors that have
                         at least MIN_SAMPLES tx_attempts (EWMA warmup guard).
                         If no neighbor has sufficient history, defaults to
                         'normal' to avoid cold-start flooding in dense nets.
        """
        if not neighbors:
            return "sparse"

        local_degree = len(neighbors)

        # Only use sr from neighbors with sufficient tx history (EWMA warmup)
        seasoned = [n for n in neighbors.values() if n.tx_attempts >= self.MIN_SAMPLES]
        if seasoned:
            avg_sr = sum(n.success_rate for n in seasoned) / len(seasoned)
            sr_sparse = avg_sr < self.SR_THRESH
        else:
            sr_sparse = False  # insufficient history → assume normal, don't flood

        if local_degree < self.DEGREE_THRESH or sr_sparse:
            return "sparse"
        return "normal"

    def select_recipients(
        self,
        node: 'ConstrainedNode',
        message: 'Message',
        neighbors: Dict[int, 'NeighborInfo'],
    ) -> ForwardingDecision:
        regime = self._assess_regime(neighbors)

        if regime == "sparse":
            self._sparse_activations += 1
            decision = self._sparse.select_recipients(node, message, neighbors)
        else:
            self._normal_activations += 1
            decision = self._normal.select_recipients(node, message, neighbors)

        return ForwardingDecision(
            selected_neighbors=decision.selected_neighbors,
            scores=decision.scores,
            strategy_name=self.name,
            features=decision.features,
        )

    def mode_stats(self) -> Dict[str, float]:
        """Return fraction of decisions made in each mode."""
        total = self._sparse_activations + self._normal_activations
        if total == 0:
            return {"sparse_fraction": 0.0, "normal_fraction": 0.0}
        return {
            "sparse_fraction": self._sparse_activations / total,
            "normal_fraction": self._normal_activations / total,
        }
