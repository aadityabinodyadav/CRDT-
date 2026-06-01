"""
Mesh Network Simulation Orchestrator.

Coordinates the discrete-event simulation of an infrastructure-less
wireless mesh network with CRDT-based distributed synchronization.

Manages:
- Node creation and placement (grid, random, clustered)
- Mobility models (static, random waypoint, Gauss-Markov)
- Wireless channel simulation (delivery, RSSI, PER)
- Event scheduling (message generation, HELLO, sync, partitions)
- Packet delivery between nodes
- Network partitions and recovery
- Convergence monitoring

This is the main simulation driver that ties together all components:
    Engine + Channel + Nodes + Metrics → Complete simulation run
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .engine import SimulationEngine
from .crdt import Message, MessageId
from .wireless import WirelessChannel, ChannelConfig
from .node import ConstrainedNode, NodeState, Packet, PacketType
from .forwarding import (
    AdaptiveStrategy, AIStrategy, FloodStrategy, ForwardingStrategy, HeuristicStrategy,
)
from .metrics import EventType, MetricsCollector


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """
    Complete simulation configuration.

    Provides all parameters needed to set up and run a mesh
    network simulation experiment.
    """
    # Network topology
    n_nodes: int = 30
    area_width: float = 300.0       # meters
    area_height: float = 300.0      # meters
    placement: str = "random"       # "random", "grid", "clustered"

    # Simulation
    sim_duration: float = 300.0     # seconds
    seed: int = 42
    warmup_time: float = 10.0      # seconds before metrics collection

    # Node parameters
    message_gen_rate: float = 0.1   # messages per second per node
    max_buffer_size: int = 256      # max messages per node
    initial_energy_j: float = 26640.0
    num_source_nodes: int = -1      # -1 = all nodes generate messages

    # Protocol parameters
    hello_interval: float = 2.0     # seconds
    sync_interval: float = 1.0      # seconds
    adaptive_sync: bool = True
    fine_sync_interval: float = 0.3
    coarse_sync_interval: float = 5.0

    # Forwarding
    forwarding_mode: str = "flood"  # "flood", "heuristic", "ai"
    ai_weights: Optional[List[float]] = None
    ai_bias: float = 0.0
    ai_model: Optional[Any] = None  # sklearn model object
    heuristic_select_fraction: float = 0.6

    # Wireless channel
    tx_power_dbm: float = 20.0
    path_loss_exponent: float = 3.0
    shadowing_std_db: float = 6.0
    noise_floor_dbm: float = -91.0
    sinr_threshold_db: float = 10.0
    max_range_m: float = 200.0

    # Mobility
    mobility_model: str = "static"  # "static", "random_waypoint"
    mobility_speed_min: float = 0.0  # m/s
    mobility_speed_max: float = 5.0  # m/s
    mobility_pause_min: float = 0.0  # seconds
    mobility_pause_max: float = 5.0  # seconds
    mobility_update_interval: float = 1.0

    # Partition scenario
    enable_partitions: bool = False
    partition_start_time: float = 100.0
    partition_duration: float = 60.0
    partition_type: str = "spatial"  # "spatial" (split by x-axis)

    # Energy constraint scenario
    energy_constrained: bool = False
    limited_energy_j: float = 100.0  # much less than default

    def __post_init__(self):
        if self.num_source_nodes == -1:
            self.num_source_nodes = self.n_nodes


# ─────────────────────────────────────────────────────────────────────
# Mobility Models
# ─────────────────────────────────────────────────────────────────────

class MobilityModel:
    """Base class for node mobility."""

    def update(self, node: ConstrainedNode, dt: float, rng: np.random.Generator,
               area_w: float, area_h: float):
        pass


class StaticMobility(MobilityModel):
    """Nodes don't move."""
    def update(self, node, dt, rng, area_w, area_h):
        pass  # Already static


class RandomWaypointMobility(MobilityModel):
    """
    Random Waypoint mobility model.

    Each node selects a random destination, moves toward it at a
    random speed, pauses for a random duration, then selects a new
    destination.

    Reference:
        Johnson, D.B. & Maltz, D.A. (1996). Dynamic Source Routing
        in Ad Hoc Wireless Networks. Mobile Computing.
    """

    def __init__(self, speed_min: float, speed_max: float,
                 pause_min: float, pause_max: float):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pause_min = pause_min
        self.pause_max = pause_max
        self._node_state: Dict[int, Dict] = {}

    def _init_node(self, node: ConstrainedNode, rng: np.random.Generator,
                   area_w: float, area_h: float):
        """Initialize mobility state for a node."""
        dest = (rng.uniform(0, area_w), rng.uniform(0, area_h))
        speed = rng.uniform(self.speed_min, self.speed_max)
        self._node_state[node.node_id] = {
            'destination': dest,
            'speed': speed,
            'pausing': False,
            'pause_remaining': 0.0,
        }

    def update(self, node: ConstrainedNode, dt: float, rng: np.random.Generator,
               area_w: float, area_h: float):
        """Update node position based on random waypoint model."""
        if node.node_id not in self._node_state:
            self._init_node(node, rng, area_w, area_h)

        state = self._node_state[node.node_id]

        # Handle pausing
        if state['pausing']:
            state['pause_remaining'] -= dt
            if state['pause_remaining'] <= 0:
                state['pausing'] = False
                state['destination'] = (
                    rng.uniform(0, area_w),
                    rng.uniform(0, area_h),
                )
                state['speed'] = rng.uniform(self.speed_min, self.speed_max)
            return

        # Move toward destination
        dest = state['destination']
        speed = state['speed']
        x, y = node.position

        dx = dest[0] - x
        dy = dest[1] - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < speed * dt:
            # Arrived at destination
            node.position = dest
            state['pausing'] = True
            state['pause_remaining'] = rng.uniform(self.pause_min, self.pause_max)
        else:
            # Move toward destination
            vx = dx / dist * speed
            vy = dy / dist * speed
            new_x = x + vx * dt
            new_y = y + vy * dt

            # Clamp to area
            new_x = max(0, min(area_w, new_x))
            new_y = max(0, min(area_h, new_y))

            node.position = (new_x, new_y)


# ─────────────────────────────────────────────────────────────────────
# Mesh Network Simulator
# ─────────────────────────────────────────────────────────────────────

class MeshNetwork:
    """
    Complete mesh network simulation orchestrator.

    Lifecycle:
        1. __init__(config) — creates engine, channel, metrics
        2. setup() — places nodes, schedules initial events
        3. run() — executes simulation
        4. results() — returns computed metrics
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Core components
        self.engine = SimulationEngine(seed=config.seed)
        self.channel = WirelessChannel(
            ChannelConfig(
                tx_power_dbm=config.tx_power_dbm,
                path_loss_exponent=config.path_loss_exponent,
                shadowing_std_db=config.shadowing_std_db,
                noise_floor_dbm=config.noise_floor_dbm,
                sinr_threshold_db=config.sinr_threshold_db,
                max_range_m=config.max_range_m,
            ),
            rng=self.rng,
        )
        self.metrics = MetricsCollector(n_nodes=config.n_nodes)

        # Nodes
        self.nodes: Dict[int, ConstrainedNode] = {}

        # Mobility
        self.mobility: MobilityModel = self._create_mobility()

        # Partition state
        self._partitioned = False
        self._partition_groups: List[Set[int]] = []

        # Track if setup has been called
        self._setup_done = False

    def _create_mobility(self) -> MobilityModel:
        """Create mobility model from config."""
        cfg = self.config
        if cfg.mobility_model == "random_waypoint":
            return RandomWaypointMobility(
                speed_min=cfg.mobility_speed_min,
                speed_max=cfg.mobility_speed_max,
                pause_min=cfg.mobility_pause_min,
                pause_max=cfg.mobility_pause_max,
            )
        else:
            return StaticMobility()

    def _create_forwarding(self) -> ForwardingStrategy:
        """Create forwarding strategy from config."""
        cfg = self.config
        if cfg.forwarding_mode == "heuristic":
            return HeuristicStrategy(select_fraction=cfg.heuristic_select_fraction)
        elif cfg.forwarding_mode == "adaptive":
            return AdaptiveStrategy(select_fraction=cfg.heuristic_select_fraction)
        elif cfg.forwarding_mode == "ai":
            # Auto-load trained weights if not explicitly provided
            ai_weights = cfg.ai_weights
            ai_bias = cfg.ai_bias
            if ai_weights is None:
                import json as _json, os as _os
                _wpath = _os.path.join(_os.path.dirname(__file__), '..', 'experiments', 'ai_weights.json')
                if _os.path.exists(_wpath):
                    _w = _json.load(open(_wpath))
                    ai_weights = _w.get('weights')
                    ai_bias = _w.get('bias', 0.0)
            strategy = AIStrategy(
                weights=ai_weights,
                bias=ai_bias,
            )
            if cfg.ai_model is not None:
                strategy.load_tree_model(cfg.ai_model)
            return strategy
        else:
            return FloodStrategy()

    # ─────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialize the simulation: create nodes, schedule events.
        """
        cfg = self.config
        self.engine.reset()

        # Create forwarding strategy
        forwarding = self._create_forwarding()

        # Place nodes
        positions = self._generate_positions()

        # Create nodes
        energy_j = cfg.limited_energy_j if cfg.energy_constrained else cfg.initial_energy_j

        for i in range(cfg.n_nodes):
            node = ConstrainedNode(
                node_id=i,
                position=positions[i],
                forwarding_strategy=forwarding,
                max_buffer_size=cfg.max_buffer_size,
                initial_energy_j=energy_j,
                message_gen_rate=cfg.message_gen_rate,
                hello_interval=cfg.hello_interval,
                sync_interval=cfg.sync_interval,
                adaptive_sync=cfg.adaptive_sync,
                fine_sync_interval=cfg.fine_sync_interval,
                coarse_sync_interval=cfg.coarse_sync_interval,
            )
            self.nodes[i] = node

            # Record initial energy
            self.metrics.log_energy(i, 0.0, initial=energy_j)
            self.metrics.log_event(
                time=0.0, node_id=i,
                event_type=EventType.NODE_START,
                position=positions[i],
            )

        # Schedule initial events
        self._schedule_initial_events()

        self._setup_done = True

    def _generate_positions(self) -> List[Tuple[float, float]]:
        """Generate node positions based on placement strategy."""
        cfg = self.config

        if cfg.placement == "grid":
            cols = int(math.ceil(math.sqrt(cfg.n_nodes)))
            rows = int(math.ceil(cfg.n_nodes / cols))
            dx = cfg.area_width / max(cols, 1)
            dy = cfg.area_height / max(rows, 1)
            positions = []
            for i in range(cfg.n_nodes):
                r = i // cols
                c = i % cols
                x = (c + 0.5) * dx
                y = (r + 0.5) * dy
                positions.append((x, y))
            return positions

        elif cfg.placement == "clustered":
            # 3-4 clusters with nodes grouped around cluster centers
            n_clusters = min(4, max(2, cfg.n_nodes // 8))
            centers = [
                (self.rng.uniform(50, cfg.area_width - 50),
                 self.rng.uniform(50, cfg.area_height - 50))
                for _ in range(n_clusters)
            ]
            positions = []
            for i in range(cfg.n_nodes):
                center = centers[i % n_clusters]
                x = center[0] + self.rng.normal(0, 30)
                y = center[1] + self.rng.normal(0, 30)
                x = max(0, min(cfg.area_width, x))
                y = max(0, min(cfg.area_height, y))
                positions.append((x, y))
            return positions

        else:  # "random"
            return [
                (self.rng.uniform(0, cfg.area_width),
                 self.rng.uniform(0, cfg.area_height))
                for _ in range(cfg.n_nodes)
            ]

    def _schedule_initial_events(self):
        """Schedule all periodic events for all nodes."""
        cfg = self.config

        for i in range(cfg.n_nodes):
            # Stagger starts to avoid thundering herd
            offset = self.rng.uniform(0, 2.0)

            # HELLO beacons
            self.engine.schedule(
                delay=cfg.warmup_time + offset,
                callback=self._on_hello_tick,
                data={'node_id': i},
                priority=2,
            )

            # Sync events
            self.engine.schedule(
                delay=cfg.warmup_time + offset + 0.5,
                callback=self._on_sync_tick,
                data={'node_id': i},
                priority=3,
            )

            # Message generation (only for source nodes)
            if i < cfg.num_source_nodes:
                gen_delay = cfg.warmup_time + offset + self.rng.exponential(
                    1.0 / cfg.message_gen_rate
                )
                self.engine.schedule(
                    delay=gen_delay,
                    callback=self._on_generate_message,
                    data={'node_id': i},
                    priority=1,
                )

        # Mobility updates
        if cfg.mobility_model != "static":
            self.engine.schedule(
                delay=cfg.mobility_update_interval,
                callback=self._on_mobility_update,
                data=None,
                priority=5,
            )

        # Partition events
        if cfg.enable_partitions:
            self.engine.schedule(
                delay=cfg.partition_start_time,
                callback=self._on_partition_start,
                data=None,
                priority=0,
            )
            self.engine.schedule(
                delay=cfg.partition_start_time + cfg.partition_duration,
                callback=self._on_partition_end,
                data=None,
                priority=0,
            )

        # Neighbor pruning (every 10s)
        self.engine.schedule(
            delay=10.0,
            callback=self._on_prune_neighbors,
            data=None,
            priority=10,
        )

    # ─────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _on_generate_message(self, data: Dict):
        """Handle message generation event."""
        node_id = data['node_id']
        node = self.nodes.get(node_id)
        if node is None or not node.is_alive:
            return

        sim_time = self.engine.now

        # Generate message
        message = node.generate_message(sim_time, self.metrics)

        if message is not None:
            # Forward to neighbors
            outgoing = node._forward_message(message, sim_time, self.metrics)
            for packet, recipients in outgoing:
                self._deliver_packet(node_id, packet, recipients)

        # Schedule next generation (Poisson process)
        interval = self.rng.exponential(1.0 / self.config.message_gen_rate)
        self.engine.schedule(
            delay=interval,
            callback=self._on_generate_message,
            data=data,
            priority=1,
        )

    def _on_hello_tick(self, data: Dict):
        """Handle periodic HELLO beacon."""
        node_id = data['node_id']
        node = self.nodes.get(node_id)
        if node is None or not node.is_alive:
            return

        sim_time = self.engine.now

        # Create and broadcast HELLO
        hello = node.create_hello_packet(sim_time)

        # Consume TX energy
        tx_energy = node.energy.consume_tx(hello.size_bytes)
        self.metrics.log_energy(node_id, tx_energy)
        self.metrics.log_event(
            time=sim_time, node_id=node_id,
            event_type=EventType.SYNC_HELLO,
            size_bytes=hello.size_bytes,
        )

        # Broadcast to all reachable nodes
        reachable = self._get_reachable_nodes(node_id)
        self._deliver_packet(node_id, hello, list(reachable))

        # Schedule next HELLO
        self.engine.schedule(
            delay=self.config.hello_interval,
            callback=self._on_hello_tick,
            data=data,
            priority=2,
        )

    def _on_sync_tick(self, data: Dict):
        """Handle periodic synchronization."""
        node_id = data['node_id']
        node = self.nodes.get(node_id)
        if node is None or not node.is_alive:
            return

        sim_time = self.engine.now

        # Create sync packets
        outgoing = node.create_sync_packets(sim_time, self.metrics)
        for packet, recipients in outgoing:
            self._deliver_packet(node_id, packet, recipients)

        # Schedule next sync (adaptive interval)
        interval = node.get_adaptive_sync_interval(sim_time)
        self.engine.schedule(
            delay=interval,
            callback=self._on_sync_tick,
            data=data,
            priority=3,
        )

    def _on_mobility_update(self, data):
        """Update positions of all mobile nodes."""
        dt = self.config.mobility_update_interval
        for node in self.nodes.values():
            if node.is_alive:
                self.mobility.update(
                    node, dt, self.rng,
                    self.config.area_width, self.config.area_height,
                )

        # Schedule next mobility update
        self.engine.schedule(
            delay=self.config.mobility_update_interval,
            callback=self._on_mobility_update,
            data=None,
            priority=5,
        )

    def _on_prune_neighbors(self, data):
        """Periodic neighbor table cleanup."""
        sim_time = self.engine.now
        for node in self.nodes.values():
            if node.is_alive:
                node.prune_stale_neighbors(sim_time)

        self.engine.schedule(
            delay=10.0,
            callback=self._on_prune_neighbors,
            data=None,
            priority=10,
        )

    def _on_partition_start(self, data):
        """Start a network partition."""
        sim_time = self.engine.now
        self._partitioned = True

        # Split by x-axis (left half vs right half)
        mid_x = self.config.area_width / 2.0
        group_a = set()
        group_b = set()

        for nid, node in self.nodes.items():
            if node.position[0] < mid_x:
                group_a.add(nid)
            else:
                group_b.add(nid)

        self._partition_groups = [group_a, group_b]

        self.metrics.log_event(
            time=sim_time, node_id=-1,
            event_type=EventType.PARTITION_START,
            group_a=len(group_a), group_b=len(group_b),
        )

    def _on_partition_end(self, data):
        """End a network partition."""
        sim_time = self.engine.now
        self._partitioned = False
        self._partition_groups = []

        self.metrics.log_event(
            time=sim_time, node_id=-1,
            event_type=EventType.PARTITION_END,
        )

    def _on_packet_arrive(self, data: Dict):
        """Handle packet arrival at a receiver after propagation delay."""
        receiver_id = data['receiver_id']
        packet = data['packet']
        rssi = data['rssi']

        receiver = self.nodes.get(receiver_id)
        if receiver is None or not receiver.is_alive:
            return

        sim_time = self.engine.now

        # Process packet and get response packets
        responses = receiver.handle_packet(packet, rssi, sim_time, self.metrics)

        # Deliver response packets
        for resp_packet, recipients in responses:
            self._deliver_packet(receiver_id, resp_packet, recipients)

    # ─────────────────────────────────────────────────────────────
    # PACKET DELIVERY
    # ─────────────────────────────────────────────────────────────

    def _deliver_packet(self, sender_id: int, packet: Packet,
                        recipients: List[int]):
        """
        Attempt to deliver a packet from sender to each recipient.

        For each recipient:
        1. Check partition constraints
        2. Compute wireless channel delivery (RSSI, PER)
        3. If successful, schedule arrival event with propagation delay
        """
        sender = self.nodes.get(sender_id)
        if sender is None:
            return

        for recv_id in recipients:
            receiver = self.nodes.get(recv_id)
            if receiver is None or not receiver.is_alive:
                continue

            # Check partition
            if self._partitioned:
                if not self._can_communicate(sender_id, recv_id):
                    if packet.type == PacketType.DATA:
                        msg = packet.data.get('message')
                        if msg:
                            self.metrics.log_event(
                                time=self.engine.now,
                                node_id=sender_id,
                                event_type=EventType.MSG_DROPPED_CHANNEL,
                                msg_origin=msg.id.origin,
                                msg_seq=msg.id.seq,
                            )
                    continue

            # Wireless channel simulation
            success, rssi, per = self.channel.attempt_delivery(
                sender.position, receiver.position, packet.size_bytes,
            )

            if not success:
                # Update sender's knowledge about link quality
                if recv_id in sender.neighbors:
                    sender.neighbors[recv_id].update_success_rate(False)

                if packet.type == PacketType.DATA:
                    msg = packet.data.get('message')
                    if msg:
                        self.metrics.log_event(
                            time=self.engine.now,
                            node_id=sender_id,
                            event_type=EventType.MSG_DROPPED_CHANNEL,
                            msg_origin=msg.id.origin,
                            msg_seq=msg.id.seq,
                        )
                continue

            # Update link quality tracking
            if recv_id in sender.neighbors:
                sender.neighbors[recv_id].update_success_rate(True)

            # Compute propagation delay
            dist = self.channel.distance(sender.position, receiver.position)
            delay = self.channel.propagation_delay(dist)

            # Schedule arrival event
            self.engine.schedule(
                delay=delay,
                callback=self._on_packet_arrive,
                data={
                    'receiver_id': recv_id,
                    'packet': packet,
                    'rssi': rssi,
                },
                priority=0,
            )

    def _get_reachable_nodes(self, sender_id: int) -> Set[int]:
        """Get set of nodes that can potentially receive from sender."""
        sender = self.nodes[sender_id]
        reachable = set()

        for nid, node in self.nodes.items():
            if nid == sender_id or not node.is_alive:
                continue

            # Partition check
            if self._partitioned and not self._can_communicate(sender_id, nid):
                continue

            # Range check (approximate, without shadowing)
            dist = self.channel.distance(sender.position, node.position)
            if dist <= self.config.max_range_m:
                reachable.add(nid)

        return reachable

    def _can_communicate(self, node_a: int, node_b: int) -> bool:
        """Check if two nodes can communicate given partition state."""
        if not self._partitioned:
            return True

        for group in self._partition_groups:
            if node_a in group and node_b in group:
                return True

        return False

    # ─────────────────────────────────────────────────────────────
    # RUN SIMULATION
    # ─────────────────────────────────────────────────────────────

    def run(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute the full simulation.

        Args:
            progress_callback: Optional function(fraction) called
                periodically with progress [0.0, 1.0].

        Returns:
            Dictionary of computed metrics.
        """
        if not self._setup_done:
            self.setup()

        duration = self.config.sim_duration

        # Run in chunks for progress reporting
        chunk_size = max(1.0, duration / 100.0)
        current = 0.0

        while current < duration:
            next_time = min(current + chunk_size, duration)
            self.engine.run(until=next_time)
            current = next_time

            if progress_callback:
                progress_callback(current / duration)

        return self.results()

    def results(self) -> Dict[str, Any]:
        """
        Compute and return all simulation metrics.
        """
        # Record final energy consumption
        for nid, node in self.nodes.items():
            # Account for idle energy over simulation duration
            idle_energy = node.energy.consume_idle(self.config.sim_duration * 0.001)

        metrics = self.metrics.flat_summary()

        # Add node-level statistics
        node_stats = [node.stats() for node in self.nodes.values()]
        metrics['node_stats'] = node_stats

        # Add configuration
        metrics['config'] = {
            'n_nodes': self.config.n_nodes,
            'forwarding_mode': self.config.forwarding_mode,
            'seed': self.config.seed,
            'sim_duration': self.config.sim_duration,
            'area': f"{self.config.area_width}x{self.config.area_height}",
            'mobility': self.config.mobility_model,
            'partitions': self.config.enable_partitions,
        }

        # Print summary
        metrics['events_processed'] = self.engine.events_processed

        return metrics

    def get_convergence_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of convergence state across all nodes.

        Returns hash comparison and message count deviations.
        """
        hashes = {}
        msg_counts = {}
        for nid, node in self.nodes.items():
            if node.is_alive:
                hashes[nid] = node.message_log.state_hash()
                msg_counts[nid] = node.message_log.size

        unique_hashes = len(set(hashes.values()))
        counts = list(msg_counts.values())

        return {
            'unique_hashes': unique_hashes,
            'converged': unique_hashes == 1,
            'msg_count_mean': float(np.mean(counts)) if counts else 0,
            'msg_count_std': float(np.std(counts)) if counts else 0,
            'msg_count_min': min(counts) if counts else 0,
            'msg_count_max': max(counts) if counts else 0,
        }

    def __repr__(self) -> str:
        return (
            f"MeshNetwork(nodes={self.config.n_nodes}, "
            f"mode={self.config.forwarding_mode}, "
            f"area={self.config.area_width}x{self.config.area_height}, "
            f"duration={self.config.sim_duration}s)"
        )
