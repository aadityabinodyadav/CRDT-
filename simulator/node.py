"""
Constrained Wireless Node for CRDT-based Mesh Synchronization.

Models a resource-constrained wireless device (ESP32-class) with:
- Limited memory (bounded message buffer)
- Limited energy (battery model with TX/RX/idle consumption)
- Full CRDT state (OR-Set message log, HLC causality)
- Gossip-based anti-entropy synchronization protocol
- Configurable forwarding strategy

The node implements a state machine for protocol operation:
    IDLE → DISCOVERING → SYNCING → FORWARDING → IDLE

Energy Model Parameters (based on ESP32 datasheet):
    TX:   0.792 W  (240 mA × 3.3V)
    RX:   0.314 W  (95 mA × 3.3V)
    Idle: 0.066 W  (20 mA × 3.3V)
    Sleep: 16.5 µW (5 µA × 3.3V)

Reference:
    Espressif Systems (2023). ESP32 Technical Reference Manual, v5.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .crdt import (
    GCounter, HybridLogicalClock, LWWRegister, Message, MessageId, ORSet,
    PNCounter, UniqueTag,
)
from .forwarding import AIStrategy, FloodStrategy, ForwardingStrategy, HeuristicStrategy

if TYPE_CHECKING:
    from .metrics import MetricsCollector


# ─────────────────────────────────────────────────────────────────────
# Energy Model
# ─────────────────────────────────────────────────────────────────────

class EnergyModel:
    """
    Battery energy model for constrained wireless devices.

    Based on the ESP32 power consumption profile:
        TX:   240 mA at 3.3V → 0.792 W
        RX:    95 mA at 3.3V → 0.314 W
        Idle:  20 mA at 3.3V → 0.066 W
        Sleep:  5 µA at 3.3V → 16.5 µW

    Battery: 3.7V LiPo, default 2000 mAh → 7.4 Wh → 26640 J
    """

    def __init__(
        self,
        initial_energy_j: float = 26640.0,  # 2000 mAh × 3.7V × 3600
        tx_power_w: float = 0.792,
        rx_power_w: float = 0.314,
        idle_power_w: float = 0.066,
        sleep_power_w: float = 16.5e-6,
        data_rate_bps: float = 250_000.0,
    ):
        self.initial_energy = initial_energy_j
        self.remaining_energy = initial_energy_j
        self.tx_power = tx_power_w
        self.rx_power = rx_power_w
        self.idle_power = idle_power_w
        self.sleep_power = sleep_power_w
        self.data_rate = data_rate_bps
        self.total_consumed = 0.0

    def consume_tx(self, packet_size_bytes: int) -> float:
        """
        Energy consumed transmitting a packet.

        E_tx = P_tx × t_tx = P_tx × (bits / data_rate)
        """
        bits = packet_size_bytes * 8
        duration = bits / self.data_rate
        energy = self.tx_power * duration
        self._consume(energy)
        return energy

    def consume_rx(self, packet_size_bytes: int) -> float:
        """Energy consumed receiving a packet."""
        bits = packet_size_bytes * 8
        duration = bits / self.data_rate
        energy = self.rx_power * duration
        self._consume(energy)
        return energy

    def consume_idle(self, duration_s: float) -> float:
        """Energy consumed during idle period."""
        energy = self.idle_power * duration_s
        self._consume(energy)
        return energy

    def consume_processing(self, duration_s: float = 0.001) -> float:
        """Energy consumed during CRDT merge or other processing."""
        # Processing power ≈ idle power (CPU active but no radio)
        energy = self.idle_power * duration_s
        self._consume(energy)
        return energy

    def _consume(self, energy_j: float):
        """Deduct energy from remaining budget."""
        self.remaining_energy = max(0, self.remaining_energy - energy_j)
        self.total_consumed += energy_j

    @property
    def residual_fraction(self) -> float:
        """Remaining energy as fraction of initial (0.0 to 1.0)."""
        if self.initial_energy <= 0:
            return 0.0
        return self.remaining_energy / self.initial_energy

    @property
    def is_alive(self) -> bool:
        """True if node has energy remaining."""
        return self.remaining_energy > 0

    def __repr__(self) -> str:
        return f"Energy({self.residual_fraction:.1%} remaining, {self.total_consumed:.2f}J consumed)"


# ─────────────────────────────────────────────────────────────────────
# Neighbor Information
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NeighborInfo:
    """
    Information about a neighboring node, maintained via HELLO exchanges.

    Extended fields for richer ML features (v2):
      rssi_history    – ring buffer of last 3 RSSI readings (mobility proxy)
      relay_timestamps – rolling 60s relay success log (delivery history)
      neighbor_set    – their reported neighbour IDs (Jaccard overlap)
      two_hop_degree  – sum of their neighbours' degree counts (2-hop approx)
    """
    node_id: int = 0
    last_seen: float = 0.0
    rssi: float = -80.0
    success_rate: float = 0.5        # Exponential moving average of delivery success
    residual_energy: float = 1.0     # Last reported energy fraction
    buffer_free: float = 1.0         # Last reported buffer availability
    degree: int = 0                  # Number of neighbors they reported
    msg_count: int = 0               # Number of messages they have
    state_hash: int = 0              # Their CRDT state hash
    tx_attempts: int = 0             # Our transmission attempts to them
    tx_successes: int = 0            # Successful transmissions to them

    # ── New ML-feature fields ──────────────────────────────────────────
    rssi_history: List[Tuple[float, float]] = field(default_factory=list)
    relay_timestamps: List[Tuple[float, bool]] = field(default_factory=list)
    neighbor_set: Set[int] = field(default_factory=set)
    two_hop_degree: int = 0

    # ── Helpers ───────────────────────────────────────────────────────

    def update_success_rate(self, success: bool, alpha: float = 0.2):
        """Update success rate with exponential moving average."""
        self.tx_attempts += 1
        if success:
            self.tx_successes += 1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)

    def record_rssi(self, sim_time: float, rssi: float):
        """Keep a ring buffer of the last 3 RSSI readings."""
        self.rssi_history.append((sim_time, rssi))
        if len(self.rssi_history) > 3:
            self.rssi_history.pop(0)

    def mobility_proxy(self) -> float:
        """
        Rate-of-change of RSSI over last 3 beacons, normalised to [-1, 1].
        Large negative value → node moving out of range.
        """
        if len(self.rssi_history) < 2:
            return 0.0
        t0, r0 = self.rssi_history[0]
        t1, r1 = self.rssi_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return max(-1.0, min(1.0, (r1 - r0) / dt / 5.0))

    def record_relay(self, sim_time: float, delivered: bool):
        """Log a relay event; prune events older than 60 s."""
        self.relay_timestamps.append((sim_time, delivered))
        cutoff = sim_time - 60.0
        self.relay_timestamps = [(t, d) for t, d in self.relay_timestamps if t >= cutoff]

    def delivery_history_ratio(self) -> float:
        """Rolling 60s relay delivery ratio (neutral prior 0.5 if no data)."""
        if not self.relay_timestamps:
            return 0.5
        return sum(1 for _, d in self.relay_timestamps if d) / len(self.relay_timestamps)

    def jaccard_overlap(self, my_neighbor_ids: Set[int]) -> float:
        """Jaccard similarity between our neighbour set and theirs."""
        union = my_neighbor_ids | self.neighbor_set
        if not union:
            return 0.0
        return len(my_neighbor_ids & self.neighbor_set) / len(union)


# ─────────────────────────────────────────────────────────────────────
# Packet Types
# ─────────────────────────────────────────────────────────────────────

class PacketType(Enum):
    """Types of protocol packets."""
    HELLO = auto()
    DATA = auto()
    SUMMARY = auto()
    REQUEST = auto()
    DELTA = auto()


@dataclass
class Packet:
    """
    Protocol packet transmitted over the wireless channel.
    """
    type: PacketType
    sender_id: int
    data: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 64
    created_at: float = 0.0

    @staticmethod
    def make_hello(sender_id: int, time: float, energy: float,
                   buffer_free: float, degree: int, msg_count: int,
                   state_hash: int, neighbor_ids: list = None,
                   two_hop_degree: int = 0) -> 'Packet':
        """Create a HELLO beacon packet."""
        return Packet(
            type=PacketType.HELLO,
            sender_id=sender_id,
            data={
                'energy': energy,
                'buffer_free': buffer_free,
                'degree': degree,
                'msg_count': msg_count,
                'state_hash': state_hash,
                'neighbor_ids': neighbor_ids or [],
                'two_hop_degree': two_hop_degree,
            },
            size_bytes=40,   # slightly larger with neighbor list
            created_at=time,
        )

    @staticmethod
    def make_data(sender_id: int, message: Message, time: float) -> 'Packet':
        """Create a DATA packet carrying a message."""
        return Packet(
            type=PacketType.DATA,
            sender_id=sender_id,
            data={
                'message': message,
            },
            size_bytes=message.size_bytes + 20,  # message + header
            created_at=time,
        )

    @staticmethod
    def make_summary(sender_id: int, msg_ids: Set[MessageId],
                     state_hash: int, time: float) -> 'Packet':
        """Create a SUMMARY packet with message ID list."""
        return Packet(
            type=PacketType.SUMMARY,
            sender_id=sender_id,
            data={
                'msg_ids': msg_ids,
                'state_hash': state_hash,
            },
            size_bytes=max(32, 4 + len(msg_ids) * 8),
            created_at=time,
        )

    @staticmethod
    def make_request(sender_id: int, missing_ids: Set[MessageId],
                     time: float) -> 'Packet':
        """Create a REQUEST packet asking for missing messages."""
        return Packet(
            type=PacketType.REQUEST,
            sender_id=sender_id,
            data={
                'missing_ids': missing_ids,
            },
            size_bytes=max(16, 4 + len(missing_ids) * 8),
            created_at=time,
        )

    @staticmethod
    def make_delta(sender_id: int, messages: List[Message],
                   time: float) -> 'Packet':
        """Create a DELTA packet with missing messages."""
        total_size = 8 + sum(m.size_bytes for m in messages)
        return Packet(
            type=PacketType.DELTA,
            sender_id=sender_id,
            data={
                'messages': messages,
            },
            size_bytes=total_size,
            created_at=time,
        )


# ─────────────────────────────────────────────────────────────────────
# Constrained Node
# ─────────────────────────────────────────────────────────────────────

class NodeState(Enum):
    """Node operational state."""
    ACTIVE = auto()
    FAILED = auto()
    ENERGY_DEPLETED = auto()


class ConstrainedNode:
    """
    Resource-constrained wireless node with CRDT synchronization.

    Implements the complete protocol stack:
    - Physical: Energy model, position, radio
    - Transport: Neighbor discovery via HELLO beacons
    - Synchronization: Anti-entropy gossip with state hashing
    - Application: OR-Set message log with HLC causality

    The node operates as a state machine:
        ACTIVE: Normal operation (generate, receive, forward, sync)
        FAILED: Crashed, does not respond (crash-recovery model)
        ENERGY_DEPLETED: Battery dead, permanently offline
    """

    def __init__(
        self,
        node_id: int,
        position: Tuple[float, float],
        forwarding_strategy: ForwardingStrategy,
        max_buffer_size: int = 256,
        initial_energy_j: float = 26640.0,
        message_gen_rate: float = 0.1,      # messages per second
        hello_interval: float = 2.0,        # seconds
        sync_interval: float = 1.0,         # seconds
        adaptive_sync: bool = True,         # enable adaptive sync
        fine_sync_interval: float = 0.3,    # fast sync when active
        coarse_sync_interval: float = 5.0,  # slow sync when idle
    ):
        # Identity
        self.node_id = node_id
        self.position = position
        self.state = NodeState.ACTIVE

        # CRDT state
        self.message_log = ORSet(max_elements=max_buffer_size)
        self.hlc = HybridLogicalClock(node_id=node_id)
        self.status_register = LWWRegister()
        self.emergency_counter = PNCounter()

        # Duplicate suppression
        self.seen_ids: Set[MessageId] = set()

        # Protocol
        self.neighbors: Dict[int, NeighborInfo] = {}
        self.forwarding = forwarding_strategy
        self.message_gen_rate = message_gen_rate
        self.hello_interval = hello_interval
        self.sync_interval = sync_interval
        self.adaptive_sync = adaptive_sync
        self.fine_sync_interval = fine_sync_interval
        self.coarse_sync_interval = coarse_sync_interval

        # Internal counters
        self._local_seq = 0
        self._current_time = 0.0
        self._last_activity = 0.0
        self._msgs_generated = 0
        self._msgs_received = 0
        self._msgs_forwarded = 0
        self._msgs_duplicated = 0

        # Energy
        self.energy = EnergyModel(initial_energy_j=initial_energy_j)

        # Scheduled event IDs (for cancellation)
        self._hello_event_id: Optional[str] = None
        self._sync_event_id: Optional[str] = None
        self._generate_event_id: Optional[str] = None

        # Buffer constraint
        self.max_buffer_size = max_buffer_size

    @property
    def current_time(self) -> float:
        """Current simulation time known to this node."""
        return self._current_time

    @property
    def buffer_free_fraction(self) -> float:
        """Fraction of message buffer that is unused."""
        used = self.message_log.size
        return max(0.0, 1.0 - used / self.max_buffer_size)

    @property
    def is_alive(self) -> bool:
        return self.state == NodeState.ACTIVE and self.energy.is_alive

    # ─────────────────────────────────────────────────────────────
    # MESSAGE GENERATION
    # ─────────────────────────────────────────────────────────────

    def generate_message(self, sim_time: float, metrics: 'MetricsCollector') -> Optional[Message]:
        """
        Generate a new crisis message.

        Creates a message with a unique ID, timestamps it with HLC,
        adds it to the local OR-Set, and prepares it for forwarding.
        """
        self._current_time = sim_time

        if not self.is_alive:
            return None

        # Buffer full check
        if self.message_log.size >= self.max_buffer_size:
            return None

        # Advance HLC
        phys_ms = int(sim_time * 1000)
        self.hlc.tick(phys_ms)

        # Create message
        msg_id = MessageId(origin=self.node_id, seq=self._local_seq)
        self._local_seq += 1

        message = Message(
            id=msg_id,
            hlc_pt=self.hlc.pt,
            hlc_lc=self.hlc.lc,
            payload=f"ALERT from node {self.node_id} at t={sim_time:.1f}",
            priority=0,
            created_at=sim_time,
            hop_count=0,
            ttl=32,
            size_bytes=64,
        )

        # Add to local OR-Set
        self.message_log.add(message, self.node_id, self.hlc)
        self.seen_ids.add(msg_id)
        self._msgs_generated += 1
        self._last_activity = sim_time

        # Log
        metrics.log_event(
            time=sim_time,
            node_id=self.node_id,
            event_type=__import__('simulator.metrics', fromlist=['EventType']).EventType.MSG_CREATED,
            msg_origin=msg_id.origin,
            msg_seq=msg_id.seq,
        )

        # Consume processing energy
        self.energy.consume_processing(0.0005)
        metrics.log_energy(self.node_id, 0.0005 * self.energy.idle_power)

        return message

    # ─────────────────────────────────────────────────────────────
    # PACKET HANDLING
    # ─────────────────────────────────────────────────────────────

    def handle_packet(
        self,
        packet: Packet,
        rssi: float,
        sim_time: float,
        metrics: 'MetricsCollector',
    ) -> List[Tuple[Packet, List[int]]]:
        """
        Handle an incoming packet and return packets to transmit.

        Returns:
            List of (packet, recipient_ids) tuples to be transmitted
            by the network layer.
        """
        from .metrics import EventType

        self._current_time = sim_time

        if not self.is_alive:
            return []

        # Consume RX energy
        rx_energy = self.energy.consume_rx(packet.size_bytes)
        metrics.log_energy(self.node_id, rx_energy)

        # Update neighbor info
        self._update_neighbor(packet.sender_id, rssi, sim_time, packet)

        outgoing: List[Tuple[Packet, List[int]]] = []

        if packet.type == PacketType.HELLO:
            self._handle_hello(packet, rssi, sim_time, metrics)

        elif packet.type == PacketType.DATA:
            out = self._handle_data(packet, sim_time, metrics)
            if out:
                outgoing.extend(out)

        elif packet.type == PacketType.SUMMARY:
            out = self._handle_summary(packet, sim_time, metrics)
            if out:
                outgoing.extend(out)

        elif packet.type == PacketType.REQUEST:
            out = self._handle_request(packet, sim_time, metrics)
            if out:
                outgoing.extend(out)

        elif packet.type == PacketType.DELTA:
            out = self._handle_delta(packet, sim_time, metrics)
            if out:
                outgoing.extend(out)

        return outgoing

    def _handle_hello(self, packet: Packet, rssi: float,
                      sim_time: float, metrics: 'MetricsCollector'):
        """Process HELLO beacon from a neighbor."""
        from .metrics import EventType

        sender = packet.sender_id
        data = packet.data

        if sender not in self.neighbors:
            self.neighbors[sender] = NeighborInfo(node_id=sender)

        nbr = self.neighbors[sender]
        nbr.last_seen = sim_time
        nbr.rssi = rssi
        nbr.residual_energy = data.get('energy', 1.0)
        nbr.buffer_free = data.get('buffer_free', 1.0)
        nbr.degree = data.get('degree', 0)
        nbr.msg_count = data.get('msg_count', 0)
        nbr.state_hash = data.get('state_hash', 0)

        # ── New ML feature tracking ────────────────────────────────────
        nbr.record_rssi(sim_time, rssi)
        nbr.two_hop_degree = data.get('two_hop_degree', 0)
        nbr.neighbor_set = set(data.get('neighbor_ids', []))

    def _handle_data(self, packet: Packet, sim_time: float,
                     metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """Process incoming DATA message."""
        from .metrics import EventType

        message: Message = packet.data['message']
        msg_id = message.id

        # Duplicate check
        if msg_id in self.seen_ids:
            self._msgs_duplicated += 1
            metrics.log_event(
                time=sim_time, node_id=self.node_id,
                event_type=EventType.MSG_DUPLICATE,
                msg_origin=msg_id.origin, msg_seq=msg_id.seq,
                size_bytes=packet.size_bytes,
            )
            return []

        # TTL check
        if message.hop_count >= message.ttl:
            metrics.log_event(
                time=sim_time, node_id=self.node_id,
                event_type=EventType.MSG_DROPPED_TTL,
                msg_origin=msg_id.origin, msg_seq=msg_id.seq,
            )
            return []

        # Buffer full check
        if self.message_log.size >= self.max_buffer_size:
            metrics.log_event(
                time=sim_time, node_id=self.node_id,
                event_type=EventType.MSG_DROPPED_BUFFER,
                msg_origin=msg_id.origin, msg_seq=msg_id.seq,
            )
            return []

        # Accept message
        self.seen_ids.add(msg_id)
        self._msgs_received += 1

        # Update HLC from message
        phys_ms = int(sim_time * 1000)
        self.hlc.update(phys_ms, message.hlc_pt, message.hlc_lc)

        # Add to OR-Set (CRDT merge)
        forwarded_msg = Message(
            id=msg_id,
            hlc_pt=message.hlc_pt,
            hlc_lc=message.hlc_lc,
            payload=message.payload,
            priority=message.priority,
            created_at=message.created_at,
            hop_count=message.hop_count + 1,
            ttl=message.ttl,
            size_bytes=message.size_bytes,
        )
        self.message_log.add(forwarded_msg, self.node_id, self.hlc)
        self._last_activity = sim_time

        # Processing energy for CRDT merge
        merge_energy = self.energy.consume_processing(0.001)
        metrics.log_energy(self.node_id, merge_energy)

        # Log delivery
        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.MSG_DELIVERED,
            msg_origin=msg_id.origin, msg_seq=msg_id.seq,
            size_bytes=packet.size_bytes,
            hop_count=forwarded_msg.hop_count,
            latency=sim_time - message.created_at,
        )

        # Forward to selected neighbors
        outgoing = self._forward_message(forwarded_msg, sim_time, metrics)

        return outgoing

    def _handle_summary(self, packet: Packet, sim_time: float,
                        metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """
        Handle SUMMARY packet during anti-entropy sync.

        Compare state hashes; if different, send REQUEST for missing messages.
        """
        from .metrics import EventType

        remote_ids: Set[MessageId] = packet.data.get('msg_ids', set())
        remote_hash: int = packet.data.get('state_hash', 0)

        my_hash = self.message_log.state_hash()
        my_ids = self.message_log.element_ids()

        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.SYNC_SUMMARY,
            size_bytes=packet.size_bytes,
        )

        # Check if states differ
        if my_hash == remote_hash:
            return []  # Already in sync

        # Find messages they have that we don't
        missing = remote_ids - my_ids - self.seen_ids

        if not missing:
            # We might have messages they don't — send our summary back
            # (handled by the caller's symmetric sync)
            return []

        # Send REQUEST for missing messages
        request = Packet.make_request(self.node_id, missing, sim_time)
        tx_energy = self.energy.consume_tx(request.size_bytes)
        metrics.log_energy(self.node_id, tx_energy)

        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.SYNC_REQUEST,
            size_bytes=request.size_bytes,
        )

        return [(request, [packet.sender_id])]

    def _handle_request(self, packet: Packet, sim_time: float,
                        metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """
        Handle REQUEST packet — send requested messages as DELTA.
        """
        from .metrics import EventType

        requested_ids: Set[MessageId] = packet.data.get('missing_ids', set())

        # Collect requested messages from our log
        messages_to_send = []
        for mid in requested_ids:
            msg = self.message_log.get(mid)
            if msg is not None:
                messages_to_send.append(msg)

        if not messages_to_send:
            return []

        # Send DELTA response
        delta = Packet.make_delta(self.node_id, messages_to_send, sim_time)
        tx_energy = self.energy.consume_tx(delta.size_bytes)
        metrics.log_energy(self.node_id, tx_energy)

        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.SYNC_DELTA,
            size_bytes=delta.size_bytes,
        )

        return [(delta, [packet.sender_id])]

    def _handle_delta(self, packet: Packet, sim_time: float,
                      metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """
        Handle DELTA packet — merge received messages into local OR-Set.
        """
        from .metrics import EventType

        messages: List[Message] = packet.data.get('messages', [])
        new_count = 0

        for msg in messages:
            if msg.id not in self.seen_ids:
                self.seen_ids.add(msg.id)
                self.message_log.add(msg, self.node_id, self.hlc)
                self._msgs_received += 1
                new_count += 1

                metrics.log_event(
                    time=sim_time, node_id=self.node_id,
                    event_type=EventType.MSG_DELIVERED,
                    msg_origin=msg.id.origin, msg_seq=msg.id.seq,
                    size_bytes=msg.size_bytes,
                    hop_count=msg.hop_count,
                    latency=sim_time - msg.created_at,
                    via='delta_sync',
                )
            else:
                self._msgs_duplicated += 1

        if new_count > 0:
            merge_energy = self.energy.consume_processing(0.001 * new_count)
            metrics.log_energy(self.node_id, merge_energy)

            metrics.log_event(
                time=sim_time, node_id=self.node_id,
                event_type=EventType.SYNC_MERGE,
                new_messages=new_count,
            )

        self._last_activity = sim_time
        return []

    # ─────────────────────────────────────────────────────────────
    # FORWARDING
    # ─────────────────────────────────────────────────────────────

    def _forward_message(self, message: Message, sim_time: float,
                         metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """Select neighbors and prepare forwarding packets."""
        from .metrics import EventType

        if not self.neighbors:
            return []

        # Use forwarding strategy to select recipients
        decision = self.forwarding.select_recipients(
            self, message, self.neighbors,
        )

        if not decision.selected_neighbors:
            return []

        # Create DATA packet
        data_pkt = Packet.make_data(self.node_id, message, sim_time)

        # Consume TX energy
        tx_energy = self.energy.consume_tx(data_pkt.size_bytes)
        metrics.log_energy(self.node_id, tx_energy)

        self._msgs_forwarded += 1

        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.MSG_FORWARDED,
            msg_origin=message.id.origin, msg_seq=message.id.seq,
            strategy=decision.strategy_name,
            recipients=len(decision.selected_neighbors),
            size_bytes=data_pkt.size_bytes,
        )

        # Log forwarding trace for ML training
        if decision.features:
            for nid in decision.selected_neighbors:
                if nid in decision.scores:
                    trace = {
                        'time': sim_time,
                        'forwarder': self.node_id,
                        'neighbor': nid,
                        'msg_origin': message.id.origin,
                        'msg_seq': message.id.seq,
                        'score': decision.scores[nid],
                        'strategy': decision.strategy_name,
                    }
                    if nid in (decision.features or {}):
                        feats = decision.features[nid]
                        for i, fname in enumerate([
                            'success_rate', 'rssi_norm', 'energy',
                            'buffer_free', 'degree_norm', 'freshness',
                            'hop_norm', 'age_norm', 'two_hop_norm',
                            'delivery_hist', 'mobility', 'jaccard',
                        ]):
                            if i < len(feats):
                                trace[fname] = feats[i]
                    metrics.log_forwarding_trace(trace)

        return [(data_pkt, decision.selected_neighbors)]

    # ─────────────────────────────────────────────────────────────
    # PERIODIC PROTOCOL EVENTS
    # ─────────────────────────────────────────────────────────────

    def create_hello_packet(self, sim_time: float) -> Packet:
        """Create a HELLO beacon for neighbor discovery."""
        self._current_time = sim_time
        two_hop = sum(nbr.degree for nbr in self.neighbors.values())
        return Packet.make_hello(
            sender_id=self.node_id,
            time=sim_time,
            energy=self.energy.residual_fraction,
            buffer_free=self.buffer_free_fraction,
            degree=len(self.neighbors),
            msg_count=self.message_log.size,
            state_hash=self.message_log.state_hash(),
            neighbor_ids=list(self.neighbors.keys()),
            two_hop_degree=two_hop,
        )

    def create_sync_packets(self, sim_time: float,
                            metrics: 'MetricsCollector') -> List[Tuple[Packet, List[int]]]:
        """
        Create synchronization packets for anti-entropy gossip.

        Sends SUMMARY to neighbors whose state hash differs from ours.
        Implements the adaptive sync heuristic:
            - Fine sync (0.3s) when recent activity detected
            - Coarse sync (5.0s) when idle
        """
        from .metrics import EventType

        self._current_time = sim_time

        if not self.is_alive or not self.neighbors:
            return []

        my_hash = self.message_log.state_hash()
        my_ids = self.message_log.element_ids()

        outgoing = []

        # Select neighbors to sync with (prioritize those with different state)
        sync_targets = []
        for nid, nbr in self.neighbors.items():
            # Skip stale neighbors (not seen for > 30s)
            if sim_time - nbr.last_seen > 30.0:
                continue

            # Prioritize neighbors with different state hash
            if nbr.state_hash != my_hash:
                sync_targets.insert(0, nid)  # priority
            elif nbr.state_hash == my_hash:
                # Already synced — skip
                continue
            else:
                sync_targets.append(nid)

        if not sync_targets:
            return []

        # Send SUMMARY to selected targets (limit to top 3 to save energy)
        for nid in sync_targets[:3]:
            summary = Packet.make_summary(self.node_id, my_ids, my_hash, sim_time)
            tx_energy = self.energy.consume_tx(summary.size_bytes)
            metrics.log_energy(self.node_id, tx_energy)

            metrics.log_event(
                time=sim_time, node_id=self.node_id,
                event_type=EventType.SYNC_SUMMARY,
                size_bytes=summary.size_bytes,
            )

            outgoing.append((summary, [nid]))

        return outgoing

    def get_adaptive_sync_interval(self, sim_time: float) -> float:
        """
        Compute adaptive sync interval based on recent activity.

        Fine sync: 0.3s when activity within last 5 seconds.
        Coarse sync: 5.0s when idle for > 5 seconds.
        """
        if not self.adaptive_sync:
            return self.sync_interval

        time_since_activity = sim_time - self._last_activity
        if time_since_activity < 5.0:
            return self.fine_sync_interval
        else:
            return self.coarse_sync_interval

    # ─────────────────────────────────────────────────────────────
    # NEIGHBOR MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def _update_neighbor(self, sender_id: int, rssi: float,
                         sim_time: float, packet: Packet):
        """Update neighbor table from received packet."""
        if sender_id not in self.neighbors:
            self.neighbors[sender_id] = NeighborInfo(node_id=sender_id)

        nbr = self.neighbors[sender_id]
        nbr.last_seen = sim_time
        nbr.rssi = rssi
        nbr.update_success_rate(True)

    def prune_stale_neighbors(self, sim_time: float, timeout: float = 30.0):
        """Remove neighbors not heard from within timeout."""
        stale = [
            nid for nid, nbr in self.neighbors.items()
            if sim_time - nbr.last_seen > timeout
        ]
        for nid in stale:
            del self.neighbors[nid]

    # ─────────────────────────────────────────────────────────────
    # NODE FAILURE / RECOVERY
    # ─────────────────────────────────────────────────────────────

    def fail(self, sim_time: float, metrics: 'MetricsCollector'):
        """Simulate node crash."""
        from .metrics import EventType

        self.state = NodeState.FAILED
        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.NODE_FAILURE,
        )

    def recover(self, sim_time: float, metrics: 'MetricsCollector'):
        """
        Recover from crash.

        In crash-recovery model, CRDT state persisted in flash
        is retained, but volatile neighbor table is lost.
        """
        from .metrics import EventType

        self.state = NodeState.ACTIVE
        self.neighbors.clear()  # Volatile state lost
        self._current_time = sim_time
        self._last_activity = sim_time

        metrics.log_event(
            time=sim_time, node_id=self.node_id,
            event_type=EventType.NODE_RECOVERY,
        )

    # ─────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return node-level statistics."""
        return {
            'node_id': self.node_id,
            'state': self.state.name,
            'msgs_generated': self._msgs_generated,
            'msgs_received': self._msgs_received,
            'msgs_forwarded': self._msgs_forwarded,
            'msgs_duplicated': self._msgs_duplicated,
            'buffer_size': self.message_log.size,
            'buffer_free': self.buffer_free_fraction,
            'energy_remaining': self.energy.residual_fraction,
            'energy_consumed': self.energy.total_consumed,
            'neighbors': len(self.neighbors),
            'crdt_metadata_bytes': self.message_log.metadata_size_bytes(),
        }

    def __repr__(self) -> str:
        return (
            f"Node({self.node_id}, pos={self.position}, "
            f"state={self.state.name}, "
            f"msgs={self.message_log.size}, "
            f"energy={self.energy.residual_fraction:.1%})"
        )
