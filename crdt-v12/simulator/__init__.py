"""
Discrete-Event Network Simulator for CRDT-based Mesh Synchronization.

Implements constrained wireless nodes with Conflict-Free Replicated Data Types,
gossip-based anti-entropy synchronization, and multiple forwarding strategies
(flooding, heuristic, ML-assisted) for infrastructure-less disaster communication.
"""

from .engine import SimulationEngine, Event
from .crdt import HybridLogicalClock, ORSet, LWWRegister, GCounter, Message, MessageId, UniqueTag
from .wireless import WirelessChannel
from .node import ConstrainedNode, EnergyModel, NeighborInfo
from .network import MeshNetwork, NetworkConfig
from .forwarding import FloodStrategy, HeuristicStrategy, AIStrategy, AODVStrategy
from .wire import pack_packet, unpack_packet, benchmark as wire_benchmark
from .metrics import MetricsCollector, EventType

__all__ = [
    'SimulationEngine', 'Event',
    'HybridLogicalClock', 'ORSet', 'LWWRegister', 'GCounter',
    'Message', 'MessageId', 'UniqueTag',
    'WirelessChannel',
    'ConstrainedNode', 'EnergyModel', 'NeighborInfo',
    'MeshNetwork', 'NetworkConfig',
    'FloodStrategy', 'HeuristicStrategy', 'AIStrategy', 'AODVStrategy',
    'pack_packet', 'unpack_packet', 'wire_benchmark',
    'MetricsCollector', 'EventType',
]
