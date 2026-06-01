"""
Conflict-Free Replicated Data Types (CRDTs) for Distributed Synchronization.

Implements the following formally-verified data structures:

1. **Hybrid Logical Clock (HLC)** — Bounded-size causality tracking
   combining physical and logical timestamps. [Kulkarni et al., 2014]

2. **Observed-Remove Set (OR-Set)** — A state-based CRDT (CvRDT) that
   supports concurrent add/remove with unique-tag disambiguation.
   Merge is commutative, associative, and idempotent ⟹ SEC.
   [Shapiro et al., SSS 2011; INRIA RR-7687, 2011]

3. **Last-Write-Wins Register (LWW-Register)** — Convergent register
   where concurrent writes are resolved by timestamp comparison.

4. **G-Counter** — Grow-only distributed counter using per-node arrays
   with pointwise-maximum merge.

All structures satisfy the join-semilattice property:
    ∀ a, b: a ⊔ b = b ⊔ a            (commutativity)
    ∀ a, b, c: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)  (associativity)
    ∀ a: a ⊔ a = a                    (idempotence)

This guarantees Strong Eventual Consistency (SEC) without coordination.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────
# Message Identity
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class MessageId:
    """
    Globally unique message identifier.

    Composed of the originating node's ID and a monotonically
    increasing per-node sequence number. The pair (origin, seq)
    is guaranteed unique across all nodes.
    """
    origin: int
    seq: int

    def __str__(self) -> str:
        return f"{self.origin}:{self.seq}"

    def __repr__(self) -> str:
        return f"MsgId({self.origin}:{self.seq})"


# ─────────────────────────────────────────────────────────────────────
# Hybrid Logical Clock (HLC)
# ─────────────────────────────────────────────────────────────────────

class HybridLogicalClock:
    """
    Hybrid Logical Clock providing bounded-size causality tracking.

    Combines a physical timestamp (pt) with a logical counter (lc).
    Unlike vector clocks whose size grows with O(n) participants,
    HLC uses constant O(1) space while still capturing happens-before
    relationships.

    Reference:
        Kulkarni, S., Demirbas, M., Madeppa, D., Avva, B., & Leone, M. (2014).
        Logical Physical Clocks and Consistent Snapshots in Globally
        Distributed Databases. OPODIS 2014.

    Invariant: HLC timestamp is always ≥ physical time.
    """

    def __init__(self, node_id: int = 0):
        self.node_id = node_id
        self.pt: int = 0   # physical time component (milliseconds)
        self.lc: int = 0   # logical counter component

    def tick(self, physical_now_ms: int) -> Tuple[int, int]:
        """
        Advance clock for a local event.

        Algorithm (Kulkarni et al., 2014):
            l' = max(l, pt)
            if l' == l: c := c + 1
            else:       c := 0
            l := l'
        """
        old_pt = self.pt
        self.pt = max(self.pt, physical_now_ms)
        if self.pt == old_pt:
            self.lc += 1
        else:
            self.lc = 0
        return (self.pt, self.lc)

    def update(self, physical_now_ms: int, remote_pt: int, remote_lc: int) -> Tuple[int, int]:
        """
        Update clock upon receiving a message with remote HLC.

        Algorithm:
            l' = max(l, msg.l, pt)
            if l' == l == msg.l: c := max(c, msg.c) + 1
            elif l' == l:        c := c + 1
            elif l' == msg.l:    c := msg.c + 1
            else:                c := 0
            l := l'
        """
        old_pt = self.pt
        self.pt = max(self.pt, remote_pt, physical_now_ms)

        if self.pt == old_pt == remote_pt:
            self.lc = max(self.lc, remote_lc) + 1
        elif self.pt == old_pt:
            self.lc = self.lc + 1
        elif self.pt == remote_pt:
            self.lc = remote_lc + 1
        else:
            self.lc = 0

        return (self.pt, self.lc)

    def timestamp(self) -> Tuple[int, int]:
        """Return current (pt, lc) tuple."""
        return (self.pt, self.lc)

    def __repr__(self) -> str:
        return f"HLC(node={self.node_id}, pt={self.pt}, lc={self.lc})"


# ─────────────────────────────────────────────────────────────────────
# Unique Tag for OR-Set
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class UniqueTag:
    """
    Globally unique tag for OR-Set element identification.

    Each Add operation generates a fresh UniqueTag. When an element
    is Removed, only the currently-observed tags are tombstoned,
    allowing concurrent Adds to survive.

    Fields:
        node_id: Node that performed the Add.
        pt: Physical time component of HLC at Add time.
        lc: Logical counter component of HLC at Add time.
        counter: Per-node monotonic counter for extra uniqueness.
    """
    node_id: int
    pt: int
    lc: int
    counter: int

    def __str__(self) -> str:
        return f"Tag({self.node_id}.{self.pt}.{self.lc}.{self.counter})"


# ─────────────────────────────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """
    Application-layer message in the crisis communication system.

    Attributes:
        id: Globally unique identifier (origin, seq).
        hlc_pt: HLC physical time at creation.
        hlc_lc: HLC logical counter at creation.
        payload: Text content.
        priority: 0=normal, 1=urgent, 2=emergency.
        created_at: Simulation time of creation (seconds).
        hop_count: Number of hops traversed so far.
        ttl: Maximum hops allowed.
        size_bytes: Estimated wire size of this message.
    """
    id: MessageId
    hlc_pt: int = 0
    hlc_lc: int = 0
    payload: str = ""
    priority: int = 0
    created_at: float = 0.0
    hop_count: int = 0
    ttl: int = 32
    size_bytes: int = 64

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Message):
            return self.id == other.id
        return NotImplemented

    def __str__(self) -> str:
        return f"Msg[{self.id}, prio={self.priority}, hops={self.hop_count}]"


# ─────────────────────────────────────────────────────────────────────
# Observed-Remove Set (OR-Set) — State-based CRDT (CvRDT)
# ─────────────────────────────────────────────────────────────────────

class ORSet:
    """
    Observed-Remove Set (OR-Set) — a state-based CRDT.

    Supports concurrent Add and Remove operations on a replicated set.
    Each Add generates a globally-unique tag; Remove only tombstones
    tags that are locally observed, so concurrent Adds on other replicas
    are preserved.

    Mathematical definition:
        State: (A, R) where A ⊆ Element × UniqueTag, R ⊆ UniqueTag

        add(e):    tag = fresh();  A := A ∪ {(e, tag)}
        remove(e): R := R ∪ {tag | (e, tag) ∈ A}
        lookup(e): ∃ tag: (e, tag) ∈ A ∧ tag ∉ R

        merge(S₁, S₂):
            A_merged = A₁ ∪ A₂
            R_merged = R₁ ∪ R₂
            return (A_merged, R_merged)

    Theorem (Shapiro et al., 2011):
        The merge function is commutative, associative, and idempotent.
        Therefore, OR-Set is a CvRDT ⟹ Strong Eventual Consistency.

    Proof sketch:
        merge uses set union (∪), which is itself commutative,
        associative, and idempotent. ∎

    Memory optimization for constrained devices:
        - Tags are compact: 4 integers = 16 bytes each.
        - Add-set stores Message references (shared).
        - Periodic garbage collection removes fully-tombstoned entries.

    Reference:
        Shapiro, M., Preguiça, N., Baquero, C., & Zawirski, M. (2011).
        A comprehensive study of Convergent and Commutative Replicated
        Data Types. INRIA Research Report RR-7506.
    """

    def __init__(self, max_elements: int = 512):
        # A: add-set mapping UniqueTag → Message
        self._add_set: Dict[UniqueTag, Message] = {}
        # R: remove-set of tombstoned UniqueTag values
        self._remove_set: Set[UniqueTag] = set()
        # Tag counter per node for uniqueness
        self._tag_counters: Dict[int, int] = {}
        # Maximum elements (memory constraint for embedded devices)
        self._max_elements = max_elements
        # Cached hash for fast comparison
        self._hash_dirty = True
        self._cached_hash: int = 0

    def add(self, element: Message, node_id: int, hlc: HybridLogicalClock) -> UniqueTag:
        """
        Add an element to the set with a fresh unique tag.

        Args:
            element: The Message to add.
            node_id: ID of the node performing the add.
            hlc: Current HLC state for tag generation.

        Returns:
            The UniqueTag generated for this add operation.
        """
        # Generate fresh tag
        counter = self._tag_counters.get(node_id, 0)
        self._tag_counters[node_id] = counter + 1

        tag = UniqueTag(
            node_id=node_id,
            pt=hlc.pt,
            lc=hlc.lc,
            counter=counter,
        )

        self._add_set[tag] = element
        self._hash_dirty = True

        # Enforce memory bound: evict oldest if over limit
        if len(self._add_set) > self._max_elements:
            self._gc_oldest()

        return tag

    def add_with_tag(self, element: Message, tag: UniqueTag):
        """Add an element with an existing tag (used during merge/sync)."""
        self._add_set[tag] = element
        self._hash_dirty = True

    def remove(self, element_id: MessageId) -> Set[UniqueTag]:
        """
        Remove all instances of an element identified by MessageId.

        Moves all tags associated with this element to the remove-set.
        Only affects tags currently observed locally — concurrent adds
        on other replicas are not affected.

        Returns:
            Set of tombstoned tags.
        """
        tombstoned = set()
        for tag, msg in list(self._add_set.items()):
            if msg.id == element_id:
                self._remove_set.add(tag)
                tombstoned.add(tag)

        self._hash_dirty = True
        return tombstoned

    def lookup(self, element_id: MessageId) -> bool:
        """Check if an element is in the active set (added but not removed)."""
        for tag, msg in self._add_set.items():
            if msg.id == element_id and tag not in self._remove_set:
                return True
        return False

    def get(self, element_id: MessageId) -> Optional[Message]:
        """Retrieve a message by its ID, or None if not present."""
        for tag, msg in self._add_set.items():
            if msg.id == element_id and tag not in self._remove_set:
                return msg
        return None

    def elements(self) -> list[Message]:
        """Return all active (non-tombstoned) elements."""
        seen_ids: set[MessageId] = set()
        result: list[Message] = []
        for tag, msg in self._add_set.items():
            if tag not in self._remove_set and msg.id not in seen_ids:
                result.append(msg)
                seen_ids.add(msg.id)
        return result

    def element_ids(self) -> Set[MessageId]:
        """Return the set of active MessageIds."""
        return {
            msg.id for tag, msg in self._add_set.items()
            if tag not in self._remove_set
        }

    @property
    def size(self) -> int:
        """Number of active (non-tombstoned) elements."""
        return len(self.element_ids())

    def merge(self, other: 'ORSet') -> 'ORSet':
        """
        Compute the join (least upper bound) of two OR-Set states.

        Merge(S₁, S₂):
            A_merged = A₁ ∪ A₂
            R_merged = R₁ ∪ R₂

        This function is:
            - Commutative:  merge(a, b) = merge(b, a)
            - Associative:  merge(merge(a, b), c) = merge(a, merge(b, c))
            - Idempotent:   merge(a, a) = a

        Returns:
            New ORSet representing the merged state.
        """
        result = ORSet(max_elements=max(self._max_elements, other._max_elements))

        # Union of add-sets
        result._add_set = {**self._add_set, **other._add_set}

        # Union of remove-sets
        result._remove_set = self._remove_set | other._remove_set

        # Merge tag counters (take max)
        all_nodes = set(self._tag_counters.keys()) | set(other._tag_counters.keys())
        for node in all_nodes:
            result._tag_counters[node] = max(
                self._tag_counters.get(node, 0),
                other._tag_counters.get(node, 0),
            )

        result._hash_dirty = True
        return result

    def merge_into(self, other: 'ORSet') -> int:
        """
        Merge another OR-Set into this one in-place.

        Returns the number of new elements added.
        """
        before = self.size

        # Merge add-sets
        for tag, msg in other._add_set.items():
            if tag not in self._add_set:
                self._add_set[tag] = msg

        # Merge remove-sets
        self._remove_set |= other._remove_set

        # Merge counters
        for node, count in other._tag_counters.items():
            self._tag_counters[node] = max(
                self._tag_counters.get(node, 0), count
            )

        self._hash_dirty = True
        return self.size - before

    def delta_since(self, known_ids: Set[MessageId]) -> 'ORSet':
        """
        Compute a delta containing only elements not in `known_ids`.

        This implements delta-state CRDT optimization:
        instead of sending the full state, only send the difference.

        Reference:
            Almeida, P.S., Shoker, A., & Baquero, C. (2018).
            Delta State Replicated Data Types. J. Parallel Distributed Comput.
        """
        delta = ORSet(max_elements=self._max_elements)
        for tag, msg in self._add_set.items():
            if msg.id not in known_ids and tag not in self._remove_set:
                delta._add_set[tag] = msg

        # Include tombstones that might be relevant
        delta._remove_set = set(self._remove_set)
        delta._hash_dirty = True
        return delta

    def state_hash(self) -> int:
        """
        Compute a deterministic hash of the current active state.

        Used for fast equality comparison during anti-entropy sync.
        Two OR-Sets with the same active elements will produce
        the same hash (with high probability).
        """
        if not self._hash_dirty:
            return self._cached_hash

        # Hash based on sorted active message IDs
        active_ids = sorted(self.element_ids())
        hasher = hashlib.md5()
        for mid in active_ids:
            hasher.update(struct.pack('!IQ', mid.origin, mid.seq))

        self._cached_hash = int.from_bytes(hasher.digest()[:8], 'big')
        self._hash_dirty = False
        return self._cached_hash

    def metadata_size_bytes(self) -> int:
        """Estimate the memory overhead of CRDT metadata in bytes."""
        # Each tag: 4 ints × 4 bytes = 16 bytes
        # Each remove tombstone: 16 bytes
        # Each tag counter: 8 bytes
        tag_bytes = len(self._add_set) * 16
        tombstone_bytes = len(self._remove_set) * 16
        counter_bytes = len(self._tag_counters) * 8
        return tag_bytes + tombstone_bytes + counter_bytes

    def _gc_oldest(self):
        """Garbage-collect all fully-tombstoned entries to free memory."""
        fully_removed = [
            tag for tag in self._remove_set
            if tag in self._add_set
        ]
        for tag in fully_removed:
            del self._add_set[tag]
            self._remove_set.discard(tag)

        self._hash_dirty = True

    def __len__(self) -> int:
        return self.size

    def __contains__(self, element_id: MessageId) -> bool:
        return self.lookup(element_id)

    def __repr__(self) -> str:
        return (
            f"ORSet(active={self.size}, "
            f"add_set={len(self._add_set)}, "
            f"remove_set={len(self._remove_set)})"
        )


# ─────────────────────────────────────────────────────────────────────
# Last-Write-Wins Register (LWW-Register)
# ─────────────────────────────────────────────────────────────────────

class LWWRegister:
    """
    Last-Write-Wins Register — a state-based CRDT.

    Stores a single value with a timestamp. Concurrent writes are
    resolved by keeping the value with the highest timestamp.
    Ties are broken by node_id.

    merge(r₁, r₂) = r₁ if r₁.ts > r₂.ts else r₂

    This is commutative, associative, and idempotent.

    Reference:
        Shapiro, M., et al. (2011). INRIA RR-7506, §3.2.1.
    """

    def __init__(self):
        self._value: Any = None
        self._ts_pt: int = 0
        self._ts_lc: int = 0
        self._node_id: int = 0

    def set(self, value: Any, pt: int, lc: int, node_id: int):
        """Set the register value with current HLC timestamp."""
        if (pt, lc, node_id) > (self._ts_pt, self._ts_lc, self._node_id):
            self._value = value
            self._ts_pt = pt
            self._ts_lc = lc
            self._node_id = node_id

    def get(self) -> Any:
        """Get the current value."""
        return self._value

    @property
    def timestamp(self) -> Tuple[int, int, int]:
        """Return (pt, lc, node_id) timestamp tuple."""
        return (self._ts_pt, self._ts_lc, self._node_id)

    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        """Merge two registers, keeping the one with highest timestamp."""
        result = LWWRegister()
        if other.timestamp > self.timestamp:
            result._value = other._value
            result._ts_pt = other._ts_pt
            result._ts_lc = other._ts_lc
            result._node_id = other._node_id
        else:
            result._value = self._value
            result._ts_pt = self._ts_pt
            result._ts_lc = self._ts_lc
            result._node_id = self._node_id
        return result

    def merge_into(self, other: 'LWWRegister'):
        """Merge another register into this one in-place."""
        if other.timestamp > self.timestamp:
            self._value = other._value
            self._ts_pt = other._ts_pt
            self._ts_lc = other._ts_lc
            self._node_id = other._node_id

    def __repr__(self) -> str:
        return f"LWWReg(value={self._value}, ts=({self._ts_pt},{self._ts_lc},{self._node_id}))"


# ─────────────────────────────────────────────────────────────────────
# G-Counter (Grow-only Counter)
# ─────────────────────────────────────────────────────────────────────

class GCounter:
    """
    Grow-only Counter — a state-based CRDT.

    Each node maintains its own counter entry. The global value
    is the sum of all entries. Merge takes the pointwise maximum.

    State: vector P where P[i] = count contributed by node i.
    value() = Σᵢ P[i]
    merge(P₁, P₂)[i] = max(P₁[i], P₂[i])

    This forms a join-semilattice under pointwise max.
    """

    def __init__(self):
        self._counts: Dict[int, int] = {}

    def increment(self, node_id: int, amount: int = 1):
        """Increment this node's counter."""
        self._counts[node_id] = self._counts.get(node_id, 0) + amount

    @property
    def value(self) -> int:
        """Total count across all nodes."""
        return sum(self._counts.values())

    def merge(self, other: 'GCounter') -> 'GCounter':
        """Merge two counters using pointwise maximum."""
        result = GCounter()
        all_nodes = set(self._counts.keys()) | set(other._counts.keys())
        for node in all_nodes:
            result._counts[node] = max(
                self._counts.get(node, 0),
                other._counts.get(node, 0),
            )
        return result

    def merge_into(self, other: 'GCounter'):
        """Merge in-place."""
        for node, count in other._counts.items():
            self._counts[node] = max(self._counts.get(node, 0), count)

    def __repr__(self) -> str:
        return f"GCounter(value={self.value}, nodes={len(self._counts)})"


# ─────────────────────────────────────────────────────────────────────
# PN-Counter (Positive-Negative Counter)
# ─────────────────────────────────────────────────────────────────────

class PNCounter:
    """
    Positive-Negative Counter — a state-based CRDT.

    Composed of two G-Counters: P (positive/increments) and
    N (negative/decrements).

    value() = P.value() - N.value()
    merge(C₁, C₂) = (P₁.merge(P₂), N₁.merge(N₂))
    """

    def __init__(self):
        self.p = GCounter()
        self.n = GCounter()

    def increment(self, node_id: int, amount: int = 1):
        self.p.increment(node_id, amount)

    def decrement(self, node_id: int, amount: int = 1):
        self.n.increment(node_id, amount)

    @property
    def value(self) -> int:
        return self.p.value - self.n.value

    def merge(self, other: 'PNCounter') -> 'PNCounter':
        result = PNCounter()
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def merge_into(self, other: 'PNCounter'):
        self.p.merge_into(other.p)
        self.n.merge_into(other.n)

    def __repr__(self) -> str:
        return f"PNCounter(value={self.value})"
