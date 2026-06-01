"""
Unit tests for CRDT correctness properties.

Verifies the join-semilattice axioms for every CRDT type:
    - Commutativity:   merge(a, b) == merge(b, a)
    - Associativity:   merge(merge(a, b), c) == merge(a, merge(b, c))
    - Idempotence:     merge(a, a) == a

Also tests HLC monotonicity and OR-Set add/remove semantics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from simulator.crdt import (
    HybridLogicalClock, Message, MessageId, ORSet,
    LWWRegister, GCounter, PNCounter, UniqueTag,
)


# ─── Helpers ─────────────────────────────────────────────────────────

def make_msg(origin: int, seq: int, payload: str = "x") -> Message:
    return Message(id=MessageId(origin, seq), payload=payload, created_at=0.0)


def orset_active_ids(s: ORSet):
    return s.element_ids()


# ─── HLC ─────────────────────────────────────────────────────────────

class TestHybridLogicalClock:

    def test_tick_monotonic(self):
        hlc = HybridLogicalClock(node_id=0)
        prev_pt, prev_lc = hlc.tick(1000)
        for t in [1000, 1000, 1001, 1005]:
            pt, lc = hlc.tick(t)
            assert (pt, lc) > (prev_pt, prev_lc)
            prev_pt, prev_lc = pt, lc

    def test_tick_never_below_physical(self):
        hlc = HybridLogicalClock(node_id=0)
        for t in [500, 1000, 800, 1200]:
            pt, _ = hlc.tick(t)
            assert pt >= t

    def test_update_advances_past_remote(self):
        hlc = HybridLogicalClock(node_id=0)
        hlc.tick(100)
        pt, lc = hlc.update(physical_now_ms=50, remote_pt=9000, remote_lc=0)
        assert pt >= 9000

    def test_update_breaks_tie_with_lc(self):
        hlc = HybridLogicalClock(node_id=0)
        hlc.tick(1000)
        # Same physical time as remote → lc must advance
        pt1, lc1 = hlc.update(physical_now_ms=1000, remote_pt=1000, remote_lc=5)
        assert lc1 > 5

    def test_logical_counter_resets_on_new_physical(self):
        hlc = HybridLogicalClock(node_id=0)
        hlc.tick(1000)
        hlc.tick(1000)  # lc becomes 1
        pt, lc = hlc.tick(2000)  # new physical time
        assert lc == 0


# ─── OR-Set ──────────────────────────────────────────────────────────

class TestORSet:

    def _populated(self):
        """Returns two OR-Sets with overlapping and distinct elements."""
        hlc = HybridLogicalClock(node_id=0)
        hlc.tick(1000)
        a, b = ORSet(), ORSet()
        m1, m2, m3 = make_msg(0, 1), make_msg(1, 1), make_msg(2, 1)
        a.add(m1, node_id=0, hlc=hlc); hlc.tick(1001)
        a.add(m2, node_id=0, hlc=hlc); hlc.tick(1002)
        b.add(m2, node_id=1, hlc=hlc); hlc.tick(1003)
        b.add(m3, node_id=1, hlc=hlc)
        return a, b, m1, m2, m3

    # Commutativity
    def test_merge_commutative(self):
        a, b, *_ = self._populated()
        ab = a.merge(b)
        ba = b.merge(a)
        assert orset_active_ids(ab) == orset_active_ids(ba)

    # Associativity
    def test_merge_associative(self):
        hlc = HybridLogicalClock()
        hlc.tick(1)
        a, b, c = ORSet(), ORSet(), ORSet()
        a.add(make_msg(0, 1), 0, hlc); hlc.tick(2)
        b.add(make_msg(1, 1), 1, hlc); hlc.tick(3)
        c.add(make_msg(2, 1), 2, hlc)
        assert orset_active_ids(a.merge(b).merge(c)) == orset_active_ids(a.merge(b.merge(c)))

    # Idempotence
    def test_merge_idempotent(self):
        a, b, *_ = self._populated()
        ab = a.merge(b)
        assert orset_active_ids(ab.merge(ab)) == orset_active_ids(ab)
        assert orset_active_ids(ab.merge(b)) == orset_active_ids(ab)

    # Add/remove semantics
    def test_add_then_lookup(self):
        hlc = HybridLogicalClock(); hlc.tick(1)
        s = ORSet()
        msg = make_msg(0, 99)
        s.add(msg, 0, hlc)
        assert s.lookup(msg.id)

    def test_remove_after_add(self):
        hlc = HybridLogicalClock(); hlc.tick(1)
        s = ORSet()
        msg = make_msg(0, 7)
        s.add(msg, 0, hlc)
        s.remove(msg.id)
        assert not s.lookup(msg.id)

    def test_concurrent_add_wins_over_remove(self):
        """Add on replica B concurrent with remove on replica A must survive merge."""
        hlc = HybridLogicalClock(); hlc.tick(1)
        a, b = ORSet(), ORSet()
        msg = make_msg(3, 3)
        # Both see the initial add
        tag = a.add(msg, 0, hlc); hlc.tick(2)
        b.add_with_tag(msg, tag)
        # A removes; B concurrently adds again with a fresh tag
        a.remove(msg.id)
        hlc.tick(3)
        b.add(msg, 1, hlc)
        merged = a.merge(b)
        assert merged.lookup(msg.id), "Concurrent add must survive remove"

    def test_gc_clears_tombstones(self):
        hlc = HybridLogicalClock(); hlc.tick(1)
        s = ORSet(max_elements=4)
        msgs = [make_msg(0, i) for i in range(8)]
        for i, m in enumerate(msgs):
            hlc.tick(i + 1)
            s.add(m, 0, hlc)
            s.remove(m.id)
        # After GC (triggered by exceeding max_elements), remove_set should be small
        assert len(s._remove_set) < 8, "GC should have cleared tombstones"

    def test_delta_since_contains_only_new(self):
        hlc = HybridLogicalClock(); hlc.tick(1)
        s = ORSet()
        m1, m2 = make_msg(0, 1), make_msg(0, 2)
        s.add(m1, 0, hlc); hlc.tick(2)
        s.add(m2, 0, hlc)
        delta = s.delta_since({m1.id})
        ids = orset_active_ids(delta)
        assert m2.id in ids
        assert m1.id not in ids


# ─── LWW-Register ────────────────────────────────────────────────────

class TestLWWRegister:

    def test_merge_commutative(self):
        a, b = LWWRegister(), LWWRegister()
        a.set("hello", pt=100, lc=0, node_id=0)
        b.set("world", pt=200, lc=0, node_id=1)
        assert a.merge(b).get() == b.merge(a).get()

    def test_merge_associative(self):
        a, b, c = LWWRegister(), LWWRegister(), LWWRegister()
        a.set("a", 100, 0, 0)
        b.set("b", 200, 0, 1)
        c.set("c", 300, 0, 2)
        assert a.merge(b).merge(c).get() == a.merge(b.merge(c)).get()

    def test_merge_idempotent(self):
        a = LWWRegister()
        a.set("x", 100, 0, 0)
        assert a.merge(a).get() == a.get()

    def test_higher_timestamp_wins(self):
        a, b = LWWRegister(), LWWRegister()
        a.set("old", pt=10, lc=0, node_id=0)
        b.set("new", pt=20, lc=0, node_id=1)
        assert a.merge(b).get() == "new"

    def test_tie_broken_by_node_id(self):
        a, b = LWWRegister(), LWWRegister()
        a.set("from-0", pt=10, lc=0, node_id=0)
        b.set("from-5", pt=10, lc=0, node_id=5)
        assert a.merge(b).get() == "from-5"


# ─── G-Counter ───────────────────────────────────────────────────────

class TestGCounter:

    def test_merge_commutative(self):
        a, b = GCounter(), GCounter()
        a.increment(0, 3); b.increment(1, 7)
        assert a.merge(b).value == b.merge(a).value

    def test_merge_associative(self):
        a, b, c = GCounter(), GCounter(), GCounter()
        a.increment(0, 1); b.increment(1, 2); c.increment(2, 3)
        assert a.merge(b).merge(c).value == a.merge(b.merge(c)).value

    def test_merge_idempotent(self):
        a = GCounter()
        a.increment(0, 5)
        assert a.merge(a).value == a.value

    def test_value_is_sum(self):
        g = GCounter()
        g.increment(0, 10); g.increment(1, 20); g.increment(2, 5)
        assert g.value == 35

    def test_merge_takes_pointwise_max(self):
        a, b = GCounter(), GCounter()
        a.increment(0, 10); b.increment(0, 3)
        assert a.merge(b)._counts[0] == 10


# ─── PN-Counter ──────────────────────────────────────────────────────

class TestPNCounter:

    def test_increment_decrement(self):
        c = PNCounter()
        c.increment(0, 10)
        c.decrement(0, 3)
        assert c.value == 7

    def test_merge_commutative(self):
        a, b = PNCounter(), PNCounter()
        a.increment(0, 5); b.decrement(1, 2)
        assert a.merge(b).value == b.merge(a).value

    def test_merge_idempotent(self):
        a = PNCounter()
        a.increment(0, 8); a.decrement(0, 3)
        assert a.merge(a).value == a.value
