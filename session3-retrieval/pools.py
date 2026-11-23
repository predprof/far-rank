#!/usr/bin/env python3
"""Candidate-pool construction for FAR-Trans (A09).

One definition, imported by both the reachability analysis and the re-ranking
driver, so the pool a prompt sees is provably the pool that was measured.

Every pool is built from as-of information only: pre-cutoff Buy history,
pre-cutoff distinct-buyer popularity, and pre-cutoff content similarity. No
pool reads the evaluation window. All pools are capped at the same
`size` budget, so no variant can win by simply being larger.

    cbf                - the provided pool: content-based similarity order.
    pop                - popularity order (distinct historical buyers).
    own_then_cbf       - the user's own assets, most recently bought first,
                         then the CBF order.
    own_then_pop       - own assets most-recently-bought first, then popularity.
    own_freq_then_pop  - own assets most-frequently-bought first, then popularity.

The protocol allows repeat purchases, so an asset the customer already holds is
a legitimate candidate, not a leak.
"""
from __future__ import annotations

NAMES = ('cbf', 'pop', 'own_then_cbf', 'own_then_pop', 'own_freq_then_pop')


def _cap(seq, eligible, size):
    """First `size` distinct eligible IDs of `seq`, order preserved."""
    out, seen = [], set()
    for isin in seq:
        if isin in seen or isin not in eligible:
            continue
        seen.add(isin)
        out.append(isin)
        if len(out) >= size:
            break
    return out


def build(counts, eligible, cbf_order, pop_order, recent_buys, size):
    """Return {pool name: candidate id list} for one user.

    counts      -- Counter of pre-cutoff Buy counts per ISIN
    eligible    -- set of assets eligible at the cutoff
    cbf_order   -- full eligible order by content score (provided retrieval)
    pop_order   -- full eligible order by popularity
    recent_buys -- (timestamp, isin) pairs of pre-cutoff Buy events
    size        -- candidate budget, identical for every pool
    """
    # Most recently bought first; ISIN breaks same-day ties, matching the
    # starter's refusal to invent intraday chronology.
    own_recent = [isin for _, isin in
                  sorted(recent_buys, key=lambda p: (p[0], p[1]), reverse=True)]
    own_frequent = [isin for isin, _ in
                    sorted(counts.items(), key=lambda p: (-p[1], p[0]))]
    return {
        'cbf': _cap(cbf_order, eligible, size),
        'pop': _cap(pop_order, eligible, size),
        'own_then_cbf': _cap(own_recent + cbf_order, eligible, size),
        'own_then_pop': _cap(own_recent + pop_order, eligible, size),
        'own_freq_then_pop': _cap(own_frequent + pop_order, eligible, size),
    }


def orders_for(starter, counts, features, pop, eligible):
    """The two full orders a pool is cut from, using the starter's own scoring."""
    scores = starter.content_scores(counts, features)
    cold = not counts or not any(scores.values())
    if cold:
        scores = pop
    cbf_order = starter.order({i: scores[i] for i in eligible})
    pop_order = starter.order({i: pop[i] for i in eligible})
    return cbf_order, pop_order, cold
