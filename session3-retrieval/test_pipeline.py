#!/usr/bin/env python3
"""Unit tests for the A09 pool construction and invalid-response policy.

Run: python3 -m unittest test_pipeline -v
"""
from __future__ import annotations

import unittest
from collections import Counter

import pools
from evaluate import repair_and_complete, salvage

A = [f'GR{i:010d}' for i in range(60)]          # synthetic in-pool ISINs
OUT = 'XX999999999Z'                             # never a candidate


class PoolConstruction(unittest.TestCase):
    def setUp(self):
        self.eligible = set(A)
        self.cbf = list(A)                       # content order
        self.pop = list(reversed(A))             # popularity order

    def build(self, counts, recent, size=50):
        return pools.build(Counter(counts), self.eligible, self.cbf, self.pop,
                           recent, size)

    def test_every_pool_respects_the_size_budget(self):
        built = self.build({A[0]: 3}, [('2021-01-01', A[0])], size=50)
        for name, pool in built.items():
            self.assertEqual(len(pool), 50, name)

    def test_no_pool_repeats_a_candidate(self):
        built = self.build({A[5]: 2, A[6]: 1},
                           [('2021-01-01', A[5]), ('2021-02-01', A[5])])
        for name, pool in built.items():
            self.assertEqual(len(pool), len(set(pool)), name)

    def test_cbf_pool_is_the_content_order_prefix(self):
        built = self.build({}, [], size=50)
        self.assertEqual(built['cbf'], self.cbf[:50])

    def test_pop_pool_is_the_popularity_order_prefix(self):
        built = self.build({}, [], size=50)
        self.assertEqual(built['pop'], self.pop[:50])

    def test_own_seed_leads_the_pool(self):
        """A previously bought asset must appear before the generic order."""
        built = self.build({A[59]: 4}, [('2021-03-01', A[59])])
        self.assertEqual(built['own_then_cbf'][0], A[59])
        self.assertEqual(built['own_freq_then_pop'][0], A[59])

    def test_own_recent_orders_by_date_descending(self):
        recent = [('2021-01-01', A[10]), ('2021-05-01', A[11]),
                  ('2021-03-01', A[12])]
        built = self.build({A[10]: 1, A[11]: 1, A[12]: 1}, recent)
        self.assertEqual(built['own_then_pop'][:3], [A[11], A[12], A[10]])

    def test_own_frequent_orders_by_count_then_isin(self):
        counts = {A[20]: 1, A[21]: 5, A[22]: 5}
        built = self.build(counts, [])
        # A[21] and A[22] tie on count 5; ISIN ascending breaks the tie.
        self.assertEqual(built['own_freq_then_pop'][:3], [A[21], A[22], A[20]])

    def test_same_day_ties_break_by_isin_not_invented_chronology(self):
        recent = [('2021-04-01', A[31]), ('2021-04-01', A[30])]
        built = self.build({A[30]: 1, A[31]: 1}, recent)
        self.assertEqual(built['own_then_pop'][:2], [A[31], A[30]])

    def test_ineligible_history_never_enters_a_pool(self):
        """An asset outside the eligible universe is dropped, not ranked."""
        built = self.build({OUT: 9}, [('2021-04-01', OUT)])
        for name, pool in built.items():
            self.assertNotIn(OUT, pool, name)

    def test_pools_are_subsets_of_the_eligible_universe(self):
        built = self.build({A[0]: 1}, [('2021-04-01', A[0])])
        for name, pool in built.items():
            self.assertTrue(set(pool) <= self.eligible, name)

    def test_small_universe_yields_short_but_valid_pools(self):
        tiny = set(A[:12])
        built = pools.build(Counter({A[0]: 1}), tiny, A[:12], A[:12][::-1],
                            [('2021-01-01', A[0])], 50)
        for name, pool in built.items():
            self.assertEqual(len(pool), 12, name)


class RepairPolicy(unittest.TestCase):
    """The A08 repair-and-complete policy, carried forward unchanged."""

    def setUp(self):
        self.pool = list(A[:50])

    def test_clean_response_is_kept_in_model_order(self):
        emitted = self.pool[20:30]
        raw = str(emitted).replace("'", '"')
        self.assertEqual(repair_and_complete(raw, self.pool), emitted)

    def test_out_of_pool_id_is_dropped_and_completed(self):
        emitted = self.pool[5:14] + [OUT]
        raw = str(emitted).replace("'", '"')
        result = repair_and_complete(raw, self.pool)
        self.assertEqual(len(result), 10)
        self.assertNotIn(OUT, result)
        self.assertEqual(result[:9], self.pool[5:14])

    def test_duplicates_collapse_and_order_is_first_emission(self):
        emitted = [self.pool[3], self.pool[3], self.pool[7]]
        raw = str(emitted).replace("'", '"')
        result = repair_and_complete(raw, self.pool)
        self.assertEqual(result[:2], [self.pool[3], self.pool[7]])
        self.assertEqual(len(set(result)), 10)

    def test_unsalvageable_response_degrades_to_the_control(self):
        """With nothing salvageable the policy must equal the pool's first ten."""
        for raw in ('', 'I cannot answer that.', None, 42, '[]'):
            self.assertEqual(repair_and_complete(raw, self.pool),
                             self.pool[:10], repr(raw))

    def test_completion_preserves_the_pools_own_order(self):
        raw = f'["{self.pool[40]}"]'
        result = repair_and_complete(raw, self.pool)
        self.assertEqual(result[0], self.pool[40])
        self.assertEqual(result[1:], [i for i in self.pool[:10]
                                      if i != self.pool[40]][:9])

    def test_prose_wrapped_response_is_salvaged(self):
        raw = f'Sure! Here you go: {self.pool[1]}, then {self.pool[2]}.'
        result = repair_and_complete(raw, self.pool)
        self.assertEqual(result[:2], [self.pool[1], self.pool[2]])

    def test_salvage_returns_only_pool_members(self):
        raw = f'["{self.pool[0]}", "{OUT}", "{self.pool[1]}"]'
        self.assertEqual(salvage(raw, self.pool), [self.pool[0], self.pool[1]])

    def test_short_pool_is_rejected_rather_than_padded(self):
        with self.assertRaises(ValueError):
            repair_and_complete('[]', self.pool[:9])

    def test_result_is_always_ten_unique_pool_members(self):
        for raw in ('[]', f'["{OUT}"]', f'["{self.pool[49]}"]',
                    str(self.pool[:10]).replace("'", '"')):
            result = repair_and_complete(raw, self.pool)
            self.assertEqual(len(result), 10)
            self.assertEqual(len(set(result)), 10)
            self.assertTrue(set(result) <= set(self.pool))



class ContextBudgetFailure(unittest.TestCase):
    """A generation that never happened must degrade to the matched control.

    The provided runtime enforces a fixed 8192-token budget and the provided
    prompt builder does not cap `buy_history_counts`, so a long-history
    customer cannot be generated for at all. Such a user is kept in the cohort
    with a fallback, and must be counted as a generation failure rather than a
    parse failure: there is no model text to salvage.
    """

    def setUp(self):
        self.pool = list(A[:50])

    def test_absent_response_yields_exactly_the_control(self):
        self.assertEqual(repair_and_complete(None, self.pool), self.pool[:10])

    def test_absent_response_salvages_nothing(self):
        self.assertEqual(salvage(None, self.pool), [])

    def test_both_policies_agree_when_there_is_no_text(self):
        """With no text the repaired and provided policies cannot differ."""
        import evaluate
        ranker = evaluate.import_provided('ranker', evaluate.LLM_DIR)
        with self.assertRaises(ValueError):
            ranker.parse_ranking(None, set(self.pool))
        self.assertEqual(repair_and_complete(None, self.pool), self.pool[:10])


if __name__ == '__main__':
    unittest.main()
