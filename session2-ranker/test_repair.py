#!/usr/bin/env python3
"""Unit tests for the repair-and-complete policy."""
import unittest

from repair import RANK_LENGTH, repair_and_complete, salvage

# Twelve-character ISIN-shaped ids: two letters then ten alphanumerics.
POOL = [f"GR{i:010d}" for i in range(1, 51)]


class TestSalvage(unittest.TestCase):
    def test_clean_json_array_is_kept_in_order(self):
        wanted = POOL[7:17]
        raw = "[" + ", ".join(f'"{i}"' for i in wanted) + "]"
        self.assertEqual(salvage(raw, POOL), wanted)

    def test_out_of_pool_tokens_are_dropped_not_repaired(self):
        raw = f'["{POOL[0]}", "XX9999999999", "{POOL[1]}"]'
        self.assertEqual(salvage(raw, POOL), [POOL[0], POOL[1]])

    def test_duplicates_collapse_to_first_occurrence(self):
        raw = f'["{POOL[3]}", "{POOL[3]}", "{POOL[4]}"]'
        self.assertEqual(salvage(raw, POOL), [POOL[3], POOL[4]])

    def test_prose_around_the_array_still_salvages(self):
        raw = (f"Sure! Here are my picks: {POOL[2]} first, then {POOL[5]}. "
               "Hope this helps.")
        self.assertEqual(salvage(raw, POOL), [POOL[2], POOL[5]])

    def test_non_string_and_empty_yield_nothing(self):
        self.assertEqual(salvage(None, POOL), [])
        self.assertEqual(salvage("", POOL), [])
        self.assertEqual(salvage("no identifiers here", POOL), [])


class TestRepairAndComplete(unittest.TestCase):
    def assert_valid(self, ranking):
        self.assertEqual(len(ranking), RANK_LENGTH)
        self.assertEqual(len(set(ranking)), RANK_LENGTH)
        self.assertTrue(set(ranking).issubset(set(POOL)))

    def test_total_garbage_reproduces_the_provided_fallback(self):
        # With nothing salvageable the policy must equal ids[:10], i.e. the
        # provided fallback and the candidate_order_first10 control.
        self.assertEqual(repair_and_complete("total garbage", POOL),
                         POOL[:RANK_LENGTH])

    def test_partial_output_keeps_model_prefix_then_completes(self):
        raw = f'["{POOL[30]}", "{POOL[31]}"]'
        ranking = repair_and_complete(raw, POOL)
        self.assert_valid(ranking)
        self.assertEqual(ranking[:2], [POOL[30], POOL[31]])
        # Completion follows CBF order and skips already-used ids.
        self.assertEqual(ranking[2:], POOL[:8])

    def test_full_valid_output_passes_through_unchanged(self):
        wanted = POOL[20:30]
        raw = "[" + ", ".join(f'"{i}"' for i in wanted) + "]"
        self.assertEqual(repair_and_complete(raw, POOL), wanted)

    def test_more_than_ten_salvaged_is_truncated(self):
        wanted = POOL[0:15]
        raw = "[" + ", ".join(f'"{i}"' for i in wanted) + "]"
        self.assertEqual(repair_and_complete(raw, POOL), wanted[:RANK_LENGTH])

    def test_always_returns_a_valid_ranking_so_fallback_never_fires(self):
        for raw in ["", None, "[]", "nonsense", '["XX9999999999"]',
                    f'["{POOL[0]}"]' * 3]:
            self.assert_valid(repair_and_complete(raw, POOL))

    def test_short_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            repair_and_complete("x", POOL[:5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
