import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2_selection import one_se_choice, paired_gap, two_stage_select  # noqa: E402


class PairedGapTests(unittest.TestCase):
    def test_direction_and_standard_error(self):
        gap, se = paired_gap([11.0, 12.0, 13.0], [10.0, 10.0, 10.0], higher_is_better=False)
        self.assertAlmostEqual(gap, 2.0)
        self.assertAlmostEqual(se, np.std([1, 2, 3], ddof=1) / np.sqrt(3))
        gap, _ = paired_gap([0.3, 0.3], [0.4, 0.5], higher_is_better=True)
        self.assertAlmostEqual(gap, 0.15)


class OneStandardErrorTests(unittest.TestCase):
    def test_noisy_gap_prefers_simpler(self):
        scores = {"simple": [20.0, 23.0, 21.0, 25.0, 22.0], "complex": [20.5, 21.5, 21.8, 23.5, 22.4]}
        chosen, best, _ = one_se_choice(["simple", "complex"], scores, higher_is_better=False)
        self.assertEqual((chosen, best), ("simple", "complex"))

    def test_steady_gap_prefers_better(self):
        scores = {"simple": [21.0, 22.0, 23.0], "complex": [20.0, 21.1, 21.9]}
        chosen, _, _ = one_se_choice(["simple", "complex"], scores, higher_is_better=False)
        self.assertEqual(chosen, "complex")


class TwoStageTests(unittest.TestCase):
    groups = [("linear", "lin_plain", ["lin_plain", "lin_a", "lin_b"]),
              ("tree", "tree_plain", ["tree_plain", "tree_a"]),
              ("forest", None, ["forest_a", "forest_b"])]

    def test_tuning_then_one_se(self):
        s = {"lin_plain": [0.30, 0.30, 0.30], "lin_a": [0.32, 0.31, 0.33], "lin_b": [0.31, 0.30, 0.31],
             "tree_plain": [0.20, 0.22, 0.21], "tree_a": [0.25, 0.26, 0.24],
             "forest_a": [0.40, 0.41, 0.42], "forest_b": [0.38, 0.40, 0.39]}
        chosen, d = two_stage_select(self.groups, s, higher_is_better=True)
        self.assertEqual(d["tuned"], {"linear": "lin_a", "tree": "tree_a", "forest": "forest_a"})
        self.assertEqual(d["lineup"], ["lin_plain", "lin_a", "tree_plain", "tree_a", "forest_a"])
        self.assertEqual(chosen, "forest_a")  # beats everything by ~0.09 in every window

    def test_ineligible_configs_are_skipped(self):
        s = {"lin_a": [30.0, 31.0, 32.0], "tree_plain": [28.5, 31.5, 29.0], "forest_b": [28.0, 30.0, 29.5]}
        chosen, d = two_stage_select(self.groups, s, higher_is_better=False)
        self.assertEqual(d["lineup"], ["lin_a", "tree_plain", "forest_b"])
        self.assertEqual(d["best"], "forest_b")
        self.assertEqual(chosen, "tree_plain")  # 0.5 days behind, SE 0.58: within one SE; lin_a (1.83, SE 0.44) is not

    def test_no_eligible_raises(self):
        with self.assertRaises(ValueError):
            two_stage_select(self.groups, {}, higher_is_better=True)


if __name__ == "__main__":
    unittest.main()
