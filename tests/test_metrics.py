from __future__ import annotations

import unittest

from trace_map.metrics import (
    brier_score,
    recovery_time,
    spearman_correlation,
    useful_at_k,
)


class MetricTests(unittest.TestCase):
    def test_brier(self):
        self.assertAlmostEqual(brier_score([0.9, 0.2], [1, 0]), 0.025)

    def test_spearman(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [30, 20, 10]), -1.0)

    def test_useful_at_k(self):
        self.assertAlmostEqual(useful_at_k([0.2, -0.1, 0.3, 0.0]), 0.5)

    def test_recovery(self):
        series = [10, 10, 10, 4, 6, 9.5, 9.6]
        self.assertEqual(recovery_time(series, shift_step=3, window=2, recovery_fraction=0.9), 2)


if __name__ == "__main__":
    unittest.main()
