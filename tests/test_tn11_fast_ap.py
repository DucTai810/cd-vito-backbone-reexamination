import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments/TN11/queue'))
from ap_nhanh import BangKhop
from tn11_fast_ap import FastAP


class FastAPTests(unittest.TestCase):
    def test_repeated_images_ties_ignored_and_missing_ground_truth(self):
        rng = np.random.default_rng(12)
        table = BangKhop.__new__(BangKhop)
        table.img_ids, table.cat_ids, table.n_iou = list(range(7)), [0, 1, 2], 10
        table.du_lieu = {}
        for k in range(3):
            for i in range(7):
                scores = np.sort(rng.choice([0.2, 0.5, 0.9], size=8))[::-1]
                table.du_lieu[k, i] = scores, rng.random((10, 8)) > 0.6, rng.random((10, 8)) > 0.8, (0 if k == 2 else 12)
        fast = FastAP(table)
        for indices in [None, np.array([2] * 7), np.array([0, 1, 0, 1, 0, 1, 0])] + [rng.integers(0, 7, 7) for _ in range(25)]:
            np.testing.assert_allclose(fast.ap(indices), table.ap(indices), rtol=0, atol=1e-10)

    def test_no_detections(self):
        table = BangKhop.__new__(BangKhop)
        table.img_ids, table.cat_ids, table.n_iou = [0], [0], 10
        table.du_lieu = {(0, 0): (np.array([]), np.empty((10, 0), bool), np.empty((10, 0), bool), 1)}
        self.assertEqual(FastAP(table).ap(), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
