"""Kiem cac lan lay mau tu sinh da duoc dang ky va doc duoc."""
import os
import sys

os.environ.setdefault("DETECTRON2_DATASETS", os.path.expanduser("~/cdfsod_data/"))
sys.path.insert(0, os.path.expanduser("~/CDFSOD-benchmark"))
import detectron2.data.datasets.builtin  # noqa: E402,F401
from detectron2.data import DatasetCatalog  # noqa: E402

ten = sorted(x for x in DatasetCatalog.list() if "shot_s" in x)
print(len(ten), "lan lay mau da dang ky; vi du:", ten[:3])
for t in ["NEUDET_10shot_s3", "ArTaxOr_30shot_s0", "DIOR_10shot_s4"]:
    d = DatasetCatalog.get(t)
    print(t, "->", len(d), "anh,", sum(len(x["annotations"]) for x in d), "hop")
