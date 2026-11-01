"""Kiem: moi ten lan lay mau da dang ky deu co trong SEEN_CLS_DICT/ALL_CLS_DICT va thu tu lop khop metadata."""
import os
import sys

os.environ.setdefault("DETECTRON2_DATASETS", os.path.expanduser("~/cdfsod_data/"))
sys.path.insert(0, os.path.expanduser("~/CDFSOD-benchmark"))
import detectron2.data.datasets.builtin  # noqa: E402,F401
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT  # noqa: E402

BO = ('ArTaxOr', 'clipart1k', 'DIOR', 'FISH', 'NEUDET', 'UODD')
ten = [x for x in DatasetCatalog.list() if x.startswith(BO) and ("shot_s" in x or x.endswith("_test_p3"))]
thieu = [t for t in ten if t not in SEEN_CLS_DICT or t not in ALL_CLS_DICT]
lech = [t for t in ten if t in SEEN_CLS_DICT and list(MetadataCatalog.get(t).thing_classes) != list(SEEN_CLS_DICT[t])]
print(f"{len(ten)} ten tu sinh; thieu khoa: {thieu}; thu tu lop lech metadata: {lech}")
sys.exit(1 if thieu or lech else 0)
