"""P3: tao tap con co dinh 200 anh test cho moi bo chon tham so va dang ky {bo}_test_p3.
Chi dung de ve quy dao AP theo buoc huan luyen; AP cuoi cung luon cham tren tap test day du."""
import json
import os
import random

ROOT = os.path.expanduser("~/cdfsod_data")
for ds in ["ArTaxOr", "clipart1k", "NEUDET"]:
    t = json.load(open(f"{ROOT}/{ds}/annotations/test.json"))
    ids = sorted(im["id"] for im in t["images"])
    chon = set(random.Random(2026).sample(ids, min(200, len(ids))))
    sub = {"images": [im for im in t["images"] if im["id"] in chon],
           "annotations": [a for a in t["annotations"] if a["image_id"] in chon],
           "categories": t["categories"]}
    json.dump(sub, open(f"{ROOT}/{ds}/annotations/test_p3.json", "w"))
    print(ds, len(sub["images"]), "anh,", len(sub["annotations"]), "hop")

# Va builtin.py: dang ky {bo}_test_p3 (idempotent)
TEP = os.path.expanduser("~/CDFSOD-benchmark/detectron2/data/datasets/builtin.py")
DAU = "# [CDFSOD-VA] dang ky tap con P3"
src = open(TEP, encoding="utf-8").read()
if DAU not in src:
    cu = "def register_all_CD(root):\n"
    moi = (f"def register_all_CD(root):\n    {DAU}\n"
           "    for _ds in datasets_name:\n"
           "        if os.path.exists(os.path.join(root, _ds, 'annotations', 'test_p3.json')):\n"
           "            _PREDEFINED_CD.append((f'{_ds}_test_p3', f'{_ds}/test', f'{_ds}/annotations/test_p3.json'))\n")
    src = src.replace(cu, moi, 1)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va builtin cho P3")
else:
    print("builtin da va P3 tu truoc")
