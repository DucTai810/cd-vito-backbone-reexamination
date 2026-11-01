"""Suy ra quy tac lay mau cua tep few-shot chinh thuc (s0): so hop moi anh trong tep shot
so voi so hop cua chinh anh do trong train.json."""
import json
import os
from collections import Counter

ROOT = os.path.expanduser("~/cdfsod_data")
for ds in ["ArTaxOr", "clipart1k", "NEUDET", "DIOR", "UODD", "FISH"]:
    tr = json.load(open(f"{ROOT}/{ds}/annotations/train.json"))
    hop_train = Counter(a["image_id"] for a in tr["annotations"])
    ten2id = {im["file_name"]: im["id"] for im in tr["images"]}
    for K in (1, 5, 10):
        s = json.load(open(f"{ROOT}/{ds}/annotations/{K}_shot.json"))
        hop_shot = Counter(a["image_id"] for a in s["annotations"])
        id_s2ten = {im["id"]: im["file_name"] for im in s["images"]}
        mot_hop = sum(1 for i in hop_shot if hop_shot[i] == 1)
        bi_bo = sum(1 for i, n in hop_shot.items() if hop_train.get(ten2id.get(id_s2ten[i]), 0) > n)
        print(f"{ds:9s} {K:2d}-shot: {len(s['images'])} anh, {len(s['annotations'])} hop, "
              f"anh chi 1 hop: {mot_hop}, anh co hop bi bo so voi train: {bi_bo}")
