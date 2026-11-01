"""Sinh cac lan lay mau few-shot tu sinh (Muc 5.4 de cuong), theo DUNG quy tac cua tep chinh thuc s0.

Quy tac (suy ra tu tep chinh thuc bang ma_nguon/kiem_quy_tac_s0.py va DATASETS.md
"make sure always to be one annotation in a picture"):
  moi anh duoc chon dong gop MOT hop cua lop c;
  uu tien anh chi co dung mot hop trong train (khong bo hop nao) - nhu NEU-DET/ArTaxOr/FISH chinh thuc;
  thieu thi lay mot hop tu anh nhieu hop (cac hop khac cua anh bi bo) - nhu DIOR/UODD/clipart1k chinh thuc;
  van thieu (vd lop 8 clipart1k chi co 7 hop) thi lay them hop tu anh da chon; ghi chu.
So ngau nhien co dinh theo (K, lan lay mau).
Anh trung ten trong train.json (ArTaxOr co 2 truong hop) chi giu ban dau tien.
s0 = tep chinh thuc (khong sinh lai). Ghi {K}_shot_s{k}.json va bao cao kiem tra.
"""
import json
import os
import random
from collections import defaultdict

ROOT = os.path.expanduser("~/cdfsod_data")
KE_HOACH = {
    # bo: {muc shot: [cac lan lay mau]}
    "ArTaxOr":   {10: range(1, 10), 5: [1, 2], 1: [1, 2], 30: [0, 1, 2]},
    "clipart1k": {10: range(1, 10), 5: [1, 2], 1: [1, 2]},        # 30-shot khong kha thi
    "NEUDET":    {10: range(1, 10), 5: [1, 2], 1: [1, 2], 30: [0, 1, 2]},
    "DIOR":      {10: range(1, 5)},
    "UODD":      {10: range(1, 5)},
    "FISH":      {10: range(1, 5)},
}


def sinh(train, K, seed):
    rng = random.Random(1000 * K + seed)
    anh_theo_ten, anh = {}, {}
    for im in train["images"]:
        if im["file_name"] not in anh_theo_ten:
            anh_theo_ten[im["file_name"]] = im["id"]
            anh[im["id"]] = im
    hop_theo_anh_lop = defaultdict(list)
    so_hop_anh = defaultdict(int)
    for a in train["annotations"]:
        if a["image_id"] in anh and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            hop_theo_anh_lop[(a["image_id"], a["category_id"])].append(a)
            so_hop_anh[a["image_id"]] += 1
    chon_hop, ghi_chu = [], []
    for cat in sorted(c["id"] for c in train["categories"]):
        anh_lop = sorted(i for (i, c) in hop_theo_anh_lop if c == cat)
        don = [i for i in anh_lop if so_hop_anh[i] == 1]          # anh chi co 1 hop
        nhieu = [i for i in anh_lop if so_hop_anh[i] > 1]
        rng.shuffle(don)
        rng.shuffle(nhieu)
        lay = []
        for i in don + nhieu:                                       # moi anh MOT hop
            if len(lay) == K:
                break
            lay.append(rng.choice(hop_theo_anh_lop[(i, cat)]))
        if len(lay) < K:                                            # truong hop hiem: lay them hop
            da = {id(h) for h in lay}
            con = [h for i in nhieu for h in hop_theo_anh_lop[(i, cat)] if id(h) not in da]
            rng.shuffle(con)
            lay += con[:K - len(lay)]
        chon_hop.extend(lay)
        if len(lay) < K:
            ghi_chu.append(f"lop {cat}: chi co {len(lay)}/{K} hop")
        elif len(don) < K:
            ghi_chu.append(f"lop {cat}: {K - min(len(don), K)} hop lay tu anh nhieu hop")
    ids = sorted({h["image_id"] for h in chon_hop})
    return {"images": [anh[i] for i in ids], "annotations": chon_hop,
            "categories": train["categories"]}, ghi_chu


if __name__ == "__main__":
    for ds, muc in KE_HOACH.items():
        train = json.load(open(f"{ROOT}/{ds}/annotations/train.json"))
        test_ten = {im["file_name"] for im in json.load(open(f"{ROOT}/{ds}/annotations/test.json"))["images"]}
        for K, seeds in muc.items():
            for s in seeds:
                out = f"{ROOT}/{ds}/annotations/{K}_shot_s{s}.json"
                j, ghi_chu = sinh(train, K, s)
                assert not ({im["file_name"] for im in j["images"]} & test_ten), "ro ri test!"
                json.dump(j, open(out, "w"))
                print(f"{ds} {K}-shot s{s}: {len(j['images'])} anh, {len(j['annotations'])} hop"
                      + (f" | {'; '.join(ghi_chu)}" if ghi_chu else ""))
