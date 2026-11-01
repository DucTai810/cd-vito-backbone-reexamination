"""Tinh vector dai dien lop voi SO NGAU NHIEN CO DINH (boc quanh cong cu goc cua CD-ViTO).

Ly do: tools/extract_instance_prototypes.py dung tang cuong ngau nhien (do sang, tuong phan, bao hoa,
lat anh, doi kich thuoc) ma khong dat so ngau nhien -> tep cong bo khong the tai tao chinh xac
(tinh lai s0 NEU-DET: cosine theo thuc the tb 0,84, min 0,58). De tai dung ban tinh lai co dinh so ngau
nhien cho MOI lan lay mau dung trong phep so ghep cap (ke ca s0).

Dung: python tinh_proto_co_dinh.py <ten_bo_dang_ky> <K> <thu_muc_ra> [so_ngau_nhien=0]
"""
import os
import random
import sys

import numpy as np
import torch

R = os.path.expanduser("~/CDFSOD-benchmark")
sys.path.insert(0, os.path.join(R, "tools"))
sys.path.insert(0, R)
os.chdir(R)


def dat_so_ngau_nhien(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


if __name__ == "__main__":
    ten, K, ra = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    s = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mh = sys.argv[5] if len(sys.argv) > 5 else "vitl14"      # vitl14 | vitb14 (TN5)
    os.makedirs(ra, exist_ok=True)
    import extract_instance_prototypes as eip  # noqa: E402
    import run_sinkhorn_cluster as rsc  # noqa: E402
    dat_so_ngau_nhien(s)
    eip.main(model=mh, dataset=ten, use_bbox="yes", epochs=1, out_dir=ra, without_mask=True)
    dat_so_ngau_nhien(s)
    rsc.main(inp=os.path.join(ra, f"{ten}.{mh}.bbox.pkl"), num_prototypes=K, momentum=0.002, epochs=30)
    print("XONG", os.path.join(ra, f"{ten}.{mh}.bbox.p{K}.sk.pkl"))
