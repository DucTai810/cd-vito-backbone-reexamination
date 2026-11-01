"""Cham cac bien the trong so (khong huan luyen them) - TN9 va WiSE-FT (TN13).

  Quy trach nhiem theo khoi (TN9):
    python danh_gia_bien_the.py dat_lai --cfg <yaml> --w <model_final D> --ra <tsv> [--khoi 0-23] [opts...]
    -> voi moi khoi b: dat lai backbone.blocks.b ve trong so goc (vitl_0089999.pth), cham, ghi AP.
       Dong 'khong' = trong so D nguyen ven (moc).
  WiSE-FT:
    python danh_gia_bien_the.py wise --cfg <yaml> --w <D> --wA <model_final A> --ra <tsv> [--alpha 0.25,0.5,0.75] [opts]
    -> theta(alpha) = (1-alpha)*theta_A + alpha*theta_D cho moi tham so co trong ca hai, cham.
Cham theo dung duong cham cua cac luot khac: fp16, chia lo hoi quy (CDFSOD_CHIA_LO_HOI_QUY), tran VRAM.
opts detectron2 (vd DATASETS.TEST, DE.CLASS_PROTOTYPES) dat sau dau '--'.
"""
import argparse
import copy
import os
import sys

REPO = os.path.expanduser("~/CDFSOD-benchmark")
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, REPO); os.chdir(REPO)
os.environ.setdefault("DETECTRON2_DATASETS", os.path.expanduser("~/cdfsod_data/"))
os.environ.setdefault("CDFSOD_CHIA_LO_HOI_QUY", "256")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402
import train_net as tn  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402

GOC = os.path.join(REPO, "weights/trained/few-shot/vitl_0089999.pth")


def nap(p):
    d = torch.load(os.path.expanduser(p), map_location="cpu", weights_only=False)
    return d.get("model", d)


def dung_cfg(tep, opts):
    cfg = get_cfg()
    cfg.merge_from_file(tep)
    cfg.merge_from_list(["DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"] + opts)
    cfg.DE.CONTROLLER = False
    cfg.OUTPUT_DIR = "/tmp/danh_gia_bien_the"
    cfg.freeze()
    return cfg


def cham(cfg, model, sd):
    model.load_state_dict(sd, strict=False)
    model._cdfsod_so_hoc = "fp16"
    model.eval()
    kq = tn.Trainer.test(cfg, model)
    return kq["bbox"]["AP"], kq["bbox"]["AP50"]


if __name__ == "__main__":
    if "--" in sys.argv:
        i = sys.argv.index("--"); opts = sys.argv[i + 1:]; sys.argv = sys.argv[:i]
    else:
        opts = []
    ap = argparse.ArgumentParser()
    ap.add_argument("che_do", choices=["dat_lai", "wise"])
    ap.add_argument("--cfg", required=True); ap.add_argument("--w", required=True); ap.add_argument("--wA")
    ap.add_argument("--ra", required=True); ap.add_argument("--khoi", default="0-23")
    ap.add_argument("--alpha", default="0.25,0.5,0.75")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CDFSOD_TRAN_VRAM", "0.88")))
    cfg = dung_cfg(a.cfg, opts)
    model = tn.Trainer.build_model(cfg)
    sd_D = nap(a.w)
    tep = open(a.ra, "a")
    if a.che_do == "dat_lai":
        sd_goc = nap(GOC)
        dau, cuoi = (int(x) for x in a.khoi.split("-"))
        ap0 = cham(cfg, model, sd_D)
        tep.write(f"khong\t{ap0[0]:.4f}\t{ap0[1]:.4f}\n"); tep.flush()
        for b in range(dau, cuoi + 1):
            sd = dict(sd_D)
            tien_to = f"backbone.blocks.{b}."
            n = 0
            for k, v in sd_goc.items():
                if k.startswith(tien_to) and k in sd:
                    sd[k] = v; n += 1
            assert n > 0, f"khong tim thay tham so cua khoi {b}"
            r = cham(cfg, model, sd)
            tep.write(f"khoi_{b:02d}\t{r[0]:.4f}\t{r[1]:.4f}\n"); tep.flush()
            print(f"[TN9] dat lai khoi {b}: AP {r[0]:.3f} (moc {ap0[0]:.3f})", flush=True)
    else:
        sd_A = nap(a.wA)
        for al in (float(x) for x in a.alpha.split(",")):
            sd = {}
            for k, vD in sd_D.items():
                vA = sd_A.get(k)
                if vA is not None and torch.is_tensor(vD) and vD.is_floating_point() and vA.shape == vD.shape:
                    sd[k] = (1 - al) * vA.float() + al * vD.float()
                else:
                    sd[k] = vD
            r = cham(cfg, model, sd)
            tep.write(f"alpha_{al}\t{r[0]:.4f}\t{r[1]:.4f}\n"); tep.flush()
            print(f"[WiSE-FT] alpha {al}: AP {r[0]:.3f}", flush=True)
    tep.close()
