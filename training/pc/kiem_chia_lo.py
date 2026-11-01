"""Kiem tra ban va chia lo cho ket qua trung khit voi ban goc tren vai anh (Bang 15)."""
import os
import sys

sys.argv = [sys.argv[0]]
REPO = os.path.expanduser("~/CDFSOD-benchmark")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import torch  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import build_detection_test_loader  # noqa: E402
import train_net as tn  # noqa: E402

if os.environ.get("TAT_TF32"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("TF32: TAT")
cfg = get_cfg()
cfg.merge_from_file("configs/neu-det/vitl_shot10_neu-det_finetune.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS", os.path.expanduser("~/ket_qua/TN1/neu-det_10shot_s0/model_final.pth"),
                     "DE.OFFLINE_RPN_CONFIG", "configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml"])
cfg.DE.CONTROLLER = False
model = tn.Trainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()
loader = build_detection_test_loader(cfg, "NEUDET_test")
sai_max = 0.0
with torch.no_grad():
    for k, batch in enumerate(loader):
        if k >= 8:
            break
        os.environ["CDFSOD_CHIA_LO_HOI_QUY"] = "0"
        a = model(batch)[0]["instances"].to("cpu")
        a2 = model(batch)[0]["instances"].to("cpu")          # ban goc chay lan 2: do nhieu tu than
        os.environ["CDFSOD_CHIA_LO_HOI_QUY"] = "64"   # nho de chac chan co chia
        b = model(batch)[0]["instances"].to("cpu")

        def lech(x, y):
            if len(x) != len(y):
                return float("inf")
            return max((x.pred_boxes.tensor - y.pred_boxes.tensor).abs().max().item(),
                       (x.scores - y.scores).abs().max().item()) if len(x) else 0.0

        d_goc, d_va = lech(a, a2), lech(a, b)
        same_cls = len(a) == len(b) and bool((a.pred_classes == b.pred_classes).all())
        sai_max = max(sai_max, d_va)
        nhieu_max = max(globals().get("nhieu_max", 0.0), d_goc)
        globals()["nhieu_max"] = nhieu_max
        print(f"anh {k}: {len(a)}/{len(b)} du doan | goc-vs-goc {d_goc:.2e} | goc-vs-va {d_va:.2e} | cung lop {same_cls}")
print("KET LUAN: goc-vs-va lon nhat", sai_max, "| goc-vs-goc lon nhat", globals().get("nhieu_max"))
