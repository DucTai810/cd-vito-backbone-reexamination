"""Kiem thu CPU cho cac thanh phan moi cua train_ext.py (LoRA, LLRD)."""
import importlib.util
import os
import sys

import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "te", os.path.join(os.path.dirname(__file__), os.environ.get("TEP_KIEM", "train_ext.py")))
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)
loi = 0


def kiem(ten, dk):
    global loi
    print(("DAT       " if dk else "KHONG DAT ") + ten)
    loi += not dk


torch.manual_seed(0)
goc = nn.Linear(64, 192)
x = torch.randn(3, 10, 64)
y0 = goc(x)
lo = te.LoRAQKV(goc, 4)
kiem("LoRA khoi tao: dau ra trung ban goc", torch.allclose(lo(x), y0, atol=1e-6))
kiem("LoRA giu ten khoa weight/bias", {"weight", "bias"} <= set(lo.state_dict().keys()))
kiem("LoRA: trong so goc bi khoa, chi A/B hoc", (not lo.weight.requires_grad) and lo.A_q.requires_grad and lo.B_v.requires_grad)
# nap trong so goc sau khi boc: khoa cu phai khop
sd = {"weight": torch.ones_like(goc.weight), "bias": torch.zeros_like(goc.bias)}
thieu, du = lo.load_state_dict(sd, strict=False)
kiem("nap state_dict goc sau khi boc: khong du khoa, chi thieu A/B", du == [] and set(thieu) == {"A_q", "B_q", "A_v", "B_v"})
kiem("gia tri nap vao dung", torch.all(lo.weight == 1).item())
# thay doi B -> chi phan q va v doi
with torch.no_grad():
    lo.B_q.fill_(0.1)
y1, y2 = lo(x), nn.functional.linear(x, lo.weight, lo.bias)
kiem("B_q != 0: chi 1/3 dau (q) thay doi", (not torch.allclose(y1[..., :64], y2[..., :64])) and torch.allclose(y1[..., 64:], y2[..., 64:]))
# LLRD
kiem("LLRD h=1 -> 1", te.he_so_llrd("blocks.3.attn.qkv.weight", 24, 1.0) == 1.0)
kiem("LLRD khoi tren cung -> 1", te.he_so_llrd("blocks.23.mlp.fc1.weight", 24, 0.75) == 1.0)
kiem("LLRD khoi 0 -> h^23", abs(te.he_so_llrd("blocks.0.norm1.weight", 24, 0.75) - 0.75 ** 23) < 1e-12)
kiem("LLRD lop nhung -> h^24", abs(te.he_so_llrd("patch_embed.proj.weight", 24, 0.75) - 0.75 ** 24) < 1e-12)
kiem("LLRD chuan hoa cuoi -> 1", te.he_so_llrd("norm.weight", 24, 0.75) == 1.0)
# ap_dung_mo_pttd tren mo hinh gia (ca DUONG MAC DINH - loi 2026-09-25: "khong" bi hieu la "k<N>")
class Khoi(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module(); self.attn.qkv = nn.Linear(8, 24); self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return x


class BBGia(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Khoi() for _ in range(24)]); self.norm = nn.LayerNorm(8)
        self.patch_embed = nn.Linear(8, 8)


class MoHinhGia(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BBGia(); self.offline_proposal_generator = nn.Linear(4, 4)
        for p in self.backbone.parameters():
            p.requires_grad = False


def dem_mo(**cau_hinh):
    te.MO.update({"mo_pttd": "khong", "lora": 0, "khoa_khoi": [], "mo_rpn": False, "checkpoint": True})
    te.MO.update(cau_hinh)
    m = MoHinhGia()
    n = te.ap_dung_mo_pttd(m)
    return n, m


n, m = dem_mo()
kiem("mac dinh 'khong': khong mo gi, khong loi", n == 0 and not getattr(m, "_cdfsod_mo_pttd", False))
n, m = dem_mo(mo_pttd="toan_bo")
kiem("toan_bo: mo moi tham so phan trich dac trung", n == sum(p.numel() for p in m.backbone.parameters()))
n, m = dem_mo(mo_pttd="k4")
kiem("k4: chi 4 khoi cuoi + chuan hoa cuoi", n == 4 * sum(p.numel() for p in m.backbone.blocks[0].parameters())
     + sum(p.numel() for p in m.backbone.norm.parameters()))
n, m = dem_mo(mo_pttd="toan_bo", khoa_khoi=[0, 5])
kiem("khoa khoi 0,5: hai khoi do khong mo", not any(p.requires_grad for p in m.backbone.blocks[5].parameters()))
n, m = dem_mo(lora=4)
kiem("lora: chi A/B hoc, qkv goc khoa", n == 24 * 2 * (4 * 8 + 8 * 4) and not m.backbone.blocks[0].attn.qkv.weight.requires_grad)
n, m = dem_mo(mo_rpn=True)
kiem("mo_rpn: dau RPN mo, phan trich dac trung giu nguyen", n == 0 and all(p.requires_grad for p in m.offline_proposal_generator.parameters()))
print("TONG KET:", "TAT CA DAT" if loi == 0 else f"{loi} KHONG DAT")
sys.exit(1 if loi else 0)
