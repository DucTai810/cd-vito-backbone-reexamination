"""Va devit.py: cho phep MO phan trich dac trung (DINOv2) khi huan luyen va chon che do so hoc.

Ban goc (dong ~1166): DINOv2 chay trong torch.no_grad() + autocast fp16 -> khong the cap nhat.
Sau va: neu model._cdfsod_mo_pttd = True va dang huan luyen -> chay voi gradient;
che do so hoc lay tu model._cdfsod_so_hoc in {"fp16","bf16","fp32"} (mac dinh "fp16" = nhu goc).
Khi khong dat thuoc tinh nao, hanh vi TRUNG voi ban goc. Idempotent.
"""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/detectron2/modeling/meta_arch/devit.py")
DAU = "# [CDFSOD-VA] mo phan trich dac trung"
src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
else:
    cu = ("        with torch.no_grad():\n"
          "            if self.backbone.training: self.backbone.eval()\n"
          "            with autocast(enabled=True):\n"
          "                all_patch_tokens = self.backbone(images.tensor)\n")
    assert src.count(cu) == 1, "khong tim thay dung mot khoi truyen xuoi DINOv2"
    moi = (f"        {DAU}\n"
           "        _mo_pttd = getattr(self, '_cdfsod_mo_pttd', False) and self.training\n"
           "        _so_hoc = getattr(self, '_cdfsod_so_hoc', 'fp16')\n"
           "        _kieu = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(_so_hoc, torch.float16)\n"
           "        with (torch.enable_grad() if _mo_pttd else torch.no_grad()):\n"
           "            if self.backbone.training: self.backbone.eval()\n"
           "            with torch.autocast('cuda', enabled=(_so_hoc != 'fp32'), dtype=_kieu):\n"
           "                all_patch_tokens = self.backbone(images.tensor)\n")
    src = src.replace(cu, moi)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va xong")
