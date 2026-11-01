"""Va devit.py: khi SUY LUAN, chay khoi hoi quy khung theo tung nhom vung ung vien.

Ly do: ROIAlign hoi quy 20x20 (K2=400) x ~1000 vung x so lop x 256 kenh -> tensor trung gian
11-14 GB, vuot card 16 GB (ban goc chay RTX 3090 24 GB).
Tinh dung: khoi hoi quy tinh doc lap theo tung vung (ROIAlign, embedding, Conv2d + BatchNorm2d o
che do eval, r2c, get_deltas); NMS nam SAU khoi nay -> ghep cac nhom cho cung ket qua (sai khac so hoc
nho do thuat toan cuDNN chon theo kich thuoc lo; do bang ma_nguon/kiem_chia_lo.py).
Chi bat khi dat CDFSOD_CHIA_LO_HOI_QUY=<so vung moi nhom>. Nhanh huan luyen KHONG doi.
Chay lai nhieu lan an toan (kiem dau hieu truoc khi va).
"""
import os
import subprocess
import sys

REPO = os.path.expanduser("~/CDFSOD-benchmark")
TEP = os.path.join(REPO, "detectron2/modeling/meta_arch/devit.py")
DAU = "# [CDFSOD-VA] chia lo hoi quy khi suy luan"

src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
    sys.exit(0)
if "[CDFSOD-" in src:  # ban va tam cu -> khoi phuc ban goc truoc
    subprocess.run(["git", "-C", REPO, "checkout", "--", "detectron2/modeling/meta_arch/devit.py"], check=True)
    src = open(TEP, encoding="utf-8").read()

PHUONG_THUC = f'''
    {DAU}
    def _cdfsod_hoi_quy_suy_luan(self, patch_tokens, rois, H, W, class_weights, class_indices,
                                 sample_class_enabled, num_active_classes):
        """Chep dung nhanh suy luan cua khoi hoi quy trong forward(); tra ve pred_deltas (N, C*4)."""
        reg_bs = len(rois)
        aug_rois, pred_roi_mask, _, _ = augment_rois(rois[:, 1:], None, img_h=H, img_w=W,
                                                     pooler_size=self.reg_roialign_size,
                                                     min_expansion=0.4, expand_shortest=True)
        aug_rois = torch.cat([rois[:, :1], aug_rois], dim=1)
        aroi_feats = self.reg_roi_align(patch_tokens, aug_rois).flatten(2)  # N x C x K2
        if self.set_cd_vito:
            bg_aroi_feats = aroi_feats.transpose(-2, -1) @ self.bg_tokens.weight.t()
        else:
            bg_aroi_feats = aroi_feats.transpose(-2, -1) @ self.bg_tokens.T
        bg_aroi_emb = self.reg_bg_dist_emb(bg_aroi_feats)
        fg_aroi_feats = aroi_feats.transpose(-2, -1) @ class_weights.T
        K2 = self.reg_roialign_size ** 2
        fg_aroi_dist_feats = torch.gather(fg_aroi_feats, 2, class_indices[:, None, :].repeat(1, K2, 1)) \\
            if sample_class_enabled else fg_aroi_feats
        fg_aroi_emb = distance_embed(fg_aroi_dist_feats.flatten(0, 1), num_pos_feats=self.Tpos_emb)
        fg_aroi_emb = self.reg_intra_dist_emb(fg_aroi_emb)
        fg_aroi_emb = fg_aroi_emb.reshape(reg_bs, K2, num_active_classes, -1)
        fg_aroi_emb = fg_aroi_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(
            reg_bs * num_active_classes, -1, self.reg_roialign_size, self.reg_roialign_size)
        bg_aroi_emb = bg_aroi_emb.permute(0, 2, 1).reshape(
            reg_bs, self.Temb, self.reg_roialign_size, self.reg_roialign_size)[:, None, :, :, :].repeat(
            1, num_active_classes, 1, 1, 1).flatten(0, 1)
        embedding = torch.cat([fg_aroi_emb, bg_aroi_emb], dim=1)
        pred_roi_mask = pred_roi_mask[:, None, :, :].repeat(1, num_active_classes, 1, 1).flatten(0, 1)
        masks = [pred_roi_mask[:, None, :, :].float(), ]
        aug_rois = aug_rois[:, None, :].repeat(1, num_active_classes, 1).flatten(0, 1)
        for rp, rp_out in [(self.rp1, self.rp1_out), (self.rp2, self.rp2_out), (self.rp3, self.rp3_out),
                           (self.rp4, self.rp4_out), (self.rp5, self.rp5_out)]:
            embedding = rp(torch.cat([embedding, torch.cat(masks, dim=1)], dim=1))
            pred_mask_logits = rp_out(embedding) / 0.1
            masks.insert(0, pred_mask_logits.sigmoid())
        pred_region_coords = self.r2c(pred_mask_logits)
        pred_abs_boxes = region_coord_2_abs_coord(aug_rois[:, 1:], pred_region_coords, self.reg_roialign_size)
        pred_deltas = self.box2box_transform.get_deltas(
            rois[:, None, 1:].repeat(1, num_active_classes, 1).flatten(0, 1), pred_abs_boxes)
        return pred_deltas.reshape(reg_bs, num_active_classes, 4).flatten(1)

'''

cu_forward = "    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):\n"
cu_dieu_kien = "        if (self.training and (not self.only_train_mask)) or (not self.training):\n            H, W = images.tensor.shape[2:]\n"
assert src.count(cu_forward) == 1 and src.count(cu_dieu_kien) == 1

moi_dieu_kien = '''        _co_lo_hq = int(os.environ.get("CDFSOD_CHIA_LO_HOI_QUY", "0"))
        if (not self.training) and _co_lo_hq > 0 and len(rois) > _co_lo_hq:
            H, W = images.tensor.shape[2:]
            pred_deltas = torch.cat([
                self._cdfsod_hoi_quy_suy_luan(
                    patch_tokens, rois[s:s + _co_lo_hq], H, W, class_weights,
                    class_indices[s:s + _co_lo_hq] if sample_class_enabled else None,
                    sample_class_enabled, num_active_classes)
                for s in range(0, len(rois), _co_lo_hq)], dim=0)
        elif (self.training and (not self.only_train_mask)) or (not self.training):
            H, W = images.tensor.shape[2:]
'''
src = src.replace(cu_forward, PHUONG_THUC + cu_forward).replace(cu_dieu_kien, moi_dieu_kien)
if "\nimport os\n" not in src:
    src = "import os\n" + src
open(TEP, "w", encoding="utf-8").write(src)
print("da va xong")
