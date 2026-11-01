"""Va devit.py: cho phep MO dau mang de xuat vung (RPN head) khi huan luyen - cau hinh C0 (TN12-K1).

Suy luan pham vi: dong "Ours + FT RPN" trong phu luc [1] co +18,1 trieu tham so, hon cac cau hinh khac
~9,4 trieu = dung dau RPN C4 (conv 3x3, 1024 kenh: 1024*1024*9). Vi vay chi mo offline_proposal_generator;
ResNet-50 offline van giu nguyen. [CAN KIEM: pham vi mo that trong ma khong cong bo cua [1]].

Khi model._cdfsod_mo_rpn = True va dang huan luyen: RPN o che do train, nhan hop that, tra mat mat
(loss_rpn_cls, loss_rpn_loc) duoc cong vao loss_dict. Khong dat thuoc tinh -> TRUNG ban goc. Idempotent.
"""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/detectron2/modeling/meta_arch/devit.py")
DAU = "# [CDFSOD-VA] mo dau RPN"
src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
else:
    cu = ("        with torch.no_grad():\n"
          "            if self.offline_backbone.training or self.offline_proposal_generator.training:\n"
          "                self.offline_backbone.eval()\n"
          "                self.offline_proposal_generator.eval()\n"
          "\n"
          "            images = self.offline_preprocess_image(batched_inputs)\n"
          "            features = self.offline_backbone(images.tensor)\n"
          "            proposals, _ = self.offline_proposal_generator(images, features, None)\n")
    assert src.count(cu) == 1, "khong tim thay dung mot khoi RPN offline"
    moi = (f"        {DAU}\n"
           "        _mo_rpn = getattr(self, '_cdfsod_mo_rpn', False) and self.training\n"
           "        with torch.no_grad():\n"
           "            if self.offline_backbone.training:\n"
           "                self.offline_backbone.eval()\n"
           "            if (not _mo_rpn) and self.offline_proposal_generator.training:\n"
           "                self.offline_proposal_generator.eval()\n"
           "\n"
           "            images = self.offline_preprocess_image(batched_inputs)\n"
           "            features = self.offline_backbone(images.tensor)\n"
           "        if _mo_rpn:\n"
           "            self.offline_proposal_generator.train()\n"
           "            _gt_rpn = [x['instances'].to(self.device) for x in batched_inputs]\n"
           "            proposals, _mat_mat_rpn = self.offline_proposal_generator(images, features, _gt_rpn)\n"
           "            loss_dict.update(_mat_mat_rpn)\n"
           "        else:\n"
           "            with torch.no_grad():\n"
           "                proposals, _ = self.offline_proposal_generator(images, features, None)\n"
           "        with torch.no_grad():\n")
    src = src.replace(cu, moi)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va xong")
