"""Va chan doan: in kich thuoc khoi hoi quy khi suy luan (chi khi CDFSOD_IN_KICH_THUOC=1). Idempotent."""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/detectron2/modeling/meta_arch/devit.py")
DAU = "# [CDFSOD-DO] in kich thuoc"
src = open(TEP, encoding="utf-8").read()
if DAU not in src:
    cu = "            aroi_feats = self.reg_roi_align(patch_tokens, aug_rois)  # N x C x K x K\n"
    assert src.count(cu) == 1
    moi = cu + (f"            {DAU}\n"
                "            if (not self.training) and os.environ.get('CDFSOD_IN_KICH_THUOC'):\n"
                "                print('[KICH THUOC] reg_bs', reg_bs, 'num_active_classes', num_active_classes,\n"
                "                      'aroi_feats', tuple(aroi_feats.shape), 'patch_tokens', tuple(patch_tokens.shape),\n"
                "                      'cap phat GiB', round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)\n")
    src = src.replace(cu, moi)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va")
else:
    print("da va tu truoc")
