"""Va detectron2/data/datasets/builtin.py: dang ky them moi tep {K}_shot_s{k}.json co trong annotations
duoi ten {bo}_{K}shot_s{k}. Idempotent."""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/detectron2/data/datasets/builtin.py")
DAU = "# [CDFSOD-VA] dang ky lan lay mau tu sinh"
src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
else:
    cu = "def register_all_CD(root):\n"
    assert src.count(cu) == 1
    moi = f'''def register_all_CD(root):
    {DAU}
    import glob as _glob, re as _re
    for _ds in datasets_name:
        for _f in sorted(_glob.glob(os.path.join(root, _ds, "annotations", "*_shot_s*.json"))):
            _m = _re.match(r"(\\d+)_shot_s(\\d+)\\.json", os.path.basename(_f))
            if _m:
                _PREDEFINED_CD.append((f"{{_ds}}_{{_m.group(1)}}shot_s{{_m.group(2)}}", f"{{_ds}}/train",
                                       f"{{_ds}}/annotations/{{os.path.basename(_f)}}"))
'''
    src = src.replace(cu, moi)
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va xong")
